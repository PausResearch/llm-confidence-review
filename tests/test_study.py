import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from auditstudy import prompts, runner
from auditstudy.analysis import direct_use, fit_domain_policy, score
from auditstudy.common import config, digest, dumps, read_json, read_jsonl, write_json, write_jsonl
from auditstudy.data import fixture_rows, prepare, preserve_option_order, verify
from auditstudy.provider import Client, FatalAPIError
from auditstudy.science import (audited, band, bootstrap_mean, expected_gain,
                                fit_audit, forecast_loss, ridge_oof, utility)
from auditstudy.store import BudgetExceeded, Ledger, TaskStore


class ScienceTests(unittest.TestCase):
    def test_direct_use_preserves_r_item_resolution_and_original_bands(self):
        p=[.9,.9,.7,.7];y=[1,0,1,0]
        r={"revised_probabilities":[{"id":str(i),"p":q} for i,q in enumerate([.99,.61,.8,.79])]}
        m={"band_accuracy":[None,.85,.7]}
        individual=direct_use("R",p,y,r)
        grouped=direct_use("M",p,y,m)
        self.assertEqual(individual["points"],2)
        self.assertEqual(individual["accepted"],2)
        self.assertEqual(individual["wrong_released"],0)
        self.assertEqual(grouped["points"],-3)
        self.assertEqual(grouped["wrong_released"],1)
        self.assertEqual(p,[.9,.9,.7,.7])

    def test_domain_rule_uses_mean_gain_and_trusts_on_ties(self):
        c=config();c.update(domains=["biology"],development_per_domain=40,batch_size=20)
        fitted=fit_domain_policy([{"domain":"biology","gross_audit_gain":2},
                                 {"domain":"biology","gross_audit_gain":4}],c)
        self.assertEqual(fitted["domains"]["biology"]["actions"],{"1":"AUDIT","3":"TRUST","6":"TRUST"})
    def test_probability_boundaries_and_rejection(self):
        self.assertEqual(band([0,.599999,.6,.799999,.8,1]).tolist(),[0,0,1,1,2,2])
        for bad in [float("nan"),float("inf"),-.1,1.1]:
            with self.assertRaises(ValueError):
                band([bad])

    def test_same_accuracy_can_require_opposite_audit_decisions(self):
        p=[.7]*10+[.9]*10
        aligned=[1]*7+[0]*3+[1]*9+[0]
        reversed_y=[1]*9+[0]+[1]*7+[0]*3
        self.assertEqual(sum(aligned),sum(reversed_y))
        # Separate 120-item calibration banks, as in the study; using the
        # 20 target labels would both leak outcomes and change shrinkage.
        good=fit_audit(p*6,aligned*6)
        bad=fit_audit(p*6,reversed_y*6)
        self.assertEqual(utility(audited(p,good),aligned)-utility(p,aligned)-3,-3)
        self.assertEqual(utility(audited(p,bad),reversed_y)-utility(p,reversed_y)-3,7)
        self.assertAlmostEqual(forecast_loss(p,aligned,[None,.7,.9]),0)
        self.assertGreater(forecast_loss(p,reversed_y,[None,.7,.9]),0)

    def test_audit_uses_calibration_only_and_preserves_empty_bands(self):
        mapping=fit_audit([.9]*4,[0,0,0,0])
        target=[.2,.7,.95]
        self.assertEqual(audited(target,mapping)[:2].tolist(),target[:2])
        self.assertAlmostEqual(audited(target,mapping)[2],9/14)
        self.assertEqual(utility([.799999,.8],[0,1]),1)

    def test_binomial_reference_against_enumerated_calibration_outcomes(self):
        # Enumerate actual labelled calibration banks, fit each and average
        # their decisions. This checks the analytical crossing probability.
        from itertools import product
        calp=[.75]*3+[.95]*2
        target=[.7,.7,.9]
        qs=[None,.6,.9]
        numerical=0
        for labels in product([0,1],repeat=5):
            probability=math.prod((qs[int(k>=3)+1] if y else 1-qs[int(k>=3)+1]) for k,y in enumerate(labels))
            changed=(audited(target,fit_audit(calp,labels,1))>=.8).astype(int)-(np.array(target)>=.8).astype(int)
            numerical+=probability*np.sum(changed*(5*np.array([.6,.6,.9])-4))
        self.assertAlmostEqual(expected_gain(target,qs,fit_audit(calp,[0]*5,1),1),numerical,12)

    def test_held_out_labels_cannot_change_their_own_predictions(self):
        rng=np.random.default_rng(8)
        x=rng.normal(size=(30,4));y=rng.normal(size=30);folds=np.arange(30)%5
        original=ridge_oof(x,y,folds)
        changed=y.copy();changed[folds==0]+=100
        other=ridge_oof(x,changed,folds)
        np.testing.assert_allclose(original[folds==0],other[folds==0])
        self.assertFalse(np.allclose(original[folds==1],other[folds==1]))

    def test_weighted_ridge_matches_augmented_least_squares(self):
        rng=np.random.default_rng(16)
        x=rng.normal(size=(30,4));y=rng.normal(size=30);folds=np.arange(30)%5
        weights=rng.integers(1,4,size=30)
        actual=ridge_oof(x,y,folds,weights=weights)
        tr=folds!=0;te=~tr;w=weights[tr]
        mean=np.average(x[tr],axis=0,weights=w)
        sd=np.sqrt(np.average((x[tr]-mean)**2,axis=0,weights=w))
        z=np.column_stack([np.ones(tr.sum()),(x[tr]-mean)/sd])
        penalty=np.diag([0]+[math.sqrt(10)]*4)
        beta=np.linalg.lstsq(np.row_stack([z*np.sqrt(w)[:,None],penalty]),
                            np.concatenate([y[tr]*np.sqrt(w),np.zeros(5)]),rcond=None)[0]
        expected=np.einsum("ni,i->n",np.column_stack([np.ones(te.sum()),(x[te]-mean)/sd]),beta)
        np.testing.assert_allclose(actual[te],expected,atol=1e-10)

    def test_bootstrap_uses_batch_units_and_handles_missing_sample(self):
        self.assertEqual(bootstrap_mean([],[],10)["batches"],0)
        r=bootstrap_mean([2,2,2,2],["a","a","b","b"],20)
        self.assertEqual(r["ci95"],[2,2])


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.item={"id":"q1","question":"Which animal?","options":["dog","cat"],
                   "gold":"SECRET_GOLD","cot_content":"SECRET_SOLUTION","answer_index":123}
        self.answer={"q1":{"answer":"A","confidence":.9}}

    def test_label_fields_never_enter_generation_or_reviews(self):
        for messages in [prompts.generate(self.item)]+[prompts.review(a,[self.item],self.answer,"biology") for a in ["M","R","S"]]:
            text=dumps(messages)
            for secret in ["SECRET_GOLD","SECRET_SOLUTION","answer_index","cot_content"]:
                self.assertNotIn(secret,text)

    def test_review_evidence_matched_and_summary_control_blinded(self):
        m=prompts.review("M",[self.item],self.answer,"biology")
        r=prompts.review("R",[self.item],self.answer,"biology")
        self.assertEqual(json.loads(m[1]["content"].split("\n",1)[1]),json.loads(r[1]["content"].split("\n",1)[1]))
        self.assertNotIn("Which animal?",dumps(prompts.review("S",[self.item],self.answer,"biology")))

    def test_decision_whitelists_unlabelled_calibration_metadata(self):
        c={"n":10,"counts":[0,0,10],"mean_p":[None,None,.9],"rates":"SECRET_AUDITED_RATE","gold":"SECRET"}
        text=dumps(prompts.decide("b",[.9],[None,None,.85],c,3,10))
        self.assertNotIn("SECRET",text)
        self.assertNotIn("Which animal?",text)

    def test_invalid_probabilities_and_duplicate_rows_are_rejected(self):
        for p in [True,float("nan"),float("inf"),1.1,-.1,".8"]:
            with self.assertRaises(ValueError):
                prompts.validate("generate",{"answer":"A","confidence":p},{"letters":["A","B"]})
        with self.assertRaises(ValueError):
            prompts.validate("review_R",{"revised_probabilities":[{"id":"q1","p":.8},{"id":"q1","p":.8}]},{"ids":["q1","q2"]})
        with self.assertRaises(ValueError):
            prompts.validate("review_M",{"band_accuracy":[0,.5,.8]},{"counts":[0,1,1]})


class StorageAndRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.ledger=Ledger(self.root/"ledger.sqlite",1)
        self.store=TaskStore(self.root/"tasks.sqlite")
        self.addCleanup(self.ledger.db.close);self.addCleanup(self.store.db.close)
        self.c=config()

    def test_concurrent_reservations_cannot_exceed_budget(self):
        def reserve(i):
            try:
                return self.ledger.reserve("run",str(i),.26)
            except BudgetExceeded:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids=[i for i in pool.map(reserve,range(20)) if i is not None]
        self.assertEqual(len(ids),3)
        self.ledger.settle(ids[0],status="unknown")
        self.assertAlmostEqual(self.ledger.summary()["conservative_cost_usd"],.78)
        self.ledger.settle(ids[1],.1,{"prompt_tokens":5,"completion_tokens":2})
        self.assertAlmostEqual(self.ledger.summary()["conservative_cost_usd"],.62)

    def test_nonfinite_budget_is_rejected(self):
        for cap in [float("nan"),float("inf"),0,-1]:
            with self.assertRaises(ValueError):
                Ledger(self.root/"unused.sqlite",cap)

    def test_cache_reuse_and_prompt_mutation(self):
        client=Client(self.c,"mock",self.ledger,self.store,self.root)
        messages=[{"role":"user","content":"JSON"}]
        with patch.object(client,"_mock",wraps=client._mock) as response:
            a=client.execute("x","generate",messages,{"letters":["A","B"]},256)
            b=client.execute("x","generate",messages,{"letters":["A","B"]},256)
            self.assertEqual(a,b);self.assertEqual(response.call_count,1)
            with self.assertRaises(FatalAPIError):
                client.execute("x","generate",[{"role":"user","content":"changed"}],{"letters":["A","B"]},256)

    def test_format_retry_is_not_reset_after_interruption(self):
        client=Client(self.c,"mock",self.ledger,self.store,self.root)
        messages=[{"role":"user","content":"JSON"}]
        phash=digest({"messages":messages,"maximum":256,"model":self.c["model"]})
        self.store.put("x","generate","running",phash)
        saved={"prompt_hash":phash,"format_retry":0,"validation_error":"Bad schema",
               "response":{"choices":[{"message":{"content":"{}"}}]}}
        write_json(self.root/"raw"/(digest("x")+"-000.json"),saved)
        bad={"model":"MOCK-NO-LLM","choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}
        with patch.object(client,"_mock",return_value=bad) as response:
            self.assertIsNone(client.execute("x","generate",messages,{"letters":["A","B"]},256))
            self.assertEqual(response.call_count,1)
            self.assertEqual(self.store.get("x")["status"],"invalid")
            self.assertIsNone(client.execute("x","generate",messages,{"letters":["A","B"]},256))
            self.assertEqual(response.call_count,1)

    def test_unknown_interrupted_request_does_not_silently_rebill(self):
        client=Client(self.c,"mock",self.ledger,self.store,self.root)
        messages=[{"role":"user","content":"JSON"}]
        phash=digest({"messages":messages,"maximum":256,"model":self.c["model"]})
        self.store.put("x","generate","running",phash)
        with patch.object(client,"_mock") as response:
            with self.assertRaises(FatalAPIError):
                client.execute("x","generate",messages,{"letters":["A"]},256)
            response.assert_not_called()


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.data=self.root/"data";self.run=self.root/"run"
        self.c=config();self.c.update(domains=["biology","economics"],calibration_per_domain=20,
            development_per_domain=20,evaluation_per_domain=20,batch_size=10,repeat_fraction=.5,workers=2)
        self.quiet=contextlib.redirect_stdout(io.StringIO());self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__,None,None,None)
        self.root_patch=patch.object(runner,"ROOT",self.root);self.root_patch.start();self.addCleanup(self.root_patch.stop)
        prepare(self.c,self.data,"mock")

    def test_disjoint_splits_and_gold_survives_option_shuffle(self):
        public=read_jsonl(self.data/"public_items.jsonl")
        keys={r["id"]:r for r in read_jsonl(self.data/"private_labels.jsonl")}
        ids=[r["id"] for r in public];self.assertEqual(len(ids),len(set(ids)))
        for row in public:
            original_i=int(keys[row["id"]]["source_question_id"].rsplit("-",1)[1])
            chosen=row["options"][ord(keys[row["id"]]["gold"])-65]
            self.assertEqual(chosen,str(original_i%10))
        self.assertTrue(preserve_option_order("Pick one",["yes","All of the above"]))
        self.assertFalse(preserve_option_order("Pick one",["dog","cat"]))

    def test_duplicates_removed_and_position_references_not_shuffled(self):
        rows=fixture_rows(self.c)
        rows[0]["options"][-1]="All of the above"
        rows.append(dict(rows[0],question_id="duplicate"))
        source=self.root/"local.jsonl";write_jsonl(source,rows)
        other=self.root/"other";manifest=prepare(self.c,other,local=source)
        self.assertEqual(manifest["duplicate_questions_removed"],1)
        public=read_jsonl(other/"public_items.jsonl")
        private={r["id"]:r for r in read_jsonl(other/"private_labels.jsonl")}
        referenced=next(r for r in public if "All of the above" in r["options"])
        self.assertEqual(private[referenced["id"]]["option_permutation"],list(range(10)))
        self.assertTrue(private[referenced["id"]]["option_order_preserved"])

    def test_mutated_dataset_and_live_mock_mixing_are_rejected(self):
        with self.assertRaises(ValueError):
            runner.initialize(self.c,self.data,self.run,"deepseek")
        with (self.data/"public_items.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaises(ValueError):
            verify(self.data)

    def test_complete_mock_pipeline_and_matched_missingness(self):
        with self.assertRaises(ValueError):
            runner.run(self.c,self.data,self.run,"mock",phase="evaluation")
        first=runner.run(self.c,self.data,self.run,"mock")
        self.assertFalse(first["errors"])
        self.assertEqual(first["statuses"],{"done":128})
        second=runner.run(self.c,self.data,self.run,"mock")
        self.assertEqual(second["valid_this_run"],0)
        report=score(self.data,self.run,bootstrap=10)
        self.assertEqual(report["primary_monitoring_comparison"]["batches"],4)
        self.assertIn("NOT LLM RESULTS",report["warning"])
        for cost in ["1","3","6"]:
            for name in ["M_direct","R_direct","domain_only"]:
                self.assertIn(name,report["decision_quality"][cost]["mean_points"])
                self.assertIn(name,report["decision_quality"][cost]["differences_from_M"])
        self.assertEqual(report["decision_quality"]["1"]["mean_points"]["R_direct"],
                         report["decision_quality"]["6"]["mean_points"]["R_direct"])
        runner.freeze(self.c,self.data,self.run)
        locked=read_json(self.run/"frozen.json")["domain_only_policy"]
        evaluation=runner.run(self.c,self.data,self.run,"mock",phase="evaluation")
        self.assertFalse(evaluation["errors"])
        full=score(self.data,self.run,phase="evaluation",bootstrap=10)
        self.assertEqual(full["primary_monitoring_comparison"]["batches"],4)
        self.assertEqual(full["domain_only_policy"],locked)
        # Remove one arm only: all comparisons must use paired support.
        store=TaskStore(self.run/"tasks.sqlite");self.addCleanup(store.db.close)
        batch=next(b for b in read_json(self.data/"splits.json")["batches"] if b["split"]=="evaluation")
        key=runner.review_key(batch["id"],"R",0)
        old=store.get(key);store.put(key,"review_R","invalid",old["prompt_hash"],detail="test omission")
        omitted=score(self.data,self.run,phase="evaluation",bootstrap=10)
        self.assertEqual(omitted["primary_monitoring_comparison"]["batches"],3)
        self.assertEqual(omitted["decision_quality"]["3"]["paired_batches"],3)

    def test_direct_use_repeats_score_actions_before_averaging(self):
        runner.run(self.c,self.data,self.run,"mock")
        score(self.data,self.run,bootstrap=5);runner.freeze(self.c,self.data,self.run)
        runner.run(self.c,self.data,self.run,"mock",phase="evaluation")
        splits=read_json(self.data/"splits.json")
        batch=next(b for b in splits["batches"] if b["id"] in splits["repeat_batches"])
        store=TaskStore(self.run/"tasks.sqlite");self.addCleanup(store.db.close)
        for rep,p in [(0,.9),(1,.7)]:
            key=runner.review_key(batch["id"],"R",rep);prior=store.get(key)
            output={"revised_probabilities":[{"id":i,"p":p} for i in batch["ids"]]}
            store.put(key,"review_R","done",prior["prompt_hash"],output)
        result=score(self.data,self.run,phase="evaluation",bootstrap=5)
        row=next(r for r in result["batch_results"] if r["batch"]==batch["id"])
        self.assertEqual(row["direct_use"]["R"]["accepted"],len(batch["ids"])/2)
        self.assertEqual(row["direct_use"]["R"]["coverage"],.5)


if __name__=="__main__":
    unittest.main()
