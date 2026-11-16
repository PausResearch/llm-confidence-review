from __future__ import annotations
import concurrent.futures
import fcntl
from pathlib import Path
from .common import ROOT, code_hash, config, digest, now, protocol_fingerprint, read_json, read_jsonl, write_json
from .data import verify
from . import prompts
from .provider import Client, FatalAPIError
from .science import summaries, aggregate
from .store import Ledger, TaskStore, BudgetExceeded

def load_public(directory):
    return {x["id"]:x for x in read_jsonl(Path(directory)/"public_items.jsonl")}

def generation_key(item_id):
    return "generate:"+item_id

def review_key(batch, arm, repeat=0):
    return f"review:{batch}:{arm}:{repeat}"

def decision_key(batch, arm, repeat, cost):
    return f"decide:{batch}:{arm}:{repeat}:{cost}"

def band_rates(arm, result, ids, p):
    return aggregate(p,[r["p"] for r in result["revised_probabilities"]]) if arm == "R" else result["band_accuracy"]

def initialize(c, data_dir, run_dir, provider):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = verify(data_dir)
    if manifest["preparation_config_hash"] != digest(c):
        raise ValueError("Dataset preparation config differs from current study config")
    if provider != "mock" and manifest["kind"] == "synthetic_mock":
        raise ValueError("Synthetic fixtures cannot be sent to a live provider.")
    fingerprint = protocol_fingerprint(c,manifest)
    path = run_dir/"study.json"
    if path.exists():
        prior=read_json(path)
        if prior["fingerprint"] != fingerprint or prior["provider"] != provider:
            raise ValueError("Study configuration/prompts/math or provider changed. Use a new run; never overwrite a study.")
        if Path(prior["data_dir"]).resolve() != Path(data_dir).resolve():
            raise ValueError("Study is bound to a different data directory")
    else:
        write_json(path,{"created":now(),"config":c,"fingerprint":fingerprint,"code_hash":code_hash(),"provider":provider,
                        "data_dir":str(Path(data_dir).resolve()),"kind":manifest["kind"],"manifest":manifest})
    return manifest

def run(c, data_dir, run_dir, provider, phase="pilot", stage="all", limit=None, cap=None, retry_uncertain=False):
    run_dir=Path(run_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    with (run_dir/"process.lock").open("w") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is running this study")
        initialize(c,data_dir,run_dir,provider)
        if phase == "evaluation":
            frozen=run_dir/"frozen.json"
            if not frozen.exists() or read_json(frozen)["fingerprint"] != read_json(run_dir/"study.json")["fingerprint"]:
                raise ValueError("Run freeze before evaluation, after checking the pilot.")
        public=load_public(data_dir)
        splits=read_json(Path(data_dir)/"splits.json")
        split="development" if phase == "pilot" else "evaluation"
        batches=[b for b in splits["batches"] if b["split"] == split]
        store=TaskStore(run_dir/"tasks.sqlite")
        # This path is shared even when the operator creates another study run.
        ledger=Ledger(ROOT/"runs"/"spend.sqlite",min(cap if cap is not None else c["budget_usd"], c["budget_usd"]))
        client=Client(c,provider,ledger,store,run_dir)
        completed, errors, invalid, scheduled=0,[],0,0

        def process(jobs):
            nonlocal completed,invalid,scheduled
            pending=[]
            for job in jobs:
                prior=store.get(job[0])
                if prior and prior["status"] in ("done","invalid"):
                    continue
                if limit is not None and scheduled >= limit:
                    break
                pending.append(job)
                scheduled+=1
            if not pending:
                return
            print(f"Starting {len(pending)} {pending[0][1]} tasks ({provider}).",flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=c["workers"]) as executor:
                future_jobs={executor.submit(client.execute,*job,retry_uncertain=retry_uncertain):job for job in pending}
                for future in concurrent.futures.as_completed(future_jobs):
                    job=future_jobs[future]
                    try:
                        value=future.result()
                        completed+=int(value is not None)
                        invalid+=int(value is None)
                    except Exception as error:
                        errors.append({"task":job[0],"error":str(error)})
                        if isinstance(error,(FatalAPIError,BudgetExceeded)):
                            client.stop.set()
                    if (completed+invalid+len(errors)) % 20 == 0:
                        print(f"Progress: {completed} valid, {invalid} skipped/invalid, {len(errors)} errors; conservative spend US${ledger.summary()['conservative_cost_usd']:.4f}",flush=True)

        if stage in ("all","generate"):
            # Interleave domains so the first smoke-test calls cover all four.
            domain_ids=[splits["domains"][d]["calibration"]+splits["domains"][d][split] for d in c["domains"]]
            ids=[group[i] for i in range(max(map(len,domain_ids))) for group in domain_ids if i<len(group)]
            jobs=[(generation_key(i),"generate",prompts.generate(public[i]),{"letters":[chr(65+j) for j in range(len(public[i]["options"]))]},c["max_tokens"]["generate"]) for i in ids]
            process(jobs)
        results=store.results()
        answers={i:results[generation_key(i)] for i in public if generation_key(i) in results}
        if stage in ("all","review") and not client.stop.is_set():
            jobs=[]
            for batch in batches:
                if not all(i in answers for i in batch["ids"]):
                    continue
                items=[public[i] for i in batch["ids"]]
                p=[answers[i]["confidence"] for i in batch["ids"]]
                repeats=[0,1] if batch["id"] in splits["repeat_batches"] else [0]
                for repeat in repeats:
                    for arm in ["M","R","S"]:
                        jobs.append((review_key(batch["id"],arm,repeat),"review_"+arm,prompts.review(arm,items,answers,batch["domain"]),
                                     {"ids":batch["ids"],"counts":summaries(p)["counts"]},c["max_tokens"]["review"]))
            process(jobs)
        results=store.results()
        if stage in ("all","decide") and not client.stop.is_set():
            jobs=[]
            for batch in batches:
                cal_ids=splits["domains"][batch["domain"]]["calibration"]
                if not all(i in answers for i in cal_ids+batch["ids"]):
                    continue  # Never silently shrink the calibration bank.
                cs=summaries([answers[i]["confidence"] for i in cal_ids])
                p=[answers[i]["confidence"] for i in batch["ids"]]
                repeats=[0,1] if batch["id"] in splits["repeat_batches"] else [0]
                for repeat in repeats:
                    for arm in ["M","R","S"]:
                        result=results.get(review_key(batch["id"],arm,repeat))
                        if result is None:
                            continue
                        rates=band_rates(arm,result,batch["ids"],p)
                        for cost in c["audit_costs"]:
                            jobs.append((decision_key(batch["id"],arm,repeat,cost),"decide",prompts.decide(batch["id"],p,rates,cs,cost,c["audit_prior_strength"]),{},c["max_tokens"]["decide"]))
            process(jobs)
        summary={"provider":provider,"phase":phase,"valid_this_run":completed,"skipped_or_invalid_this_run":invalid,
                 "errors":errors,"stop_reason":client.stop_reason,"statuses":store.statuses(),"ledger":ledger.summary(),"timestamp":now()}
        write_json(run_dir/("last_"+phase+"_run.json"),summary)
        print(f"Finished invocation. Task states: {summary['statuses']}; estimated cumulative spend US${summary['ledger']['conservative_cost_usd']:.4f}",flush=True)
        if errors:
            print("First error: "+errors[0]["error"],flush=True)
        return summary

def freeze(c,data_dir,run_dir):
    run_dir=Path(run_dir)
    study=read_json(run_dir/"study.json")
    initialize(c,data_dir,run_dir,study["provider"])
    path=run_dir/"frozen.json"
    if path.exists():
        print("Protocol already frozen; preserved original timestamp.")
        return
    results=TaskStore(run_dir/"tasks.sqlite").results()
    splits=read_json(Path(data_dir)/"splits.json")
    required=[]
    for domain in c["domains"]:
        required += [generation_key(i) for s in ["calibration","development"] for i in splits["domains"][domain][s]]
    for batch in splits["batches"]:
        if batch["split"] == "development":
            required += [review_key(batch["id"],a) for a in ["M","R","S"]]
            required += [decision_key(batch["id"],a,0,cost) for a in ["M","R","S"] for cost in c["audit_costs"]]
    missing=[key for key in required if key not in results]
    if missing:
        raise ValueError(f"Pilot incomplete: {len(missing)} required outputs missing/invalid. Inspect before freezing.")
    report_path=run_dir/"reports/pilot/results.json"
    if not report_path.exists():
        raise ValueError("Score and inspect the pilot before locking the protocol")
    report=read_json(report_path)
    if report.get("results_hash")!=digest(results) or report.get("fingerprint")!=study["fingerprint"]:
        raise ValueError("Pilot report is stale; rerun score before freezing")
    if report["pilot_checks"]["batches_with_changed_acceptance"] < 2:
        raise ValueError("Pilot fails the prespecified feasibility gate: fewer than two batches change acceptance after audit. Inspect and document an amendment before evaluation.")
    domain_policy=report.get("domain_only_policy")
    if not domain_policy or not all(domain_policy["domains"][d]["complete"] for d in c["domains"]):
        raise ValueError("Pilot must fit the domain-only rule before evaluation")
    write_json(path,{"timestamp":now(),"fingerprint":study["fingerprint"],"code_hash":code_hash(),"pilot_completed_tasks":len(required),
                    "domain_only_policy":domain_policy,
                    "statement":"Operational protocol lock before evaluation; not external preregistration."})
    print("Protocol locked. Evaluation can now run without changing prompts, model, dataset or scoring.")
