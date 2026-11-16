from __future__ import annotations
import csv
import html
from pathlib import Path
import numpy as np
from .common import digest, now, protocol_fingerprint, read_json, read_jsonl, write_json
from .data import verify
from .runner import generation_key, review_key, decision_key, band_rates, load_public
from .science import (aggregate, audited, band, bootstrap_mean, expected_gain,
                      fit_audit, forecast_loss, released_stats, ridge_oof, summaries, utility)
from .store import TaskStore

ARMS = ["M","R","S"]

def average(values):
    return float(np.mean(values))

def direct_use(arm, original_p, y, output):
    """Release frozen answers using the review itself, with no paid audit.

    R keeps its individual-item probabilities; M/S apply each predicted rate
    to the items in that ORIGINAL confidence band. No target labels select p.
    """
    revised = ([r["p"] for r in output["revised_probabilities"]] if arm == "R"
               else audited(original_p,{"rates":output["band_accuracy"]}))
    if len(revised)!=len(original_p):
        raise ValueError("Direct-use review does not match the frozen answers")
    return {"points":utility(revised,y), **released_stats(revised,y)}

def fit_domain_policy(development_rows, c):
    """Fit the plan's domain-only rule on development outcomes, never evaluation."""
    by_domain={}
    expected=c["development_per_domain"]//c["batch_size"]
    for domain in c["domains"]:
        rows=[r for r in development_rows if r["domain"]==domain]
        if len(rows)!=expected:
            by_domain[domain]={"complete":False,"batches":len(rows)}
            continue
        gain=average([r["gross_audit_gain"] for r in rows])
        by_domain[domain]={"complete":True,"batches":len(rows),"mean_development_gross_gain":gain,
                           "actions":{str(cost):"AUDIT" if gain>cost else "TRUST" for cost in c["audit_costs"]}}
    return {"training_split":"development","domains":by_domain,
            "rule":"Audit iff mean development gross gain exceeds price; ties trust.",
            "note":"Pilot performance is in-sample. The rule is locked before evaluation."}

def incremental_information(rows, domains, n_boot, seed):
    if len(rows) < 25 or any(sum(r["domain"] == d for r in rows) < 5 for d in domains):
        return {"status":"insufficient batches for five-fold incremental-information analysis"}
    x=np.array([r["features"] for r in rows])
    x_plus=np.column_stack([x,[r["expected_M"] for r in rows]])
    y=np.array([r["gross_audit_gain"] for r in rows])
    folds=np.zeros(len(rows),dtype=int)
    for d in domains:
        indices=sorted([i for i,r in enumerate(rows) if r["domain"] == d],key=lambda i:digest([seed,rows[i]["batch"],"fold"]))
        for j,i in enumerate(indices):
            folds[i]=j%5
    base=ridge_oof(x,y,folds)
    plus=ridge_oof(x_plus,y,folds)
    delta=(base-y)**2-(plus-y)**2
    # Bootstrap the entire fitted procedure, preserving cluster identity and
    # original fold assignments. No duplicate batch enters another fold.
    rng=np.random.default_rng(seed)
    strata=[np.array([i for i,r in enumerate(rows) if r["domain"] == d and folds[i] == f]) for d in domains for f in range(5)]
    effects=[]
    for _ in range(n_boot):
        draw=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata if len(s)])
        weights=np.bincount(draw,minlength=len(rows))
        b=ridge_oof(x,y,folds,weights=weights)
        p=ridge_oof(x_plus,y,folds,weights=weights)
        effects.append(float(np.average((b-y)**2-(p-y)**2,weights=weights)))
    return {"status":"complete","batches":len(rows),"ridge_penalty":10,"folds":5,
            "baseline_oof_mse":float(np.mean((base-y)**2)),"plus_meta_oof_mse":float(np.mean((plus-y)**2)),
            "mse_improvement":float(delta.mean()),"ci95":np.quantile(effects,[.025,.975]).tolist(),
            "interpretation":"Positive improvement supports additional predictive information; not a latent metacognitive mechanism."}

def score(data_dir, run_dir, phase="pilot", bootstrap=2000):
    run_dir=Path(run_dir)
    study=read_json(run_dir/"study.json")
    c=study["config"]
    manifest=verify(data_dir)
    if digest(manifest)!=digest(study["manifest"]):
        raise ValueError("Scoring dataset differs from recorded study")
    if protocol_fingerprint(c,manifest)!=study["fingerprint"]:
        raise ValueError("Protocol implementation changed since this run; preserve the original before amending")
    if phase=="evaluation" and not (run_dir/"frozen.json").exists():
        raise ValueError("Evaluation analysis requires a prior protocol lock")
    public=load_public(data_dir)
    labels={r["id"]:r["gold"] for r in read_jsonl(Path(data_dir)/"private_labels.jsonl")}
    results=TaskStore(run_dir/"tasks.sqlite").results()
    splits=read_json(Path(data_dir)/"splits.json")
    target_split="development" if phase=="pilot" else "evaluation"
    batches=[b for b in splits["batches"] if b["split"]==target_split]
    answers={i:results[generation_key(i)] for i in public if generation_key(i) in results}
    calibration={}
    for d in c["domains"]:
        ids=splits["domains"][d]["calibration"]
        if all(i in answers for i in ids):
            p=[answers[i]["confidence"] for i in ids]
            y=[int(answers[i]["answer"]==labels[i]) for i in ids]
            calibration[d]={"p":np.asarray(p),"y":np.asarray(y),"mapping":fit_audit(p,y,c["audit_prior_strength"])}
    rows,excluded,curve_records,details=[],[],[],[]
    for b in batches:
        if not all(i in answers for i in b["ids"]):
            excluded.append({"batch":b["id"],"reason":"incomplete/invalid initial answers"})
            continue
        if b["domain"] not in calibration:
            excluded.append({"batch":b["id"],"reason":"incomplete calibration bank"})
            continue
        p=np.array([answers[i]["confidence"] for i in b["ids"]])
        y=np.array([int(answers[i]["answer"]==labels[i]) for i in b["ids"]])
        mapping=calibration[b["domain"]]["mapping"]
        pa=audited(p,mapping)
        raw,ua=utility(p,y),utility(pa,y)
        reps=[0,1] if b["id"] in splits["repeat_batches"] else [0]
        summary=summaries(p)
        row={"batch":b["id"],"domain":b["domain"],"n":len(p),"accuracy":float(y.mean()),
             "raw_points":raw,"audit_gross_points":ua,"gross_audit_gain":ua-raw,
             "audit_changes_acceptance":bool(np.any((p>=.8)!=(pa>=.8))),
             "original_forecast_loss":forecast_loss(p,y,summary["mean_p"]),
             "raw_release":released_stats(p,y),"audited_release":released_stats(pa,y),
             "reviews":{},"policy":{},"direct_use":{}}
        for arm in ARMS:
            estimates=[]
            missing=[]
            for rep in reps:
                output=results.get(review_key(b["id"],arm,rep))
                if output is None:
                    missing.append(rep)
                    continue
                rates=band_rates(arm,output,b["ids"],p)
                estimates.append({"repeat":rep,"rates":rates,"loss":forecast_loss(p,y,rates),
                                  "gain":expected_gain(p,rates,mapping,c["audit_prior_strength"]),
                                  "direct_use":direct_use(arm,p,y,output)})
            if missing:
                row["reviews"][arm]={"complete":False,"missing_repeats":missing}
                continue
            row["reviews"][arm]={"complete":True,"loss":average([x["loss"] for x in estimates]),
                                  "expected_gross_gain":average([x["gain"] for x in estimates]),"estimates":estimates}
            # Score each repeat before averaging: averaging probabilities first
            # would create an unintended ensemble before thresholding.
            row["direct_use"][arm]={k:average([x["direct_use"][k] for x in estimates])
                                    for k in ["points","accepted","wrong_released","coverage"]}
            ds=row["direct_use"][arm]
            ds["accepted_accuracy"]=1-ds["wrong_released"]/ds["accepted"] if ds["accepted"] else None
            for cost in c["audit_costs"]:
                actions=[results.get(decision_key(b["id"],arm,r,cost),{}).get("action") for r in reps]
                if all(a in ["AUDIT","TRUST"] for a in actions):
                    audit_fraction=average([a=="AUDIT" for a in actions])
                    row["policy"][f"{arm}@{cost}"]={"points":raw+audit_fraction*(ua-raw-cost),"audit_fraction":audit_fraction,
                        "accepted":row["raw_release"]["accepted"]*(1-audit_fraction)+row["audited_release"]["accepted"]*audit_fraction,
                        "wrong_released":row["raw_release"]["wrong_released"]*(1-audit_fraction)+row["audited_release"]["wrong_released"]*audit_fraction}
                else:
                    row["policy"][f"{arm}@{cost}"]={"missing":True,"worst_case_points":min(raw,ua-cost),"best_case_points":max(raw,ua-cost)}
                ref_fraction=average([x["gain"]>cost for x in estimates])
                row["policy"][f"{arm}_reference@{cost}"]={"points":raw+ref_fraction*(ua-raw-cost),"audit_fraction":ref_fraction}
        rows.append(row)
        for k,count in enumerate(summary["counts"]):
            if count:
                curve_records.append({"batch":b["id"],"domain":b["domain"],"band":k,"n":count,
                    "original_confidence":summary["mean_p"][k],"actual_accuracy":float(y[band(p)==k].mean()),
                    **{arm:average([e["rates"][k] for e in row["reviews"][arm]["estimates"]]) if row["reviews"][arm]["complete"] else None for arm in ARMS}})
        details.append({"batch":b["id"],"domain":b["domain"],"p":p,"y":y,"row":row})
    paired=[r for r in rows if r["reviews"]["M"]["complete"] and r["reviews"]["R"]["complete"]]
    if phase=="pilot":
        domain_policy=fit_domain_policy(rows,c)
    else:
        domain_policy=read_json(run_dir/"frozen.json").get("domain_only_policy")
        if not domain_policy:
            raise ValueError("Frozen protocol is missing the development-fitted domain-only baseline")
    domains=[r["domain"] for r in paired]
    effect=[r["reviews"]["R"]["loss"]-r["reviews"]["M"]["loss"] for r in paired]
    primary=bootstrap_mean(effect,domains,bootstrap,c["seed"])
    primary["definition"]="Mean L_R minus L_M; positive favors explicit confidence-reliability review. Repeats averaged inside batch."
    primary["conditional_on"]="These domains, questions, model configuration and calibration banks."
    direct_comparison={"paired_batches":len(paired),
        "mean_points":{a:average([r["direct_use"][a]["points"] for r in paired]) if paired else None for a in ["M","R"]},
        "M_minus_R":bootstrap_mean([r["direct_use"]["M"]["points"]-r["direct_use"]["R"]["points"] for r in paired],domains,bootstrap,c["seed"]),
        "note":"No audit fee or additional API calls. Task points do not subtract review API costs; those are metered separately."}
    comparisons={}
    for arm in ARMS:
        available=[r for r in rows if r["reviews"][arm]["complete"]]
        comparisons[arm]={"batches":len(available),"mean_loss":average([r["reviews"][arm]["loss"] for r in available]) if available else None}
    policy={}
    for cost in c["audit_costs"]:
        # Compare M and R on the SAME complete batches, not different supports.
        common=[r for r in paired if not r["policy"].get(f"M@{cost}",{"missing":True}).get("missing") and not r["policy"].get(f"R@{cost}",{"missing":True}).get("missing")]
        n=len(common)
        scored={"M": [r["policy"][f"M@{cost}"]["points"] for r in common],
                "R": [r["policy"][f"R@{cost}"]["points"] for r in common],
                "M_reference": [r["policy"][f"M_reference@{cost}"]["points"] for r in common],
                "R_reference": [r["policy"][f"R_reference@{cost}"]["points"] for r in common],
                "M_direct": [r["direct_use"]["M"]["points"] for r in common],
                "R_direct": [r["direct_use"]["R"]["points"] for r in common],
                "always_audit":[r["audit_gross_points"]-cost for r in common],
                "never_audit":[r["raw_points"] for r in common],"defer_all":[0.0]*n}
        domain_fractions=[]
        if all(domain_policy["domains"][r["domain"]]["complete"] for r in common):
            domain_fractions=[float(domain_policy["domains"][r["domain"]]["actions"][str(cost)]=="AUDIT") for r in common]
            scored["domain_only"]=[r["raw_points"]+f*(r["gross_audit_gain"]-cost) for r,f in zip(common,domain_fractions)]
        if all(r["reviews"]["S"]["complete"] and not r["policy"].get(f"S@{cost}",{"missing":True}).get("missing") for r in common):
            scored["S"]=[r["policy"][f"S@{cost}"]["points"] for r in common]
        release_metrics={}
        for name in scored:
            if name in ["M_direct","R_direct"]:
                arm=name[0]
                accepted=sum(r["direct_use"][arm]["accepted"] for r in common)
                wrong=sum(r["direct_use"][arm]["wrong_released"] for r in common)
            else:
                fractions=(domain_fractions if name=="domain_only" else
                           [0.0 if name=="never_audit" else 1.0 if name=="always_audit" else
                            r["policy"][f"{name}@{cost}"]["audit_fraction"] for r in common] if name!="defer_all" else [])
                accepted=sum(r["raw_release"]["accepted"]*(1-f)+r["audited_release"]["accepted"]*f for r,f in zip(common,fractions))
                wrong=sum(r["raw_release"]["wrong_released"]*(1-f)+r["audited_release"]["wrong_released"]*f for r,f in zip(common,fractions))
            release_metrics[name]={"mean_accepted_per_batch":accepted/n if n else None,
                                   "mean_wrong_released_per_batch":wrong/n if n else None,
                                   "pooled_accepted_accuracy":1-wrong/accepted if accepted else None,
                                   "coverage":accepted/sum(r["n"] for r in common) if n else None}
        policy[str(cost)]={"paired_batches":n,"mean_points":{k:average(v) if n else None for k,v in scored.items()},
            "release_metrics":release_metrics,
            "M_minus_R":bootstrap_mean(np.array(scored["M"])-np.array(scored["R"]),[r["domain"] for r in common],bootstrap,c["seed"]),
            "differences_from_M":{name:bootstrap_mean(np.array(scored["M"])-np.array(v),[r["domain"] for r in common],bootstrap,c["seed"]) for name,v in scored.items() if name!="M"},
            "audit_rates":{a:average([r["policy"][f"{a}@{cost}"]["audit_fraction"] for r in common]) if n else None for a in ["M","R"]}}
    feature_rows=[]
    for r in paired:
        d=next(x for x in details if x["batch"]==r["batch"])
        ps=summaries(d["p"])
        rr=[average([e["rates"][k] for e in r["reviews"]["R"]["estimates"]]) if ps["counts"][k] else 0 for k in range(3)]
        feature_rows.append({"batch":r["batch"],"domain":r["domain"],"gross_audit_gain":r["gross_audit_gain"],
            "expected_M":r["reviews"]["M"]["expected_gross_gain"],
            "features":[float(r["domain"]==x) for x in c["domains"]]+[float(d["p"].mean()),float(d["p"].std())]+
                       [x/len(d["p"]) for x in ps["counts"]]+[x or 0 for x in ps["mean_p"]]+
                       [average([len(public[i]["question"]) for b in batches if b["id"]==r["batch"] for i in b["ids"]])]+rr})
    incremental=incremental_information(feature_rows,c["domains"],bootstrap,c["seed"])
    # Hold native choices fixed while perturbing the audit bank. This is only a
    # sensitivity analysis; no claim about decisions under changed metadata.
    sensitivity=[]
    eligible=[d for d in details if not d["row"]["policy"].get(f"M@{c['primary_cost']}",{"missing":True}).get("missing") and not d["row"]["policy"].get(f"R@{c['primary_cost']}",{"missing":True}).get("missing")]
    if eligible:
        rng=np.random.default_rng(c["seed"]+1)
        for _ in range(bootstrap):
            maps={}
            for dom,cal in calibration.items():
                ix=rng.integers(0,len(cal["p"]),size=len(cal["p"]))
                maps[dom]=fit_audit(cal["p"][ix],cal["y"][ix],c["audit_prior_strength"])
            vals=[]
            for d in eligible:
                gain=utility(audited(d["p"],maps[d["domain"]]),d["y"])-d["row"]["raw_points"]-c["primary_cost"]
                am=d["row"]["policy"][f"M@{c['primary_cost']}"]["audit_fraction"]
                ar=d["row"]["policy"][f"R@{c['primary_cost']}"]["audit_fraction"]
                vals.append((am-ar)*gain)
            sensitivity.append(average(vals))
    incomplete_decisions={}
    for arm in ARMS:
        lo,hi=[],[]
        for r in rows:
            v=r["policy"].get(f"{arm}@{c['primary_cost']}")
            if v and not v.get("missing"):
                lo.append(v["points"]);hi.append(v["points"])
            else:
                lo.append(min(r["raw_points"],r["audit_gross_points"]-c["primary_cost"]))
                hi.append(max(r["raw_points"],r["audit_gross_points"]-c["primary_cost"]))
        incomplete_decisions[arm]={"mean_lower_bound":average(lo) if lo else None,"mean_upper_bound":average(hi) if hi else None,
                                  "note":"Only batches with complete initial answers/calibration; not bounds for omitted initial generations."}
    report={"created":now(),"provider":study["provider"],"data_kind":study["kind"],"phase":phase,
            "fingerprint":study["fingerprint"],"results_hash":digest(results),
            "warning":"MOCK PIPELINE CHECK: NOT LLM RESULTS" if study["provider"]=="mock" else ("Development pilot only; not confirmatory findings" if phase=="pilot" else "Evaluation results under the recorded protocol"),
            "requested_batches":len(batches),"complete_initial_batches":len(rows),"excluded_batches":excluded,
            "review_availability":comparisons,"primary_monitoring_comparison":primary,
            "per_domain_primary":{d:bootstrap_mean([v for v,dom in zip(effect,domains) if dom==d],[d]*sum(x==d for x in domains),bootstrap,c["seed"]) for d in c["domains"]},
            "decision_quality":policy,"incremental_information":incremental,
            "direct_use_comparison":direct_comparison,"domain_only_policy":domain_policy,
            "audit_bank_sensitivity_fixed_choices":{"M_minus_R_range95":np.quantile(sensitivity,[.025,.975]).tolist() if sensitivity else None},
            "invalid_decision_bounds":incomplete_decisions,
            "pilot_checks":{"batches_with_changed_acceptance":sum(r["audit_changes_acceptance"] for r in rows),
                            "net_audit_gains_primary_cost":[r["gross_audit_gain"]-c["primary_cost"] for r in rows],
                            "initial_accuracy":average([r["accuracy"] for r in rows]) if rows else None},
            "batch_results":rows}
    out=run_dir/"reports"/phase
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"results.json",report)
    if curve_records:
        with (out/"calibration_curves.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(curve_records[0]));writer.writeheader();writer.writerows(curve_records)
    render_html(report,out/"report.html",c)
    print(f"Report: {out/'report.html'}\n{report['warning']}\nPrimary paired batches: {len(paired)}; loss advantage R-M: {primary['mean']}",flush=True)
    return report

def render_html(r,path,c):
    def fmt(x):
        return "not available" if x is None else f"{x:.4f}"
    def esc(x):
        return html.escape(str(x))
    parts=["<!doctype html><html><head><meta charset='utf-8'><title>Confidence self-audit results</title><style>body{font:16px/1.5 system-ui;margin:40px auto;max-width:1000px;padding:0 24px;color:#173047}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #cbd6de;text-align:left}.notice{background:#fff2d1;padding:18px}small{color:#52616c}code{background:#edf3f4}h1,h2{line-height:1.2}</style></head><body>",
           "<h1>Confidence self-audit experiment</h1><p class='notice'><strong>"+esc(r["warning"])+"</strong></p>",
           f"<p>{r['requested_batches']} requested batches; {r['complete_initial_batches']} complete initial batches. Costs are task points, not API dollars.</p>"]
    p=r["primary_monitoring_comparison"]
    parts.append(f"<h2>Primary comparison</h2><p>Ordinary reassessment loss minus confidence-reliability review loss: <strong>{fmt(p['mean'])}</strong>; 95% paired bootstrap interval {esc(p['ci95'])}; n={p['batches']} batches. Positive favors reliability review. Repeats are averaged within batches.</p>")
    parts.append("<table><tr><th>Review</th><th>Complete batches</th><th>Mean forecast loss</th></tr>")
    for arm,v in r["review_availability"].items():
        parts.append(f"<tr><td>{arm}</td><td>{v['batches']}</td><td>{fmt(v['mean_loss'])}</td></tr>")
    parts.append("</table><p>M: confidence-reliability review; R: ordinary answer reassessment; S: confidence summaries without question text. Availability can differ; the primary comparison is paired.</p>")
    for cost,data in r["decision_quality"].items():
        parts.append(f"<h2>Decision quality at audit cost {cost}</h2><p>{data['paired_batches']} common batches; {'primary cost' if int(cost)==c['primary_cost'] else 'sensitivity analysis'}.</p><table><tr><th>Policy</th><th>Mean net points</th><th>Coverage</th><th>Wrong answers released per batch</th></tr>")
        for policy,value in data["mean_points"].items():
            metrics=data["release_metrics"][policy]
            parts.append(f"<tr><td>{esc(policy)}</td><td>{fmt(value)}</td><td>{fmt(metrics['coverage'])}</td><td>{fmt(metrics['mean_wrong_released_per_batch'])}</td></tr>")
        parts.append(f"</table><p>M minus R: {fmt(data['M_minus_R']['mean'])}, interval {esc(data['M_minus_R']['ci95'])}.</p>")
    parts.append("<p>M_direct releases using the predicted rate for each original confidence band. R_direct uses each item's revised probability. Both use the same 0.8 threshold, preserve the original answers and incur no audit fee. Review API costs are metered separately, not subtracted from task points. The domain-only rule is fitted on development outcomes; its pilot score is in-sample.</p>")
    parts.append("<h2>Incremental information</h2><pre>"+esc(json_pretty(r["incremental_information"]))+"</pre>")
    parts.append("<h2>Missing outputs and sensitivity</h2><p>"+esc(json_pretty({"excluded_batches":r["excluded_batches"],"bank_sensitivity":r["audit_bank_sensitivity_fixed_choices"],"invalid_decision_bounds":r["invalid_decision_bounds"]}))+"</p>")
    parts.append("<p><a href='results.json'>Full machine-readable results</a> | <a href='calibration_curves.csv'>Calibration-curve records</a></p><small>This measures a behavioral comparison, not a distinct internal third-order mechanism. Intervals condition on these domains and calibration banks; bank resampling is a separate fixed-choice sensitivity check. A single realized audit gain is not a verdict on rationality.</small></body></html>")
    Path(path).write_text("\n".join(parts))

def json_pretty(value):
    import json
    return json.dumps(value,indent=2)
