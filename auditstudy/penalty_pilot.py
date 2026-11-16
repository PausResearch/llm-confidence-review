"""Decision-only pilot amendment; never generates or evaluates held-out items."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import html
import json
import sqlite3
from pathlib import Path

import numpy as np

from . import prompts
from .common import ROOT, digest, load_env, now, protocol_fingerprint, read_json, write_json
from .data import verify
from .provider import Client, FatalAPIError
from .runner import band_rates, decision_key, generation_key, review_key
from .science import audited, fit_audit, summaries
from .store import BudgetExceeded, Ledger, TaskStore

ARMS = ("M", "R", "S")
THRESHOLD = .9
REWARD_CORRECT = 1
REWARD_WRONG = -9


def task_rows(path):
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        return {key: {"kind": kind, "status": status, "prompt_hash": phash,
                      "result": json.loads(result) if result else None}
                for key, kind, status, phash, result in db.execute(
                    "SELECT id,kind,status,prompt_hash,result FROM tasks")}


def decision_prompt(batch, p, rates, cs, cost, strength):
    messages = prompts.decide(batch, p, rates, cs, cost, strength)
    # Preserve every other word, original confidence bands and option order.
    text = messages[1]["content"]
    for old, new in (("accepted wrong answers earn -4", "accepted wrong answers earn -9"),
                     ("threshold is probability at least 0.8", "threshold is probability at least 0.9")):
        if text.count(old) != 1:
            raise ValueError("Original decision prompt changed; inspect amendment")
        text = text.replace(old, new)
    messages[1]["content"] = text
    return messages


def release(p, y):
    p, y = np.asarray(p), np.asarray(y)
    keep = p >= THRESHOLD
    n = int(keep.sum())
    return {"points": float(np.sum(keep * ((REWARD_CORRECT - REWARD_WRONG) * y + REWARD_WRONG))),
            "accepted": n, "wrong_released": int(np.sum(keep & (y == 0))),
            "coverage": float(keep.mean()), "accepted_accuracy": float(y[keep].mean()) if n else None}


def choice_metrics(rows, arm, cost):
    counts = {"useful_taken": 0, "useful_missed": 0, "harmful_taken": 0,
              "harmful_avoided": 0, "ties_audited": 0, "ties_trusted": 0, "missing": 0}
    points, regrets = [], []
    for row in rows:
        action = row["actions"][str(cost)].get(arm)
        if action is None:
            counts["missing"] += 1
            continue
        gain = row["gross_audit_gain"] - cost
        if gain > 0:
            counts["useful_taken" if action == "AUDIT" else "useful_missed"] += 1
        elif gain < 0:
            counts["harmful_taken" if action == "AUDIT" else "harmful_avoided"] += 1
        else:
            counts["ties_audited" if action == "AUDIT" else "ties_trusted"] += 1
        value = row["trust"]["points"] + (gain if action == "AUDIT" else 0)
        points.append(value)
        regrets.append(row["trust"]["points"] + max(0, gain) - value)
    non_ties = sum(counts[k] for k in ("useful_taken", "useful_missed", "harmful_taken", "harmful_avoided"))
    return {**counts, "complete_batches": len(points),
            "correct_non_tie_choices": counts["useful_taken"] + counts["harmful_avoided"],
            "non_tie_batches": non_ties,
            "mean_points": float(np.mean(points)) if points else None,
            "mean_realized_regret": float(np.mean(regrets)) if regrets else None}


def load_parent(parent):
    parent = Path(parent).resolve()
    study = read_json(parent / "study.json")
    data_dir = Path(study["data_dir"])
    manifest = verify(data_dir)
    if manifest != study["manifest"] or protocol_fingerprint(study["config"], manifest) != study["fingerprint"]:
        raise ValueError("Original study or protocol changed")
    c = study["config"]
    if (c["reward_correct"], c["reward_wrong"], c["threshold"]) != (1, -4, .8):
        raise ValueError("This amendment requires the original +1/-4 pilot")
    splits = read_json(data_dir / "splits.json")
    batches = [b for b in splits["batches"] if b["split"] == "development"]
    rows = task_rows(parent / "tasks.sqlite")
    ids = {i for domain in splits["domains"].values() for split in ("development", "calibration") for i in domain[split]}
    keys = {generation_key(i) for i in ids}
    keys.update(review_key(b["id"], a) for b in batches for a in ARMS)
    if any(k not in rows or rows[k]["status"] != "done" for k in keys):
        raise ValueError("Original pilot answers/reviews are incomplete")
    evaluation_ids = {i for domain in splits["domains"].values() for i in domain["evaluation"]}
    if any(generation_key(i) in rows for i in evaluation_ids) or any(":evaluation-" in k or "-evaluation-" in k for k in rows):
        raise ValueError("Expected an untouched evaluation set")
    result = {k: rows[k]["result"] for k in keys}
    return {"parent": parent, "study": study, "data_dir": data_dir, "config": c,
            "splits": splits, "batches": batches, "ids": ids, "results": result,
            "parent_tasks_hash": digest(rows), "source_tasks_hash": digest({k: rows[k] for k in sorted(keys)})}


def build_jobs(source):
    c, results, splits = source["config"], source["results"], source["splits"]
    jobs = []
    for batch in source["batches"]:
        p = [results[generation_key(i)]["confidence"] for i in batch["ids"]]
        cal_ids = splits["domains"][batch["domain"]]["calibration"]
        cs = summaries([results[generation_key(i)]["confidence"] for i in cal_ids])
        for arm in ARMS:
            rates = band_rates(arm, results[review_key(batch["id"], arm)], batch["ids"], p)
            for cost in c["audit_costs"]:
                messages = decision_prompt(batch["id"], p, rates, cs, cost, c["audit_prior_strength"])
                jobs.append((decision_key(batch["id"], arm, 0, cost), "decide", messages, {}, c["max_tokens"]["decide"]))
    return jobs


def initialize(source, directory, jobs):
    directory = Path(directory)
    if directory.resolve() == source["parent"]:
        raise ValueError("Never overwrite the original pilot")
    c = dict(source["config"], threshold=THRESHOLD, reward_correct=REWARD_CORRECT,
             reward_wrong=REWARD_WRONG, protocol_version="0.2-decision-only-penalty9-pilot")
    spec = {"config": c, "parent": str(source["parent"]),
            "parent_fingerprint": source["study"]["fingerprint"],
            "parent_tasks_hash": source["parent_tasks_hash"], "source_tasks_hash": source["source_tasks_hash"],
            "source_code": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(Path(__file__).parent.glob("*.py"))},
            "decision_prompt_hash": digest(jobs), "phase": "development_decisions_only",
            "planned_calls": len(jobs), "evaluation_calls": 0,
            "reason": "The original pilot audit changed no acceptance decisions. A development-only check found useful and harmful audits with +1/-9 payoffs. User authorized this amendment before new choices.",
            "interpretation": "Exploratory pilot amendment. Useful means positive realized net audit gain, not certainty about optimal expected decisions."}
    fingerprint = digest(spec)
    path = directory / "amendment.json"
    if path.exists():
        existing = read_json(path)
        if existing["fingerprint"] != fingerprint:
            raise ValueError("Amendment or parent changed; preserve this run and use a new directory")
    else:
        write_json(path, {**spec, "fingerprint": fingerprint, "created": now()})
        write_json(directory / "served_model.json", read_json(source["parent"] / "served_model.json"))
    return c


def run_decisions(source, directory, c, jobs, cap):
    directory = Path(directory)
    with (directory / "process.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = TaskStore(directory / "tasks.sqlite")
        ledger = Ledger(ROOT / "runs/spend.sqlite", min(cap, c["budget_usd"]))
        client = Client(c, source["study"]["provider"], ledger, store, directory)
        pending = [j for j in jobs if not store.get(j[0]) or store.get(j[0])["status"] not in ("done", "invalid")]
        before = ledger.summary()
        errors, valid = [], 0
        print(f"Running {len(pending)} amended decision calls; no generation, review or evaluation calls.", flush=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=c["workers"]) as executor:
                futures = {executor.submit(client.execute, *j): j[0] for j in pending}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        valid += int(future.result() is not None)
                    except Exception as error:
                        errors.append({"task": futures[future], "error": str(error)})
                        if isinstance(error, (FatalAPIError, BudgetExceeded)):
                            client.stop.set()
                    if (valid + len(errors)) % 12 == 0:
                        print(f"{valid} valid decisions; {len(errors)} errors.", flush=True)
            after = ledger.summary()
            summary = {"created": now(), "valid_this_run": valid, "errors": errors,
                       "stop_reason": client.stop_reason, "statuses": store.statuses(),
                       "invocation_cost_usd": after["conservative_cost_usd"] - before["conservative_cost_usd"],
                       "global_spend": after}
            write_json(directory / "last_run.json", summary)
            return summary
        finally:
            store.db.close()
            ledger.db.close()


def score(source, directory, jobs):
    directory = Path(directory)
    current_parent = task_rows(source["parent"] / "tasks.sqlite")
    if digest(current_parent) != source["parent_tasks_hash"]:
        raise ValueError("Original task database changed")
    decisions = task_rows(directory / "tasks.sqlite")
    expected = {j[0] for j in jobs}
    if not set(decisions).issubset(expected) or any(r["kind"] != "decide" for r in decisions.values()):
        raise ValueError("Only prescribed development decision tasks are permitted")
    for key, _, messages, _, maximum in jobs:
        row = decisions.get(key)
        phash = digest({"messages": messages, "maximum": maximum, "model": source["config"]["model"]})
        if row and row["prompt_hash"] != phash:
            raise ValueError("Saved decision prompt does not match amendment")
    # Only calibration and development labels enter any score.
    labels = {}
    with (source["data_dir"] / "private_labels.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in source["ids"]:
                labels[row["id"]] = row["gold"]
    if set(labels) != source["ids"]:
        raise ValueError("Missing pilot/calibration labels")
    results, c = source["results"], source["config"]
    def arrays(ids):
        return ([results[generation_key(i)]["confidence"] for i in ids],
                [int(results[generation_key(i)]["answer"] == labels[i]) for i in ids])
    mapping = {d: fit_audit(*arrays(parts["calibration"]), c["audit_prior_strength"])
               for d, parts in source["splits"]["domains"].items()}
    rows = []
    for batch in source["batches"]:
        p, y = arrays(batch["ids"])
        pa = audited(p, mapping[batch["domain"]])
        trust, audit = release(p, y), release(pa, y)
        direct = {}
        for arm in ARMS:
            output = results[review_key(batch["id"], arm)]
            q = [x["p"] for x in output["revised_probabilities"]] if arm == "R" else audited(p, {"rates": output["band_accuracy"]})
            direct[arm] = release(q, y)
        actions = {}
        for cost in c["audit_costs"]:
            actions[str(cost)] = {}
            for arm in ARMS:
                saved = decisions.get(decision_key(batch["id"], arm, 0, cost))
                actions[str(cost)][arm] = saved["result"]["action"] if saved and saved["status"] == "done" else None
        rows.append({"batch": batch["id"], "domain": batch["domain"], "trust": trust, "audit_gross": audit,
                     "gross_audit_gain": audit["points"] - trust["points"],
                     "changed_acceptance": int(np.sum((np.asarray(p) >= THRESHOLD) != (pa >= THRESHOLD))),
                     "direct_use": direct, "actions": actions})
    by_cost = {}
    for cost in c["audit_costs"]:
        paired = [r for r in rows if all(r["actions"][str(cost)][a] is not None for a in ARMS)]
        by_cost[str(cost)] = {"paired_batches": len(paired), "arms": {a: choice_metrics(paired, a, cost) for a in ARMS},
            "missing_decisions": {a: sum(r["actions"][str(cost)][a] is None for r in rows) for a in ARMS},
            "baseline_mean_points": {"never_audit": float(np.mean([r["trust"]["points"] for r in paired])) if paired else None,
                                     "always_audit": float(np.mean([r["audit_gross"]["points"] - cost for r in paired])) if paired else None,
                                     "defer_all": 0.,
                                     **{a + "_direct": float(np.mean([r["direct_use"][a]["points"] for r in paired])) if paired else None for a in ("M", "R")}}}
    with sqlite3.connect((ROOT / "runs/spend.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        usage = db.execute("SELECT COUNT(*),COALESCE(SUM(charged),0) FROM attempts WHERE study=?", (str(directory.resolve()),)).fetchone()
    report = {"created": now(), "warning": "Exploratory development pilot; not confirmatory findings",
              "threshold": THRESHOLD, "reward_correct": REWARD_CORRECT, "reward_wrong": REWARD_WRONG,
              "primary_cost": c["primary_cost"], "by_cost": by_cost, "batch_results": rows,
              "amendment_fingerprint": read_json(directory / "amendment.json")["fingerprint"],
              "decision_results_hash": digest(decisions),
              "api_attempts_this_amendment": usage[0], "estimated_cost_usd_this_amendment": usage[1],
              "checks": {"original_task_database_unchanged": True, "evaluation_tasks": 0,
                         "new_task_kinds": sorted({r["kind"] for r in decisions.values()}),
                         "new_generation_or_review_calls": 0, "used_label_splits": ["calibration", "development"]},
              "interpretation": "Useful/harmful refer to realized net gains on these batches. A missed realized gain is not proof of irrationality before outcomes were known. Direct-use policies incur no audit fee; review API costs are separate."}
    out = directory / "reports/pilot"
    write_json(out / "results.json", report)
    render_html(report, out / "report.html")
    print(json.dumps({"primary_cost": c["primary_cost"], "results": by_cost[str(c["primary_cost"])],
                      "estimated_cost_usd": usage[1], "checks": report["checks"]}, indent=2), flush=True)
    print(f"Report: {out / 'report.html'}", flush=True)
    return report


def render_html(report, path):
    primary = report["by_cost"][str(report["primary_cost"])]
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Penalty −9 pilot</title>",
             "<style>body{font:16px/1.5 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#173047}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #cbd6de;text-align:left}.notice{background:#fff2d1;padding:18px}</style></head><body>",
             "<h1>Did the model choose useful audits?</h1><p class='notice'>Exploratory pilot amendment. Same frozen answers, reviews and calibration bank; correct +1, wrong −9, release threshold 90%. No evaluation calls.</p>",
             "<p>M = explicit confidence-reliability review; R = ordinary answer reassessment; S = domain and confidence summaries only. Each feeds an otherwise identical fresh audit-choice prompt.</p>",
             "<h2>Primary audit fee: 3 points</h2><p>Useful means a positive realized gain after paying the audit fee. Ties are excluded from correct/incorrect-choice counts.</p>",
             "<table><tr><th>Batch</th><th>Audit net gain</th><th>M choice</th><th>R choice</th><th>S choice</th></tr>"]
    for row in report["batch_results"]:
        parts.append("<tr><td>" + html.escape(row["batch"]) + "</td><td>" + f"{row['gross_audit_gain'] - report['primary_cost']:+.0f}" + "</td>" + "".join(f"<td>{html.escape(str(row['actions'][str(report['primary_cost'])][a]))}</td>" for a in ARMS) + "</tr>")
    parts.append("</table><h2>Choices and payoff</h2><table><tr><th>Arm</th><th>Useful taken / available</th><th>Harmful taken / available</th><th>Correct non-tie choices</th><th>Mean points</th></tr>")
    for arm, m in primary["arms"].items():
        parts.append(f"<tr><td>{arm}</td><td>{m['useful_taken']}/{m['useful_taken'] + m['useful_missed']}</td><td>{m['harmful_taken']}/{m['harmful_taken'] + m['harmful_avoided']}</td><td>{m['correct_non_tie_choices']}/{m['non_tie_batches']}</td><td>{m['mean_points']}</td></tr>")
    parts.append("</table><h2>Baselines: mean points per batch</h2><table><tr><th>Policy</th><th>Mean points</th></tr>")
    for policy, value in primary["baseline_mean_points"].items():
        parts.append(f"<tr><td>{html.escape(policy)}</td><td>{value}</td></tr>")
    parts.append("</table><p>Direct-use policies apply the existing review estimates at the new 90% threshold without buying an audit or changing answers.</p><h2>All audit fees and missing outputs</h2><pre>" + html.escape(json.dumps(report["by_cost"], indent=2)) + "</pre>")
    parts.append(f"<p>Estimated additional API cost: US${report['estimated_cost_usd_this_amendment']:.6f}. Additional attempts: {report['api_attempts_this_amendment']}.</p><p>" + html.escape(report["interpretation"]) + "</p><p><a href='results.json'>Machine-readable results</a> · <a href='../../amendment.json'>Recorded amendment</a></p></body></html>")
    Path(path).write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "runs/main")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/penalty9")
    parser.add_argument("--budget-usd", type=float, default=1., help="Cumulative project cap, including the original pilot")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    source = load_parent(args.parent)
    jobs = build_jobs(source)
    c = initialize(source, args.run_dir, jobs)
    if not args.score_only:
        load_env()
        result = run_decisions(source, args.run_dir, c, jobs, args.budget_usd)
        if result["errors"] or result["stop_reason"]:
            raise SystemExit("Amendment stopped; inspect last_run.json before resuming")
    score(source, args.run_dir, jobs)


if __name__ == "__main__":
    main()
