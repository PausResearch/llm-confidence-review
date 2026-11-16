"""Authorized held-out evaluation; pilot prompts and fitted mappings stay fixed."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path

from auditstudy import provider, prompts
from auditstudy.common import ROOT, digest, dumps, load_env, now, read_json, write_json
from auditstudy.penalty_pilot import load_parent, task_rows
from auditstudy.provider import Client, FatalAPIError
from auditstudy.runner import generation_key
from auditstudy.store import BudgetExceeded, Ledger, TaskStore
from diagnostics.item_confidence import messages_for, validate

DIRECTORY = ROOT / "runs/item_confidence_evaluation"
PILOT = ROOT / "runs/item_confidence"
WORKERS = 24
CAP_USD = 5.0
ARMS = ("M", "R", "R2")


def review_messages(item, answer, arm):
    if item["split"] != "evaluation":
        raise ValueError("Only evaluation records allowed in this runner")
    # Compatibility adapter ONLY for the old pure prompt formatter. The split
    # is not serialized by that formatter; actual data and membership remain
    # evaluation. This neither changes stored rows nor bypasses membership tests.
    return messages_for(dict(item, split="development"), answer, arm)


def source_data():
    source = load_parent(ROOT / "runs/main")
    pilot_protocol = read_json(PILOT / "protocol.json")
    for name, expected in pilot_protocol["code_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen pilot source changed: " + name)
    pilot = read_json(PILOT / "results.json")
    if pilot["fit_n"] != 480 or pilot["score_n"] != 160 or pilot["missing"] or pilot["errors"]:
        raise ValueError("Expected completed 480/160 pilot")
    evaluation_ids = {i for parts in source["splits"]["domains"].values() for i in parts["evaluation"]}
    if len(evaluation_ids) != 1600 or evaluation_ids & source["ids"]:
        raise ValueError("Evaluation membership is not 1600 distinct held-out items")
    old_records = read_json(PILOT / "records.json")
    if {r["id"] for r in old_records} != source["ids"] or any(r["id"] in evaluation_ids for r in old_records):
        raise ValueError("Pilot/evaluation overlap")
    items = {}
    with (source["data_dir"] / "public_items.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in evaluation_ids:
                if row["split"] != "evaluation":
                    raise ValueError("Stored evaluation membership disagrees with split")
                items[row["id"]] = row
    if set(items) != evaluation_ids:
        raise ValueError("Missing evaluation items")
    return source, pilot, items


def generation_jobs(items):
    jobs = [(generation_key(i), "generate", prompts.generate(row),
             {"letters": [chr(65+j) for j in range(len(row["options"]))]}, 256)
            for i, row in sorted(items.items())]
    return sorted(jobs, key=lambda j: digest([20261115, j[0], "evaluation-generation"]))


def review_jobs(items, saved):
    jobs, missing = [], []
    for ident, item in sorted(items.items()):
        initial = saved.get(generation_key(ident))
        if not initial or initial["status"] != "done":
            missing.append(ident)
            continue
        for arm in ARMS:
            jobs.append((f"evaluation_review:{ident}:{arm}", "item_reliability" if arm == "M" else "review_R",
                         review_messages(item, initial["result"], arm), {} if arm == "M" else {"ids": [ident]}, 128))
    return sorted(jobs, key=lambda j: digest([20261115, j[0], "evaluation-review"])), missing


def freeze(source, pilot, items, jobs):
    files = [Path(__file__), Path(__file__).with_name("item_confidence_evaluation_analysis.py"),
             Path(__file__).with_name("item_confidence.py"), Path(__file__).with_name("item_confidence_analysis.py"),
             *sorted((ROOT / "auditstudy").glob("*.py"))]
    c = dict(source["config"], workers=WORKERS)
    spec = {"version": "item-confidence-heldout-v1", "evaluation_ids": sorted(items),
            "evaluation_domain_counts": {d: sum(i["domain"] == d for i in items.values()) for d in sorted({i["domain"] for i in items.values()})},
            "generation_jobs_hash": digest(jobs), "config": c, "cumulative_spend_cap_usd": CAP_USD,
            "initial_answer_calls": 1600, "review_calls": 4800, "generation_max_tokens": 256, "review_max_tokens": 128,
            "format_retries": {"generation": 1, "reviews": 0}, "worker_count": WORKERS,
            "parent_tasks_hash": source["parent_tasks_hash"], "parent_protocol": source["study"]["fingerprint"],
            "pilot_tasks_hash": digest(task_rows(PILOT / "tasks.sqlite")), "pilot_results_hash": digest(pilot),
            "pilot_protocol_fingerprint": read_json(PILOT / "protocol.json")["fingerprint"],
            "frozen_models_hash": digest(pilot["models"]),
            "model_fit": "Reuse EXACT five mappings and scalers from 480 calibration cases; no fitting, feature changes or parameter selection on evaluation.",
            "sampling": "All 1600 preselected evaluation IDs, 400 in each of four domains; no outcome-based filtering.",
            "prompts": "Initial generation identical to original pilot. M/R/R2 messages byte-identical to item-level pilot given the same visible record; split marker is never sent. R2 has a distinct task key and separate request; no review output is shown to another review.",
            "primary": "Brier(p_r) minus Brier(p_q) on evaluation; negative favors reliability. One additional review call each.",
            "key_specificity": "Brier(p_q_r) minus Brier(p_q_q2); two review calls each. Interpret as specificity evidence only with a favorable primary result; otherwise descriptive.",
            "secondary": ["Brier(p_q_r)-Brier(p_q): extra-call complementarity", "Brier(p_r)-Brier(p): comparison to simple original-confidence calibration", "log loss, AUROC, fixed .5/.75/.9 coverage accuracy, +1/-4 acceptance utility at .8; descriptive domain and reliability groups"],
            "intervals": "5000 paired item bootstrap replicates stratified by subject, seed20261115; coefficients/scalers held fixed. Conditional on this fitted pipeline, not training-sample or provider variability. No equivalence claim from nonsignificance.",
            "missingness": "Original generation allows its existing single format-only correction; reviews allow none. No correctness feedback or outcome-based retries. Score shared complete cases; disclose missingness and completion fractions. Failure remains failure, not evidence about accuracy.",
            "label_policy": "Private evaluation labels only enter offline scoring after generation/reviews complete. They never enter requests or predictor fitting.",
            "stopping": "Fixed 1600-question sample. No interim performance peeking or early stopping by effect; only budget, errors or user instruction can stop collection.",
            "limitations": "One model alias, four public benchmark subjects, one prompt wording, retrospective separate-call transcript assessment, binary reliability, restricted statistical mappings. Positive effects need not be self-specific or reflect distinct internal mechanisms; no effect does not imply absence of metacognition. Equal call counts do not equalize actual computation.",
            "registration": "Locally timestamped protocol after exploratory pilots but before any evaluation API calls; not externally preregistered.",
            "code_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    fingerprint = digest(spec)
    path = DIRECTORY / "protocol.json"
    if path.exists():
        if read_json(path)["fingerprint"] != fingerprint or digest(read_json(DIRECTORY / "frozen_models.json")) != spec["frozen_models_hash"]:
            raise ValueError("Frozen evaluation protocol/model changed; do not overwrite")
    else:
        if (DIRECTORY / "tasks.sqlite").exists():
            raise ValueError("Task database exists without frozen protocol")
        write_json(path, {**spec, "fingerprint": fingerprint, "recorded_before_evaluation_calls": now()})
        write_json(DIRECTORY / "frozen_models.json", pilot["models"])
        write_json(DIRECTORY / "served_model.json", read_json(PILOT / "served_model.json"))
        write_json(DIRECTORY / "generation_prompts.json", [{"task": j[0], "messages": j[2]} for j in jobs])
    return c, fingerprint


def save_review_plan(jobs, missing):
    spec = {"jobs_hash": digest(jobs), "planned_reviews": len(jobs), "generation_missing": missing}
    path = DIRECTORY / "review_plan.json"
    if path.exists():
        if any(read_json(path)[k] != v for k, v in spec.items()):
            raise ValueError("Frozen answers or review plan changed")
    else:
        write_json(path, {**spec, "recorded_before_review_calls": now()})
        write_json(DIRECTORY / "review_prompts.json", [{"task": j[0], "messages": j[2]} for j in jobs])


def run_stage(client, jobs, store, stage):
    pending = [j for j in jobs if not store.get(j[0]) or store.get(j[0])["status"] not in ("done", "invalid")]
    print(f"{stage}: {len(pending)} pending requests", flush=True)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(client.execute, *job): job[0] for job in pending}
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:
                errors.append({"task": futures[future], "error": str(error)})
                if isinstance(error, (FatalAPIError, BudgetExceeded)):
                    client.stop.set()
            if n % 240 == 0 or n == len(pending):
                print(dumps({"stage": stage, "returned": n, "pending_total": len(pending), "errors": len(errors), "statuses": store.statuses()}), flush=True)
    if client.stop_reason:
        errors.append({"stop_reason": client.stop_reason})
    return errors


def score(source, items, all_jobs, fingerprint, ledger, errors):
    saved = task_rows(DIRECTORY / "tasks.sqlite")
    expected = {j[0]: j for j in all_jobs}
    if set(saved) - set(expected):
        raise ValueError("Unexpected evaluation task")
    for key, row in saved.items():
        job = expected[key]
        phash = digest({"messages": job[2], "maximum": job[4], "model": source["config"]["model"]})
        if phash != row["prompt_hash"]:
            raise ValueError("Stored prompt differs from frozen evaluation plan")
    # This is the first and only place evaluation correctness enters the runner.
    labels = {}
    with (source["data_dir"] / "private_labels.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in items:
                labels[row["id"]] = row["gold"]
    if set(labels) != set(items):
        raise ValueError("Missing evaluation labels")
    rows, missing = [], []
    for ident, item in sorted(items.items()):
        keys = [generation_key(ident)] + [f"evaluation_review:{ident}:{a}" for a in ARMS]
        bad = [key for key in keys if key not in saved or saved[key]["status"] != "done"]
        if bad:
            missing.append({"id": ident, "domain": item["domain"], "tasks": [{"key": k, "status": saved[k]["status"] if k in saved else "absent"} for k in bad]})
            continue
        a, m, r, r2 = [saved[k]["result"] for k in keys]
        rows.append({"id": ident, "domain": item["domain"], "split": "evaluation", "answer": a["answer"],
                     "p": a["confidence"], "y": int(a["answer"] == labels[ident]), "r": int(m["reliability"] == "HIGH"),
                     "q": r["revised_probabilities"][0]["p"], "q2": r2["revised_probabilities"][0]["p"]})
    write_json(DIRECTORY / "records.json", rows)
    from diagnostics.item_confidence_evaluation_analysis import analyze_evaluation
    result = analyze_evaluation(rows, read_json(DIRECTORY / "frozen_models.json"))
    protocol = read_json(DIRECTORY / "protocol.json")
    checks = {"original_tasks_unchanged": digest(task_rows(source["parent"] / "tasks.sqlite")) == protocol["parent_tasks_hash"],
              "pilot_tasks_unchanged": digest(task_rows(PILOT / "tasks.sqlite")) == protocol["pilot_tasks_hash"],
              "pilot_results_unchanged": digest(read_json(PILOT / "results.json")) == protocol["pilot_results_hash"],
              "frozen_models_unchanged": digest(read_json(DIRECTORY / "frozen_models.json")) == protocol["frozen_models_hash"],
              "evaluation_disjoint_from_pilot": not bool(set(items) & source["ids"]), "no_evaluation_fitting": True,
              "dataset_verified": True}
    if not all(checks.values()):
        raise ValueError("Evaluation integrity check failed")
    usage = ledger.db.execute("SELECT COUNT(*),COALESCE(SUM(charged),0),COALESCE(SUM(prompt_tokens),0),COALESCE(SUM(completion_tokens),0) FROM attempts WHERE study=?", (str(DIRECTORY.resolve()),)).fetchone()
    result.update(created=now(), protocol_fingerprint=fingerprint, expected_items=1600, missing=missing, errors=errors, checks=checks,
                  usage=dict(zip(("api_attempts", "estimated_cost_usd", "prompt_tokens", "completion_tokens"), usage)), cumulative_spend=ledger.summary())
    write_json(DIRECTORY / "results.json", result)
    print(dumps({k: result[k] for k in ("contrasts", "usage", "checks")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    source, pilot, items = source_data()
    gen_jobs = generation_jobs(items)
    c, fingerprint = freeze(source, pilot, items, gen_jobs)
    if args.prepare_only:
        print(dumps({"frozen": fingerprint, "items": len(items), "planned_calls": 6400, "estimated_cap_usd": CAP_USD}))
        return
    if args.score_only and not (DIRECTORY / "tasks.sqlite").exists():
        raise SystemExit("No evaluation responses exist; scoring cannot start")
    with (DIRECTORY / "process.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        load_env()
        store = TaskStore(DIRECTORY / "tasks.sqlite")
        ledger = Ledger(ROOT / "runs/spend.sqlite", CAP_USD)
        original_validate = provider.validate
        provider.validate = validate
        errors = []
        try:
            if not args.score_only:
                generator = Client(dict(c, format_retries=1), source["study"]["provider"], ledger, store, DIRECTORY)
                errors.extend(run_stage(generator, gen_jobs, store, "generation"))
                if errors or generator.stop.is_set():
                    write_json(DIRECTORY / "interruption.json", {"stage": "generation", "errors": errors, "time": now()})
                    raise SystemExit("Generation interrupted; no evaluation labels scored")
            initial_rows = task_rows(DIRECTORY / "tasks.sqlite")
            if any(j[0] not in initial_rows or initial_rows[j[0]]["status"] not in ("done", "invalid") for j in gen_jobs):
                raise SystemExit("Generation incomplete; review plan and evaluation scores remain unopened")
            reviews, failed_generation = review_jobs(items, initial_rows)
            save_review_plan(reviews, failed_generation)
            if not args.score_only:
                reviewer = Client(dict(c, format_retries=0), source["study"]["provider"], ledger, store, DIRECTORY)
                errors.extend(run_stage(reviewer, reviews, store, "review"))
                if errors or reviewer.stop.is_set():
                    write_json(DIRECTORY / "interruption.json", {"stage": "review", "errors": errors, "time": now()})
                    raise SystemExit("Review interrupted; no evaluation labels scored")
            # No interim scoring; every planned task must have returned or be invalid.
            saved = task_rows(DIRECTORY / "tasks.sqlite")
            if any(j[0] not in saved or saved[j[0]]["status"] not in ("done", "invalid") for j in gen_jobs + reviews):
                raise SystemExit("Collection incomplete; evaluation labels remain unscored")
            result = score(source, items, gen_jobs + reviews, fingerprint, ledger, errors)
            if result["missing"]:
                print(f"Completed with {len(result['missing'])} missing complete cases; retained in report.", flush=True)
        finally:
            provider.validate = original_validate
            store.db.close()
            ledger.db.close()


if __name__ == "__main__":
    main()
