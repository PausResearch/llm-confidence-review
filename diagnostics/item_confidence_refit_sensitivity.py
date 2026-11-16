"""Post-hoc paired bootstrap of calibration fitting plus evaluation sampling.

No model API requests, prompt changes or evaluation fitting. All original data,
models and analyses are read-only. The conditional frozen-model analysis stays
primary; these intervals address a broader, training-sample-sensitive pipeline.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from auditstudy.common import ROOT, digest, now, read_json, write_json
from diagnostics.item_confidence_analysis import MODELS, fit, losses, predict
from diagnostics.item_confidence_evaluation_analysis import CONTRASTS

REPETITIONS = 2000
SEED = 20261116
WORKERS = 4
DIRECTORY = ROOT / "runs/item_confidence_sensitivity"


def validate_inputs(train, evaluation):
    for rows, split in ((train, "calibration"), (evaluation, "evaluation")):
        if not rows or any(r["split"] != split for r in rows):
            raise ValueError("Expected unmixed " + split + " rows")
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate source IDs")
        for row in rows:
            for key in ("p", "q", "q2", "r", "y"):
                if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                    raise ValueError("Invalid source value " + key)
            if row["r"] not in (0, 1) or row["y"] not in (0, 1):
                raise ValueError("Expected binary labels and reliability")
    if {r["id"] for r in train} & {r["id"] for r in evaluation}:
        raise ValueError("Calibration/evaluation ID overlap")
    if {r["domain"] for r in train} != {r["domain"] for r in evaluation}:
        raise ValueError("Calibration/evaluation domains differ")


def stratified_indices(rows, rng):
    domains = np.array([r["domain"] for r in rows])
    return np.concatenate([rng.choice(np.flatnonzero(domains == domain),
                                     size=int(np.sum(domains == domain)), replace=True)
                           for domain in sorted(set(domains.tolist()))])


def replicate(train, evaluation, index, seed=SEED):
    # A reproducible independent stream per replicate makes worker scheduling irrelevant.
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    train_idx = stratified_indices(train, rng)
    eval_idx = stratified_indices(evaluation, rng)
    sampled_train = [train[i] for i in train_idx]
    sampled_eval = [evaluation[i] for i in eval_idx]
    y = np.array([r["y"] for r in sampled_eval])
    values = {}
    for name, signals in MODELS.items():
        model = fit(sampled_train, signals)
        values[name] = losses(predict(model, sampled_eval), y)
    contrasts = {}
    for name, (first, second) in CONTRASTS.items():
        contrasts[name] = {metric: float(np.mean(values[first][metric] - values[second][metric]))
                           for metric in ("brier", "log_loss")}
    return {"replicate": index, "contrasts": contrasts}


_TRAIN = None
_EVAL = None


def worker_initialize(train, evaluation):
    global _TRAIN, _EVAL
    _TRAIN, _EVAL = train, evaluation


def worker(index):
    try:
        return replicate(_TRAIN, _EVAL, index)
    except Exception as error:
        return {"replicate": index, "failure": type(error).__name__ + ": " + str(error)}


def source_hashes(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def summarize(replicates, original):
    successful = [r for r in replicates if "failure" not in r]
    failures = [r for r in replicates if "failure" in r]
    result = {}
    for key, (first, second) in CONTRASTS.items():
        result[key] = {"first_model": first, "second_model": second}
        for metric in ("brier", "log_loss"):
            values = np.array([r["contrasts"][key][metric] for r in successful])
            result[key][metric] = {
                "original_point_estimate": original["contrasts"][key][metric]["mean_difference"],
                "original_frozen_model_ci95": original["contrasts"][key][metric]["ci95"],
                "refit_bootstrap_ci95": np.quantile(values, [.025, .975]).tolist() if len(values) else None,
                "bootstrap_mean": float(values.mean()) if len(values) else None,
                "bootstrap_standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
                "direction": "negative favors first model"}
    return {"contrasts": result, "successful_replicates": len(successful), "failed_replicates": len(failures),
            "failures": failures, "failures_replaced": False,
            "note": "Percentile intervals use successful replicates only; any failure is reported. Original frozen-model intervals and point estimates remain the main analysis."}


def main():
    paths = [ROOT / "runs/item_confidence/records.json", ROOT / "runs/item_confidence_evaluation/records.json",
             ROOT / "runs/item_confidence_evaluation/results.json", ROOT / "runs/item_confidence_evaluation/frozen_models.json",
             ROOT / "runs/item_confidence/protocol.json", ROOT / "runs/item_confidence_evaluation/protocol.json",
             ROOT / "diagnostics/item_confidence_analysis.py", ROOT / "diagnostics/item_confidence_evaluation_analysis.py"]
    before = source_hashes(paths)
    train = sorted([r for r in read_json(paths[0]) if r["split"] == "calibration"], key=lambda r: r["id"])
    evaluation = sorted(read_json(paths[1]), key=lambda r: r["id"])
    validate_inputs(train, evaluation)
    domains = sorted({r["domain"] for r in train})
    if len(train) != 480 or len(evaluation) != 1600 or len(domains) != 4:
        raise ValueError("Expected the completed 480/1600 dataset")
    if any(sum(r["domain"] == d for r in train) != 120 or sum(r["domain"] == d for r in evaluation) != 400 for d in domains):
        raise ValueError("Unexpected source stratum counts")
    original = read_json(paths[2])
    models = read_json(paths[3])
    # Check exact original recipe before bootstrapping; no parameters are selected.
    for name, signals in MODELS.items():
        fitted = fit(train, signals)
        if digest(fitted) != digest(models[name]):
            raise ValueError("Refitting the unresampled calibration rows did not reproduce the frozen map " + name)
    plan = {"type": "post_hoc_sensitivity_after_evaluation_results_known", "repetitions": REPETITIONS,
            "seed": SEED, "stream": "numpy default_rng(SeedSequence([seed, replicate_index]))", "workers": WORKERS,
            "train_n": 480, "evaluation_n": 1600, "domains": domains,
            "sampling": "Resample calibration and evaluation independently within subject, preserving 120/400 cases per subject. All compared methods use the SAME paired train and evaluation resamples within a replicate.",
            "fitting": "Exact original fit() recipe, refitting scaling and ridge-logistic coefficients exclusively on the resampled calibration cases. Evaluation labels never enter fitting.",
            "metrics": ["brier", "log_loss"], "contrasts": CONTRASTS,
            "uncertainty": "Percentile 95% intervals over refitted-resampled pipeline differences. Original point estimates retained; bootstrap means labeled separately.",
            "failure_policy": "Fixed 2000 attempted replicates; report all fit failures, no selective replacement or outcome-based stopping.",
            "interpretation": "Post-hoc sensitivity broadens uncertainty to the finite calibration sample; it does not replace or rescue the frozen-model primary analysis, establish equivalence or identify an internal metacognitive mechanism. It does not resample model generations.",
            "source_hashes": before, "code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fingerprint = digest(plan)
    plan_path = DIRECTORY / "refit_plan.json"
    if plan_path.exists():
        if read_json(plan_path)["fingerprint"] != fingerprint:
            raise ValueError("Recorded sensitivity changed; preserve earlier run")
    else:
        write_json(plan_path, {**plan, "fingerprint": fingerprint, "recorded_before_computation": now()})
    if (DIRECTORY / "refit_results.json").exists():
        old = read_json(DIRECTORY / "refit_results.json")
        if old["plan_fingerprint"] != fingerprint:
            raise ValueError("Saved results differ from plan")
        print("Completed sensitivity already exists; no recomputation.", flush=True)
        return
    results = []
    print(f"Running {REPETITIONS} calibration+evaluation bootstrap replicates; no API calls.", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, initializer=worker_initialize,
                                                initargs=(train, evaluation)) as pool:
        for n, result in enumerate(pool.map(worker, range(REPETITIONS), chunksize=10), 1):
            results.append(result)
            if n % 100 == 0:
                print(json.dumps({"completed": n, "failures": sum("failure" in r for r in results)}), flush=True)
    if before != source_hashes(paths):
        raise ValueError("Source files changed during sensitivity analysis")
    summary = summarize(results, original)
    summary.update(created=now(), plan_fingerprint=fingerprint, post_hoc=True, source_hashes=before,
                   checks={"source_files_unchanged": True, "no_evaluation_fitting": True,
                           "source_ids_disjoint": True, "original_fit_exactly_reproduced": True, "api_calls": 0},
                   interpretation=plan["interpretation"])
    write_json(DIRECTORY / "refit_replicates.json", sorted(results, key=lambda r: r["replicate"]))
    write_json(DIRECTORY / "refit_results.json", summary)
    print(json.dumps({k: summary[k] for k in ("contrasts", "successful_replicates", "failed_replicates", "checks")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
