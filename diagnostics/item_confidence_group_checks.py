"""Post-hoc subject and calibration-group checks of saved evaluation outputs.

No model calls, fitting, prompt selection or edits to original results occur.
The group comparison is a descriptive check of reliability-label meaning, not
a test distinguishing reliability awareness from ordinary correctness checking.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np

from auditstudy.common import ROOT, digest, now, read_json, write_json
from diagnostics.item_confidence_analysis import paired_bootstrap
from diagnostics.item_confidence_evaluation_analysis import CONTRASTS

REPETITIONS = 5000
SEED = 20261116
BANDS = ("[0,0.6)", "[0.6,0.8)", "[0.8,1]")
DIRECTORY = ROOT / "runs/item_confidence_sensitivity"
INPUTS = (ROOT / "runs/item_confidence_evaluation/records.json",
          ROOT / "runs/item_confidence_evaluation/results.json")


def band_index(p):
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Confidence must be finite and in [0,1]")
    return 0 if p < .6 else 1 if p < .8 else 2


def percentile(values):
    return [float(v) for v in np.quantile(values, [.025, .975])]


def cell_summary(part, rng, repetitions=REPETITIONS):
    """Resample whole observations so probability and correctness stay paired."""
    n = len(part)
    if not n:
        return {"n": 0, "small_group": True, "mean_original_confidence": None,
                "accuracy": None, "signed_gap": None, "signed_gap_ci95": None,
                "absolute_gap": None, "absolute_gap_ci95": None}, None
    p = np.asarray([x["p"] for x in part], float)
    y = np.asarray([x["y"] for x in part], float)
    residual = p - y
    indexes = rng.integers(0, n, size=(repetitions, n))
    samples = residual[indexes].mean(axis=1)
    signed_gap = float(residual.mean())
    return {"n": n, "small_group": n < 20,
            "mean_original_confidence": float(p.mean()), "accuracy": float(y.mean()),
            "signed_gap": signed_gap, "signed_gap_ci95": percentile(samples),
            "absolute_gap": abs(signed_gap), "absolute_gap_ci95": percentile(abs(samples))}, samples


def validate(records, predictions):
    if not records or len({x["id"] for x in records}) != len(records):
        raise ValueError("Unique evaluation records required")
    if len({x["id"] for x in predictions}) != len(predictions):
        raise ValueError("Duplicate saved prediction IDs")
    by_id = {x["id"]: x for x in predictions}
    if set(by_id) != {x["id"] for x in records}:
        raise ValueError("Predictions and records must have identical support")
    for row in records:
        if row["split"] != "evaluation" or row["y"] not in (0, 1) or row["r"] not in (0, 1):
            raise ValueError("Expected binary evaluation records")
        band_index(row["p"])
        pred = by_id[row["id"]]
        if any(pred[k] != row[k] for k in ("domain", "p", "y", "r")):
            raise ValueError("Saved prediction metadata does not match record")
        for first, second in CONTRASTS.values():
            for name in (first, second):
                p = pred["fitted_predictions"][name]
                if not np.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError("Invalid saved prediction")
    return by_id


def analyze_groups(records, predictions, repetitions=REPETITIONS, seed=SEED):
    by_id = validate(records, predictions)
    records = sorted(records, key=lambda x: x["id"])
    domains = sorted({r["domain"] for r in records})
    domain_results = []
    for domain in domains:
        part = [r for r in records if r["domain"] == domain]
        contrasts = {}
        for name, (first, second) in CONTRASTS.items():
            differences = [
                (by_id[r["id"]]["fitted_predictions"][first] - r["y"]) ** 2 -
                (by_id[r["id"]]["fitted_predictions"][second] - r["y"]) ** 2
                for r in part]
            contrasts[name] = paired_bootstrap(differences, [domain] * len(part), repetitions, seed)
        domain_results.append({"domain": domain, "n": len(part), "contrasts": contrasts})
    rng = np.random.default_rng(seed)
    cells, comparisons = [], []
    for domain in domains:
        for band in range(3):
            values, samples = {}, {}
            for high in (0, 1):
                part = [r for r in records if r["domain"] == domain and band_index(r["p"]) == band and r["r"] == high]
                values[high], samples[high] = cell_summary(part, rng, repetitions)
                cells.append({"domain": domain, "band": BANDS[band], "reliability": "HIGH" if high else "LOW", **values[high]})
            comparison = {"domain": domain, "band": BANDS[band], "n_high": values[1]["n"], "n_low": values[0]["n"],
                          "any_small_group": values[0]["small_group"] or values[1]["small_group"],
                          "mean_p_high_minus_low": None, "absolute_gap_high_minus_low": None, "ci95": None}
            if all(samples[h] is not None for h in (0, 1)):
                comparison.update(mean_p_high_minus_low=values[1]["mean_original_confidence"] - values[0]["mean_original_confidence"],
                                  absolute_gap_high_minus_low=values[1]["absolute_gap"] - values[0]["absolute_gap"],
                                  ci95=percentile(abs(samples[1]) - abs(samples[0])))
            comparisons.append(comparison)
    # Overall descriptive calibration preserves the original subject mixture.
    overall = {"n": len(records), "accuracy": float(np.mean([r["y"] for r in records])),
               "mean_original_confidence": float(np.mean([r["p"] for r in records]))}
    overall["signed_gap"] = overall["mean_original_confidence"] - overall["accuracy"]
    overall["signed_gap_ci95"] = paired_bootstrap([r["p"] - r["y"] for r in records],
                                                  [r["domain"] for r in records], repetitions, seed)["ci95"]
    return {"status": "post_hoc_descriptive_sensitivity", "repetitions": repetitions, "seed": seed,
            "overall_original_calibration": overall, "by_domain": domain_results,
            "calibration_cells": cells, "high_vs_low_within_domain_band": comparisons,
            "interpretation": [
                "Subject-specific checks are exploratory; four intervals are not four independent confirmations.",
                "The original fixed confidence bands are reused; no outcome-dependent bin selection or merging.",
                "Confidence and correctness are resampled as paired observations within each reliability cell.",
                "HIGH and LOW cells have different sizes and may have different confidence distributions even within a band.",
                "Absolute empirical calibration gaps are biased/noisy in small groups; intervals involving absolute values are descriptive, especially near zero.",
                "All cells are reported; n<20 cells are flagged and should not receive inferential interpretation.",
                "A better-calibrated HIGH group would validate a useful label pattern but cannot distinguish reliability awareness from another correctness assessment.",
                "No group-level pattern labels any individual probability as truly calibrated or miscalibrated.",
                "Original primary results and their confirmation rule are unchanged."]}


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}


def record_plan():
    plan = {"version": "group-sensitivity-v1", "status": "post_hoc_after_evaluation_results_known",
            "input_hashes": hashes(), "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "domains": "all four original subjects", "contrasts": CONTRASTS, "confidence_bands": BANDS,
            "repetitions": REPETITIONS, "seed": SEED, "small_group_cutoff": 20,
            "scope": "Original four Brier contrasts by subject; original confidence calibration by subject x fixed confidence band x HIGH/LOW; no new model fits or API calls.",
            "cell_resampling": "Resample whole p,y observations within a cell; independently resample HIGH and LOW cells in each replicate for absolute-gap differences.",
            "output_rules": "Report all cells and subject contrasts, including n<20 and empty cells. No merging, outcome-selected bins, best-domain selection, or aggregate HIGH/LOW claim ignoring case mix.",
            "limits": "Descriptive uncertainty and effect concentration; cannot discriminate an internal reliability-awareness mechanism from ordinary correctness re-evaluation. Original frozen evaluation remains the main analysis."}
    fingerprint = digest(plan)
    path = DIRECTORY / "group_plan.json"
    if path.exists():
        if read_json(path)["fingerprint"] != fingerprint:
            raise ValueError("Post-hoc group plan changed; preserve previous results")
    else:
        write_json(path, {**plan, "fingerprint": fingerprint, "recorded_before_group_checks": now()})
    return plan, fingerprint


def render(result):
    def f(x):
        return "—" if x is None else f"{x:+.5f}"
    def interval(v):
        return "—" if v is None else f"[{v[0]:+.5f}, {v[1]:+.5f}]"
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Post hoc subject and calibration group checks</title>",
             "<style>body{font:16px/1.55 system-ui;max-width:1250px;margin:40px auto;padding:0 24px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ccd3d7;text-align:left}th{background:#f0f3f5}.small{color:#7b3a25}h2{margin-top:30px}</style>",
             "<h1>Post hoc subject and calibration group checks</h1><p>These checks use existing evaluation outputs. They neither change the main result nor establish a metacognitive mechanism. Intervals are descriptive and unadjusted for the multiple subgroup comparisons.</p>",
             "<h2>Original Brier contrasts by subject</h2><p>Negative differences favor the reliability-based method. Each subject contains 400 items.</p>",
             "<table><tr><th>Subject</th><th>Contrast</th><th>Difference</th><th>95% interval</th></tr>"]
    labels = {"matched_one_call": "Primary: reliability vs reassessment", "matched_two_calls": "Two reviews: reassessment + reliability vs repeated reassessment",
              "incremental_r": "Add reliability to reassessment", "reliability_vs_confidence_only": "Reliability vs confidence-only calibration"}
    for row in result["by_domain"]:
        for key, v in row["contrasts"].items():
            parts.append(f"<tr><td>{html.escape(row['domain'])}</td><td>{labels[key]}</td><td>{f(v['mean_difference'])}</td><td>{interval(v['ci95'])}</td></tr>")
    parts.append("</table><h2>Original confidence calibration by group</h2><p>Signed gap = mean original confidence minus accuracy. Positive values indicate overconfidence on average. Small groups (n&lt;20) are flagged; do not interpret their intervals inferentially.</p><table><tr><th>Subject</th><th>Confidence band</th><th>Reliability</th><th>n</th><th>Mean confidence</th><th>Accuracy</th><th>Signed gap</th><th>95% interval</th></tr>")
    for row in result["calibration_cells"]:
        cells = (row["domain"], row["band"], row["reliability"], str(row["n"]) + (" (small)" if row["small_group"] else ""),
                 f(row["mean_original_confidence"]), f(row["accuracy"]), f(row["signed_gap"]), interval(row["signed_gap_ci95"]))
        parts.append("<tr>" + "".join("<td>" + html.escape(x) + "</td>" for x in cells) + "</tr>")
    parts.append("</table><h2>HIGH versus LOW absolute calibration gaps</h2><p>Negative values mean the HIGH group has a smaller empirical absolute gap. Unequal sample sizes, different confidence distributions within bins and noise near zero limit this comparison. It cannot separate reliability awareness from ordinary answer checking.</p><table><tr><th>Subject</th><th>Band</th><th>HIGH / LOW n</th><th>Mean p difference</th><th>Absolute-gap difference</th><th>95% interval</th></tr>")
    for row in result["high_vs_low_within_domain_band"]:
        cells = (row["domain"], row["band"], f"{row['n_high']} / {row['n_low']}" + (" (small cell)" if row["any_small_group"] else ""),
                 f(row["mean_p_high_minus_low"]), f(row["absolute_gap_high_minus_low"]), interval(row["ci95"]))
        parts.append("<tr>" + "".join("<td>" + html.escape(x) + "</td>" for x in cells) + "</tr>")
    parts.append("</table><h2>Interpretation limits</h2><ul>" + "".join("<li>" + html.escape(x) + "</li>" for x in result["interpretation"]) + "</ul><p><a href='group_plan.json'>Recorded post hoc plan</a> · <a href='group_results.json'>Complete results</a></p></html>")
    (DIRECTORY / "group_report.html").write_text("\n".join(parts))


def main():
    plan, fingerprint = record_plan()
    records, saved = read_json(INPUTS[0]), read_json(INPUTS[1])
    result = analyze_groups(records, saved["predictions"])
    if hashes() != plan["input_hashes"]:
        raise ValueError("Original evaluation files changed")
    result.update(created=now(), plan_fingerprint=fingerprint, source_files_unchanged=True)
    write_json(DIRECTORY / "group_results.json", result)
    render(result)
    print(json.dumps({"subjects": result["by_domain"], "calibration_cells": len(result["calibration_cells"]),
                      "small_cells": sum(r["small_group"] for r in result["calibration_cells"]), "source_files_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
