"""Frozen-map evaluation of the item-level confidence-reliability experiment.

This module has no fitting path. Coefficients, training-domain indicators and
scalers are supplied from the completed 480-item calibration fit. Evaluation
labels affect scores, never predictions. The caller must verify that evaluation
IDs are disjoint from the calibration/development IDs and bind the supplied maps
to the pre-evaluation protocol hash; model coefficients do not contain item IDs.

The primary contrast and secondary sequence are fixed before evaluation. Paired
bootstrap intervals condition on the supplied fitted maps. No lack of a detected
benefit is interpreted as equivalence or absence of metacognition.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from auditstudy.common import digest
from diagnostics.item_confidence_analysis import (
    BOOTSTRAPS, EPSILON, MODELS, SEED, losses, metrics, paired_bootstrap, predict,
)

CONTRASTS = {
    "matched_one_call": ("p_r", "p_q"),
    "matched_two_calls": ("p_q_r", "p_q_q2"),
    "incremental_r": ("p_q_r", "p_q"),
    "reliability_vs_confidence_only": ("p_r", "p"),
}


def validate_evaluation(rows, models):
    if not rows:
        raise ValueError("No complete evaluation rows")
    if set(models) != set(MODELS):
        raise ValueError("Expected the five frozen pilot calibration mappings")
    domains = models["p"]["domains"]
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("Invalid frozen domains")
    for name, signals in MODELS.items():
        spec = models[name]
        if spec["domains"] != domains or tuple(spec["signals"]) != signals:
            raise ValueError("Frozen mappings have inconsistent domains or signal definitions")
        if spec["n_fit"] != 480:
            raise ValueError("Mappings must be the frozen 480-item calibration fits")
        if spec["clip"] != EPSILON:
            raise ValueError("Frozen probability transform differs from the pilot")
        count = 1 + len(domains) - 1 + 1 + 2 * len(signals)
        if len(spec["coefficients"]) != count or len(spec["feature_names"]) != count:
            raise ValueError("Invalid frozen coefficient dimensions")
        if not all(math.isfinite(x) for x in spec["coefficients"]):
            raise ValueError("Nonfinite frozen coefficients")
        for signal in ("p",) + signals:
            scaler = spec["scalers"][signal]
            if not math.isfinite(scaler["mean"]) or not math.isfinite(scaler["scale"]) or scaler["scale"] <= 0:
                raise ValueError("Invalid frozen scaler")
    seen = set()
    for row in rows:
        if row["split"] != "evaluation":
            raise ValueError("Only evaluation rows may enter evaluation scoring")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("Evaluation item IDs must be unique nonempty strings")
        seen.add(row["id"])
        if row["domain"] not in domains:
            raise ValueError("Unseen evaluation domain")
        for key in ("p", "q", "q2", "r", "y"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid evaluation value: " + key)
        if row["r"] not in (0, 1) or row["y"] not in (0, 1):
            raise ValueError("Reliability and correctness must be binary")


def descriptions(rows):
    """Descriptive diagnostics, without assigning per-item calibration labels."""
    def summarize(part):
        p = np.array([row["p"] for row in part])
        q = np.array([row["q"] for row in part])
        q2 = np.array([row["q2"] for row in part])
        r = np.array([row["r"] for row in part])
        return {"n": len(part), "high_count": int(r.sum()), "high_fraction": float(r.mean()),
                "q_equals_p_fraction": float(np.mean(q == p)),
                "q2_equals_p_fraction": float(np.mean(q2 == p)),
                "q_equals_q2_fraction": float(np.mean(q == q2)),
                "q_mean_absolute_revision": float(np.mean(abs(q - p))),
                "q_vs_q2_mean_absolute_difference": float(np.mean(abs(q - q2))),
                "original_accuracy": float(np.mean([row["y"] for row in part])),
                "mean_original_confidence": float(p.mean())}
    domains = sorted({row["domain"] for row in rows})
    groups = []
    for domain in domains:
        for lower, upper in ((0, .6), (.6, .8), (.8, 1.000001)):
            for high in (0, 1):
                part = [row for row in rows if row["domain"] == domain and
                        lower <= row["p"] < upper and row["r"] == high]
                if not part:
                    continue
                mean_p = float(np.mean([row["p"] for row in part]))
                accuracy = float(np.mean([row["y"] for row in part]))
                groups.append({"domain": domain, "p_band": [lower, min(upper, 1)],
                               "reliability": "HIGH" if high else "LOW", "n": len(part),
                               "mean_original_confidence": mean_p, "accuracy": accuracy,
                               "signed_confidence_minus_accuracy": mean_p - accuracy,
                               "absolute_group_gap": abs(mean_p - accuracy)})
    return {"evaluation": summarize(rows),
            "by_domain": {domain: summarize([row for row in rows if row["domain"] == domain]) for domain in domains},
            "evaluation_groups_descriptive_only": groups}


def analyze_evaluation(rows, models):
    """Score evaluation with unchanged supplied maps; never fit or tune anything."""
    validate_evaluation(rows, models)
    original_hash = digest(models)
    rows = sorted(rows, key=lambda row: row["id"])
    predictions = {name: predict(spec, rows) for name, spec in models.items()}
    predictions.update({"raw_" + key: np.array([row[key] for row in rows]) for key in ("p", "q", "q2")})
    y = np.array([row["y"] for row in rows])
    ids, domains = [row["id"] for row in rows], [row["domain"] for row in rows]
    all_losses = {name: losses(p, y) for name, p in predictions.items()}
    contrasts = {}
    for name, (first, second) in CONTRASTS.items():
        contrasts[name] = {"first_model": first, "second_model": second,
                           "interpretation": "negative loss difference favors first model"}
        for metric in ("brier", "log_loss"):
            contrasts[name][metric] = paired_bootstrap(
                all_losses[first][metric] - all_losses[second][metric], domains,
                repetitions=BOOTSTRAPS, seed=SEED)
    primary = contrasts["matched_one_call"]["brier"]
    specificity = contrasts["matched_two_calls"]["brier"]
    primary_supported = primary["ci95"][1] < 0
    specificity_supported = primary_supported and specificity["ci95"][1] < 0
    if digest(models) != original_hash:
        raise AssertionError("Frozen models were mutated during scoring")
    return {"analysis_version": "item_confidence_evaluation_v1",
            "status": "held_out_evaluation_with_frozen_pilot_maps",
            "primary": "matched_one_call.brier", "key_specificity_check": "matched_two_calls.brier",
            "fit_n": 480, "score_n": len(rows), "domains": sorted(set(domains)),
            "frozen_models_hash": original_hash, "models": copy.deepcopy(models),
            "fit_performed": False,
            "scores": {name: metrics(p, y, ids) for name, p in predictions.items()},
            "contrasts": contrasts, "diagnostics": descriptions(rows),
            "interpretation_gate": {
                "rule": "Primary benefit requires primary Brier interval wholly below zero. A stronger specificity claim additionally requires the matched-two-call Brier interval wholly below zero. Secondary results cannot rescue the primary.",
                "primary_benefit_supported": primary_supported,
                "specificity_benefit_supported_in_fixed_sequence": specificity_supported,
                "no_equivalence_or_absence_claim": True,
                "intervals_are_conditional_bootstrap_not_exact_tests": True},
            "predictions": [{"id": row["id"], "domain": row["domain"], "y": row["y"],
                             "p": row["p"], "q": row["q"], "q2": row["q2"], "r": row["r"],
                             "fitted_predictions": {name: float(p[i]) for name, p in predictions.items()}}
                            for i, row in enumerate(rows)],
            "limitations": [
                "Intervals condition on the original 480-item fitted mappings; training-set and generation variability are omitted.",
                "The model identifier may be an alias rather than immutable weights; one model, four domains and one elicitation wording limit generalization.",
                "MMLU-Pro questions are public and may have appeared in pretraining; held out here means unused in this study before evaluation.",
                "Positive gains establish utility over these specified fitted baselines, not privileged introspection or a distinct higher-order mechanism.",
                "No detected gain establishes neither equivalence nor absence of metacognition.",
                "Group calibration gaps are descriptive; an individual incorrect answer is not by itself a miscalibrated probability.",
                "The caller must verify evaluation/calibration/development ID disjointness and frozen model provenance.",
                "Secondary contrasts, per-domain results and decision metrics cannot rescue the primary contrast.",
                "Complete-case scoring shares the same support across methods; missingness must be reported by the runner."]}
