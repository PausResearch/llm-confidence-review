"""Frozen analysis for the item-level confidence-reliability pilot.

Fit simple ridge logistic calibration mappings on the old calibration split;
score them on old development items. These splits were explored previously:
this is an exploratory pilot, NOT a fresh confirmatory evaluation. No evaluation
items are accepted. The target is incremental predictive information, not an
individual trial's 'true calibration' or privileged introspection.

All features/scales and domain indicators are fitted on calibration items only.
Penalty = summed Bernoulli negative log likelihood + .5 * sum(beta[1:]**2).
No hyperparameter/model selection is performed. Confidence logits are clipped
at 1e-4 and 1-1e-4. Each optional signal contributes a standardized main effect
and interaction with standardized logit(original confidence), allowing reliable
LOW confidence to predict lower accuracy. Intercept is unpenalized.

Bootstrap intervals resample development items in matched pairs, stratified by
domain. They are conditional on the already fitted mappings: they do not include
calibration-training uncertainty, prompt/model variability, or dataset shift.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

EPSILON = 1e-4
RIDGE = 1.0
BOOTSTRAPS = 5000
SEED = 20261115
MODELS = {"p": (), "p_q": ("q",), "p_r": ("r",),
          "p_q_r": ("q", "r"), "p_q_q2": ("q", "q2")}
CONTRASTS = {
    "matched_one_call": ("p_r", "p_q"),
    "matched_two_calls": ("p_q_r", "p_q_q2"),
    "incremental_r": ("p_q_r", "p_q"),
}


def validate_rows(rows):
    if not rows:
        raise ValueError("No complete rows to analyze")
    seen = set()
    for row in rows:
        if row["split"] not in ("calibration", "development"):
            raise ValueError("Only calibration/development items are permitted; evaluation remains sealed")
        if not isinstance(row["id"], str) or row["id"] in seen:
            raise ValueError("Each item must have a unique string id")
        seen.add(row["id"])
        if not isinstance(row["domain"], str) or not row["domain"]:
            raise ValueError("Domain must be a nonempty string")
        for key in ("p", "q", "q2", "r", "y"):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {key} for {row['id']}")
        if row["r"] not in (0, 1) or row["y"] not in (0, 1):
            raise ValueError("Reliability and correctness must be binary")
    train = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "development"]
    if not train or not test:
        raise ValueError("Both calibration and development items are required")
    if len({row["y"] for row in train}) != 2:
        raise ValueError("Calibration fit requires both correct and incorrect answers")
    domains = {row["domain"] for row in train}
    if any(row["domain"] not in domains for row in test):
        raise ValueError("Development domain absent from calibration data")


def transformed(rows, key):
    values = np.array([row[key] for row in rows], dtype=float)
    if key == "r":
        return values
    values = np.clip(values, EPSILON, 1 - EPSILON)
    return np.log(values) - np.log1p(-values)


def design(rows, spec):
    n = len(rows)
    columns = [np.ones(n)]
    for domain in spec["domains"][1:]:
        columns.append(np.array([row["domain"] == domain for row in rows], float))
    def standardized(key):
        scale = spec["scalers"][key]
        return (transformed(rows, key) - scale["mean"]) / scale["scale"]
    p = standardized("p")
    columns.append(p)
    for key in spec["signals"]:
        signal = standardized(key)
        columns.extend((signal, signal * p))
    if any(row["domain"] not in spec["domains"] for row in rows):
        raise ValueError("Cannot predict an unseen domain")
    return np.column_stack(columns)


def fit(train, signals=()):
    """Fit on calibration only; return a JSON-serializable frozen model."""
    if not train or any(row["split"] != "calibration" for row in train):
        raise ValueError("Fits must use calibration rows exclusively")
    if len({row["y"] for row in train}) != 2:
        raise ValueError("Calibration fit requires both outcome classes")
    domains = sorted({row["domain"] for row in train})
    names = ["intercept"] + ["domain=" + d for d in domains[1:]] + ["logit_p"]
    scalers = {}
    for key in ("p",) + tuple(signals):
        values = transformed(train, key)
        std = float(values.std())
        scalers[key] = {"mean": float(values.mean()), "scale": std if std > 1e-12 else 1.0,
                        "constant": std <= 1e-12, "transform": "binary" if key == "r" else "clipped_logit"}
    for key in signals:
        names.extend((key, key + "*logit_p"))
    spec = {"domains": domains, "reference_domain": domains[0], "signals": list(signals),
            "feature_names": names, "scalers": scalers, "ridge": RIDGE, "clip": EPSILON}
    x, y = design(train, spec), np.array([row["y"] for row in train], float)
    def objective(beta):
        # Explicit reductions avoid platform BLAS floating-point-status warnings
        # seen on otherwise finite small matrices with some macOS NumPy builds.
        z = np.einsum("ij,j->i", x, beta)
        penalty = beta.copy()
        penalty[0] = 0
        value = np.sum(np.logaddexp(0, z) - y * z) + .5 * RIDGE * np.sum(penalty ** 2)
        grad = np.einsum("ij,i->j", x, expit(z) - y) + RIDGE * penalty
        return float(value), grad
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError("Calibration optimization failed: " + str(result.message))
    spec.update(coefficients=result.x.tolist(), n_fit=len(train), optimizer_message=str(result.message),
                objective=float(result.fun), max_abs_gradient=float(np.max(np.abs(result.jac))))
    return spec


def predict(spec, rows):
    return expit(np.einsum("ij,j->i", design(rows, spec), np.array(spec["coefficients"])))


def paired_bootstrap(differences, domains, repetitions=BOOTSTRAPS, seed=SEED):
    """Paired loss difference; resample within domains preserving their sizes."""
    differences = np.asarray(differences, float)
    domains = np.asarray(domains)
    if differences.ndim != 1 or len(differences) != len(domains) or not len(differences):
        raise ValueError("Matched nonempty difference/domain arrays required")
    if not np.all(np.isfinite(differences)) or repetitions < 1:
        raise ValueError("Finite differences and positive repetitions required")
    rng = np.random.default_rng(seed)
    totals = np.zeros(repetitions)
    for domain in sorted(set(domains.tolist())):
        values = differences[domains == domain]
        indices = rng.integers(0, len(values), size=(repetitions, len(values)))
        totals += values[indices].sum(axis=1)
    samples = totals / len(differences)
    return {"mean_difference": float(differences.mean()),
            "ci95": [float(v) for v in np.quantile(samples, [.025, .975])],
            "n_items": len(differences), "n_domains": len(set(domains.tolist())),
            "bootstrap_repetitions": repetitions, "seed": seed,
            "direction": "negative favors first model", "conditional_on_fitted_models": True}


def losses(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return {"brier": (p - y) ** 2,
            "log_loss": -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))}


def metrics(p, y, ids):
    p, y = np.asarray(p, float), np.asarray(y, float)
    result = {key: float(value.mean()) for key, value in losses(p, y).items()}
    positive = y == 1
    npos, nneg = int(positive.sum()), int((~positive).sum())
    result["auroc"] = float((rankdata(p)[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None
    order = sorted(range(len(ids)), key=lambda i: (-p[i], hashlib.sha256(ids[i].encode()).hexdigest()))
    result["selective_accuracy"] = {}
    for coverage in (.5, .75, .9):
        count = int(math.ceil(coverage * len(ids)))
        chosen = order[:count]
        result["selective_accuracy"][str(coverage)] = {
            "accepted": count, "realized_coverage": count / len(ids),
            "accuracy": float(y[chosen].mean()), "errors": int(np.sum(y[chosen] == 0))}
    keep = p >= .8
    result["utility_secondary"] = {"threshold": .8, "correct_reward": 1, "wrong_reward": -4,
        "accepted": int(keep.sum()), "points": float(np.sum(keep * (5 * y - 4))),
        "points_per_item": float(np.mean(keep * (5 * y - 4))),
        "accepted_accuracy": float(y[keep].mean()) if np.any(keep) else None}
    return result


def descriptions(rows):
    result = {}
    for split in ("calibration", "development"):
        part = [row for row in rows if row["split"] == split]
        r = np.array([row["r"] for row in part])
        p = np.array([row["p"] for row in part])
        q = np.array([row["q"] for row in part])
        q2 = np.array([row["q2"] for row in part])
        result[split] = {"n": len(part), "high_count": int(r.sum()), "high_fraction": float(r.mean()),
                        "q_equals_p_fraction": float(np.mean(q == p)),
                        "q2_equals_p_fraction": float(np.mean(q2 == p)),
                        "q_equals_q2_fraction": float(np.mean(q == q2)),
                        "q_mean_absolute_revision": float(np.mean(abs(q - p))),
                        "q_vs_q2_mean_absolute_difference": float(np.mean(abs(q - q2)))}
    groups = []
    # This is descriptive subgroup calibration, not a label of single-trial reliability.
    for domain in sorted({row["domain"] for row in rows}):
        for lower, upper in ((0, .6), (.6, .8), (.8, 1.000001)):
            for high in (0, 1):
                group = [row for row in rows if row["split"] == "development" and row["domain"] == domain
                         and lower <= row["p"] < upper and row["r"] == high]
                if group:
                    mean_p = float(np.mean([row["p"] for row in group]))
                    accuracy = float(np.mean([row["y"] for row in group]))
                    groups.append({"domain": domain, "p_band": [lower, min(upper, 1)], "reliability": "HIGH" if high else "LOW",
                                   "n": len(group), "mean_original_confidence": mean_p, "accuracy": accuracy,
                                   "signed_confidence_minus_accuracy": mean_p - accuracy,
                                   "absolute_group_gap": abs(mean_p - accuracy)})
    result["development_groups_descriptive_only"] = groups
    return result


def analyze(rows):
    """Return fit specs, development predictions, scores and paired contrasts."""
    validate_rows(rows)
    # Stable order makes exports and floating point reductions reproducible.
    rows = sorted(rows, key=lambda row: row["id"])
    train = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "development"]
    fitted = {name: fit(train, signals) for name, signals in MODELS.items()}
    predictions = {name: predict(spec, test) for name, spec in fitted.items()}
    predictions.update({"raw_" + key: np.array([row[key] for row in test]) for key in ("p", "q", "q2")})
    y, ids = np.array([row["y"] for row in test]), [row["id"] for row in test]
    domains = [row["domain"] for row in test]
    all_losses = {name: losses(p, y) for name, p in predictions.items()}
    contrasts = {}
    for name, (first, second) in CONTRASTS.items():
        contrasts[name] = {"first_model": first, "second_model": second,
                           "interpretation": "negative loss difference favors first model"}
        for metric in ("brier", "log_loss"):
            contrasts[name][metric] = paired_bootstrap(all_losses[first][metric] - all_losses[second][metric], domains)
    return {"analysis_version": "item_confidence_v1", "status": "exploratory_pilot",
            "primary": "matched_one_call.brier", "key_specificity_check": "matched_two_calls.brier",
            "fit_n": len(train), "score_n": len(test), "domains": sorted(set(domains)),
            "models": fitted, "scores": {name: metrics(p, y, ids) for name, p in predictions.items()},
            "contrasts": contrasts, "diagnostics": descriptions(rows),
            "predictions": [{"id": row["id"], "domain": row["domain"], "y": row["y"],
                             "p": row["p"], "q": row["q"], "q2": row["q2"], "r": row["r"],
                             "fitted_predictions": {name: float(p[i]) for name, p in predictions.items()}}
                            for i, row in enumerate(test)],
            "limitations": ["Old development labels informed earlier pilots; this is not confirmatory evidence.",
                "Bootstrap intervals condition on fitted mappings and do not include training or model-generation uncertainty.",
                "Positive prediction gains do not establish privileged introspection or a distinct higher-order mechanism.",
                "No gain does not prove absence of metacognitive capacity.",
                "Subgroup calibration gaps are descriptive and noisy; no individual trial is labeled miscalibrated.",
                "Original/revised confidence is clipped for logistic fitting, not for raw Brier scoring.",
                "Secondary outcomes are descriptive; no multiple-outcome significance claims are warranted."]}
