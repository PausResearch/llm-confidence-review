from __future__ import annotations
import numpy as np
from scipy.stats import binom

def band(p):
    p = np.asarray(p, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probability")
    return np.minimum(np.searchsorted([.6, .8, 1.0], p, side="right"), 2)

def summaries(p):
    p = np.asarray(p, dtype=float)
    groups = band(p)
    return {"n": len(p), "counts": [int((groups == k).sum()) for k in range(3)],
            "mean_p": [float(p[groups == k].mean()) if np.any(groups == k) else None for k in range(3)]}

def aggregate(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return [float(q[band(p) == k].mean()) if np.any(band(p) == k) else None for k in range(3)]

def fit_audit(p, y, strength=10):
    p, y = np.asarray(p), np.asarray(y)
    result = summaries(p)
    result["rates"] = [float((y[band(p) == k].sum() + strength * result["mean_p"][k]) / (result["counts"][k] + strength)) if result["counts"][k] else None for k in range(3)]
    return result

def audited(p, mapping):
    p = np.asarray(p, dtype=float)
    result = p.copy()
    for k, rate in enumerate(mapping["rates"]):
        if rate is not None:
            result[band(p) == k] = rate
    return result

def utility(p, y):
    return float(np.sum((np.asarray(p) >= .8) * (5 * np.asarray(y) - 4)))

def released_stats(p, y):
    chosen = np.asarray(p) >= .8
    y = np.asarray(y)
    n = int(chosen.sum())
    return {"accepted": n, "wrong_released": int((chosen & (y == 0)).sum()),
            "accepted_accuracy": float(y[chosen].mean()) if n else None,
            "coverage": float(chosen.mean())}

def forecast_loss(p, y, rates):
    p, y = np.asarray(p), np.asarray(y)
    return float(sum((band(p) == k).sum() * (rates[k] - y[band(p) == k].mean()) ** 2
                     for k in range(3) if np.any(band(p) == k)) / len(p))

def expected_gain(p, rates, cal_summary, strength=10):
    """Plug-in binomial reference; not a hindsight oracle or full Bayes policy."""
    p = np.asarray(p)
    result = 0.0
    for k in range(3):
        nt, nc = int((band(p) == k).sum()), cal_summary["counts"][k]
        if not nt or not nc:
            continue
        q = float(rates[k])
        cutoff = int(np.ceil(.8 * (nc + strength) - strength * cal_summary["mean_p"][k] - 1e-10))
        accept_prob = float(binom.sf(cutoff - 1, nc, q))
        result += nt * (accept_prob - float(k == 2)) * (5 * q - 4)
    return float(result)

def bootstrap_mean(values, domains, n=2000, seed=20261115):
    values, domains = np.asarray(values, dtype=float), np.asarray(domains)
    if not len(values):
        return {"mean": None, "ci95": None, "batches": 0}
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(domains == d) for d in sorted(set(domains))]
    draws = np.concatenate([rng.choice(ix, (n, len(ix)), replace=True) for ix in indices], axis=1)
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(), "batches": len(values)}

def ridge_oof(x, y, folds, ridge=10.0, weights=None):
    x, y, folds = np.asarray(x, float), np.asarray(y, float), np.asarray(folds)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights)
    if not all(np.all(np.isfinite(a)) for a in [x,y,weights]) or np.any(weights<0):
        raise ValueError("Cross-fit inputs and weights must be finite; weights must be nonnegative")
    pred = np.empty(len(y))
    for f in sorted(set(folds)):
        tr, te = (folds != f) & (weights > 0), folds == f
        if tr.sum() < 2:
            raise ValueError("Not enough training batches for cross fitting")
        w = weights[tr]
        mean = np.average(x[tr], axis=0, weights=w)
        sd = np.sqrt(np.average((x[tr] - mean) ** 2, axis=0, weights=w))
        sd[sd < 1e-10] = 1
        z = np.column_stack([np.ones(tr.sum()), (x[tr] - mean) / sd])
        penalty = np.eye(z.shape[1]) * ridge
        penalty[0, 0] = 0
        # Small matrices: explicit contractions avoid platform-specific BLAS
        # floating-point warnings and preserve the weighted ridge equation.
        gram = np.einsum("ni,n,nj->ij",z,w,z,optimize=False) + penalty
        rhs = np.einsum("ni,n,n->i",z,w,y[tr],optimize=False)
        beta = np.linalg.solve(gram,rhs)
        test = np.column_stack([np.ones(te.sum()), (x[te] - mean) / sd])
        pred[te] = np.einsum("ni,i->n",test,beta,optimize=False)
    if not np.all(np.isfinite(pred)):
        raise ValueError("Nonfinite cross-fit prediction; inspect data and numerical environment")
    return pred
