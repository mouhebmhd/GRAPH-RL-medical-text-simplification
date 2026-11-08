"""P1.3 - confidence intervals for every headline metric (pure numpy; no scipy needed)."""
from __future__ import annotations
import numpy as np


def paired_bootstrap(a, b, n=5000, seed=0):
    """Paired bootstrap of mean(a - b) over examples. Returns (mean_diff, lo, hi, p) with a 95% CI and a two-sided
    bootstrap p-value (2 * min(P(mean<=0), P(mean>=0)))."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    p = 2 * min(float((bs <= 0).mean()), float((bs >= 0).mean()))
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), min(1.0, max(p, 1.0 / n))


def mean_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if v is not None and v == v], float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family {name: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def mean_std_across_seeds(values):
    """Seed-level summary: mean, std (ddof=1), and a t-based 95% CI half-width over per-seed means."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan")
    from scipy.stats import t
    sd = float(v.std(ddof=1))
    return float(v.mean()), sd, float(t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)))


def holm_bonferroni(p_values: dict, alpha: float = 0.05) -> dict:
    """Task 18: {name: p} -> {name: {"adj_p": ..., "reject": bool}} controlling the family-wise error rate at `alpha`."""
    adj = holm(p_values)
    return {k: dict(adj_p=adj[k], reject=adj[k] < alpha) for k in p_values}
