"""P2.1 - selective prediction: risk-coverage curves, AURC, calibration (ECE) and reliability bins.
`scores`: confidence that the item is CORRECT (higher = keep). `correctness`: 1 correct / 0 wrong."""
from __future__ import annotations
import numpy as np


def risk_coverage(scores, correctness, coverages=np.linspace(0.1, 1.0, 19)):
    """Sort by confidence desc; at each coverage c keep the top c*N and report the ERROR RATE on the accepted set."""
    scores, correctness = np.asarray(scores, float), np.asarray(correctness, float)
    order = np.argsort(-scores, kind="stable")
    out = []
    for c in coverages:
        k = max(1, int(round(c * len(scores))))
        out.append((float(c), float(1.0 - correctness[order[:k]].mean())))
    return out


def aurc(curve) -> float:
    """Area under the risk-coverage curve (lower is better; a perfect ranker has risk 0 until coverage = accuracy)."""
    x, y = np.array([c for c, _ in curve]), np.array([r for _, r in curve])
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(y, x) / (x[-1] - x[0])) if len(x) > 1 else float(y[0])


def reliability(scores, correctness, n_bins=15):
    scores, correctness = np.asarray(scores, float), np.asarray(correctness, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)
    return [dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=int((idx == b).sum()),
                 conf=float(scores[idx == b].mean()), acc=float(correctness[idx == b].mean())) for b in range(n_bins) if (idx == b).any()]


def ece(scores, correctness, n_bins=15) -> float:
    """Expected calibration error: sum_b (n_b / N) * |acc_b - conf_b|. Scores must be probabilities in [0, 1]."""
    rel, N = reliability(scores, correctness, n_bins), len(scores)
    return float(sum(b["n"] / N * abs(b["acc"] - b["conf"]) for b in rel))


def auroc(scores, correctness) -> float:
    s, y = np.asarray(scores, float), np.asarray(correctness, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def fit_logistic(X, y, l2=1e-2, iters=300, lr=0.5):
    """Tiny numpy logistic regression (features standardised). Returns a predict(X)->P(correct) closure."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]])
    return lambda Xn: 1 / (1 + np.exp(-np.c_[np.ones(len(Xn)), (np.asarray(Xn, float) - mu) / sd] @ w))
