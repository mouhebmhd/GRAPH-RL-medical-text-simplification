"""Task 11 - S-ranked candidate selection: among candidate wordings for a span, pick the one that gives the best suitability S
of the resulting text; ties broken by corpus count (then original order)."""
from __future__ import annotations


def select_candidate(cands, S_of):
    """cands: [(layman, count|None)]. S_of(layman) -> S of the sentence with that wording. Returns (layman, rank, n_candidates, count)."""
    scored = [(S_of(w), (c or 0), -i, w, c) for i, (w, c) in enumerate(cands)]
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    best = scored[0]
    return best[3], 1, len(cands), best[4]
