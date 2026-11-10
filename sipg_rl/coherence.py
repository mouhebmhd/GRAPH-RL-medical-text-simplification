"""Textual-coherence metrics for the Health Literacy module: FKGL, RVS and RTS.

OPERATIONAL DEFINITIONS (written without the source PDFs at hand - if your paper defines RVS/RTS differently,
change ONLY `sentence_grades`, `rvs` and `rts` below; everything else consumes these three functions):
  * sentence grade g_i  = Flesch-Kincaid grade of sentence i (captures both sentence length and word complexity)
  * RVS (Readability Variance Score)   = population std-dev of g_1..g_n          (how much complexity varies)
  * RTS (Readability Transition Score) = mean |g_{i+1} - g_i| over adjacent pairs (how abruptly it changes)
Both are 0.0 for texts with fewer than 2 sentences. Low RVS/RTS = "steady flow". Pure functions, no model needed.
"""
from __future__ import annotations
import re
from statistics import pstdev
from .utils import syllables, WORD_RE

_SENT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split((text or "").strip()) if s]


def _grade(sentence: str) -> float:
    words = WORD_RE.findall(sentence)
    n = max(1, len(words))
    syl = sum(1 if w[0].isdigit() else syllables(w) for w in words)
    return 0.39 * n + 11.8 * (syl / n) - 15.59


def sentence_grades(text: str) -> list[float]:
    return [_grade(s) for s in split_sentences(text)]


def fkgl(text: str) -> float:
    """Document-level Flesch-Kincaid grade (same formula the Suitability Scorer uses)."""
    return _grade(text) if (text or "").strip() else 0.0


def rvs(text: str) -> float:
    g = sentence_grades(text)
    return float(pstdev(g)) if len(g) >= 2 else 0.0


def rts(text: str) -> float:
    g = sentence_grades(text)
    return float(sum(abs(b - a) for a, b in zip(g, g[1:])) / (len(g) - 1)) if len(g) >= 2 else 0.0


def calibrate_coherence(reference_texts, quantile: float = 0.75) -> dict:
    """Derive steady-flow caps from REFERENCE simplifications (e.g. Med-EASi train 'Simple' side), the same way the
    S bands are fitted: returns {'rvs_max': q, 'rts_max': q} at the given quantile over multi-sentence texts."""
    import numpy as np
    r = [(rvs(t), rts(t)) for t in reference_texts if len(split_sentences(t)) >= 2]
    if not r:
        return {}
    a = np.array(r)
    return {"rvs_max": float(np.quantile(a[:, 0], quantile)), "rts_max": float(np.quantile(a[:, 1], quantile))}
