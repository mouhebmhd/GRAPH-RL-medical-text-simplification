"""Task 28 - acceptance filter for an LLM span rewrite. Keeps the LLM on a short leash: a SPAN swap, never a restructure."""
from __future__ import annotations
import re

RESTRUCTURE = re.compile(r"[.!?;:\n]|\b(because|which|whereas|although|however|therefore|in order|so that)\b", re.I)


def accept_span_rewrite(span: str, new: str, max_ratio: float = 2.0, min_allow: int = 3):
    """-> (ok, reason). Rejects: empty / unchanged; longer than max(min_allow, max_ratio x span) words (an explanation, not a swap);
    sentence punctuation or restructuring markers; digits that were not in the span (invented numbers)."""
    new, span = (new or "").strip(), (span or "").strip()
    if not new:
        return False, "empty"
    if new.lower() == span.lower():
        return False, "unchanged"
    if len(new.split()) > max(min_allow, max_ratio * len(span.split())):
        return False, "longer than 2x span"
    if RESTRUCTURE.search(new):
        return False, "restructuring marker / sentence punctuation"
    if set(re.findall(r"\d+", new)) - set(re.findall(r"\d+", span)):
        return False, "new digits"
    return True, "ok"
