"""Evaluation metrics for text simplification (stdlib + numpy only, so they are testable without torch/spaCy).

SARI follows Xu et al. (2016): n-gram (n=1..4) KEEP F1, DELETE precision, ADD F1, averaged, single reference.
It is a faithful re-implementation of the widely used reference code, NOT a byte-for-byte copy: for numbers you
publish, cross-check with EASSE (`pip install easse`). BLEU is deliberately not re-implemented (use sacrebleu).
"""
from __future__ import annotations
import ast, json, re
from collections import Counter
import numpy as np
from .utils import syllables, WORD_RE

_TOK = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")


def tokenize(s: str) -> list:
    return _TOK.findall((s or "").lower())


def _ngrams(toks, n):
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def _sari_n(s, c, r):
    """One n-gram order, single reference. Returns (keep_f1, del_precision, add_f1)."""
    sc, cc, rc = Counter(s), Counter(c), Counter(r)
    # KEEP
    keep_c, keep_good, keep_all = sc & cc, (sc & cc) & rc, sc & rc
    kp = sum(keep_good[g] / keep_c[g] for g in keep_good) / len(keep_c) if len(keep_c) else 1.0
    kr = sum(keep_good.values()) / sum(keep_all.values()) if sum(keep_all.values()) else 1.0
    keep = 2 * kp * kr / (kp + kr) if kp + kr else 0.0
    # DELETE (precision only, as in the original)
    del_c = sc - cc
    del_good = del_c - rc
    dp = sum(del_good[g] / del_c[g] for g in del_good) / len(del_c) if len(del_c) else 1.0
    # ADD
    add_c, ss, rs = set(cc) - set(sc), set(sc), set(rc)
    add_good, add_all = add_c & rs, rs - ss
    ap = len(add_good) / len(add_c) if add_c else 1.0
    ar = len(add_good) / len(add_all) if add_all else 1.0
    add = 2 * ap * ar / (ap + ar) if ap + ar else 0.0
    return keep, dp, add


def sari(source: str, output: str, reference: str) -> dict:
    s, c, r = tokenize(source), tokenize(output), tokenize(reference)
    parts = [_sari_n(_ngrams(s, n), _ngrams(c, n), _ngrams(r, n)) for n in range(1, 5)]
    keep, dele, add = (float(np.mean([p[i] for p in parts])) for i in range(3))
    return dict(sari=100 * (keep + dele + add) / 3, keep=100 * keep, delete=100 * dele, add=100 * add)


def rouge_l(output: str, reference: str) -> float:
    """ROUGE-L F1 (LCS based), 0-100."""
    a, b = tokenize(output), tokenize(reference)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 100 * 2 * p * r / (p + r)


# ------------------------------------------------------------------------------------------------ readability
def _counts(text):
    sents = [x for x in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if x]
    words = WORD_RE.findall(text or "")
    return max(1, len(sents)), words


def fkgl(text: str) -> float:
    ns, w = _counts(text)
    if not w:
        return 0.0
    return 0.39 * len(w) / ns + 11.8 * sum(syllables(x) for x in w) / len(w) - 15.59


def ari(text: str) -> float:
    ns, w = _counts(text)
    if not w:
        return 0.0
    return 4.71 * sum(len(x) for x in w) / len(w) + 0.5 * len(w) / ns - 21.43


# ------------------------------------------------------------------------------------------------ safety (independent of the verifier)
_NEG = {"no", "not", "never", "without", "none", "denies", "denied", "deny", "negative", "absent", "cannot",
        "neither", "nor", "unable"}
_NUMRE = re.compile(r"\d+(?:\.\d+)?")


def numbers_kept(src: str, out: str) -> bool:
    """Every number in the source still occurs (with at least the same multiplicity) in the output."""
    return not (Counter(_NUMRE.findall(src)) - Counter(_NUMRE.findall(out)))


def negations_kept(src: str, out: str) -> bool:
    cnt = lambda t: sum(1 for w in re.findall(r"[a-z']+", t.lower()) if w in _NEG or w.endswith("n't"))
    return cnt(out) >= cnt(src)


# ------------------------------------------------------------------------------------------------ Med-EASi term fields
def parse_terms(v) -> list:
    """Forgiving parser for the `expert_terms` / `layman_terms` string fields (format not assumed)."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        items = list(v)
    else:
        s = str(v).strip()
        items = []
        try:
            obj = json.loads(s)
        except Exception:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                obj = None
        if isinstance(obj, (list, tuple, set)):
            items = list(obj)
        elif isinstance(obj, dict):
            items = list(obj.keys())
        else:
            q = re.findall(r"'([^']+)'|\"([^\"]+)\"", s)
            items = [a or b for a, b in q] or [x for x in re.split(r"[;,|]", s) if x.strip()]
    flat = []
    for it in items:
        flat += [str(x) for x in it] if isinstance(it, (list, tuple)) else [str(it)]
    return [t.strip().lower() for t in flat if len(t.strip()) >= 3]


def term_retention(terms: list, text: str) -> float | None:
    """Share of `terms` that still occur in `text` (None when there are no terms)."""
    if not terms:
        return None
    low = (text or "").lower()
    return sum(1 for t in terms if t in low) / len(terms)


# ------------------------------------------------------------------------------------------------ statistics
def mean_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_diff_ci(a, b, n=5000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
