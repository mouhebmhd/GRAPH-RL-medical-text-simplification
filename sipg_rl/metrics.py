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
    from . import numnorm
    lost, _ = numnorm.compare(src, out)
    return not lost


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


# ================================================================================================================
# Evaluation-suite additions: SMOG, per-category critical-span checks, edit types, stats, risk-coverage
# ================================================================================================================
import difflib
from .lexicon import CRITICAL_TERMS, HEDGE_CUES, COND_CUES, NEGATION_CUES


def smog(text: str) -> float:
    """SMOG grade (McLaughlin 1969); 3+ syllable words / sentences. Needs >= ~30 sentences to be well calibrated."""
    ns, w = _counts(text)
    poly = sum(1 for x in w if syllables(x) >= 3)
    return 1.0430 * (poly * 30.0 / ns) ** 0.5 + 3.1291 if w else 0.0


CRIT_CATEGORIES = ("numbers", "units", "negations", "hedges", "conditionals", "drug_names")
_UNITS = re.compile(r"\b(?:mg|mcg|µg|g|kg|ml|l|iu|units?|mmhg|mmol|meq)\b|%", re.I)
_WORDS = lambda t: re.findall(r"[a-z']+", (t or "").lower())


def critical_report(src: str, out: str) -> dict:
    """Per category: True = preserved, False = lost, None = category absent from the source.
    Independent of the verifier (separate code), so it can audit the verifier's own output."""
    from collections import Counter
    res = {}
    ns, no = Counter(_NUMRE.findall(src)), Counter(_NUMRE.findall(out))
    res["numbers"] = None if not ns else not (ns - no)
    us = Counter(m.lower() for m in _UNITS.findall(src)); uo = Counter(m.lower() for m in _UNITS.findall(out))
    res["units"] = None if not us else not (us - uo)
    cnt = lambda t, cues: sum(1 for w in _WORDS(t) if w in cues or (cues is NEGATION_CUES and w.endswith("n't")))
    for cat, cues in (("negations", set(NEGATION_CUES) | _NEG), ("hedges", set(HEDGE_CUES)), ("conditionals", set(COND_CUES))):
        cs = cnt(src, cues)
        res[cat] = None if cs == 0 else cnt(out, cues) >= cs
    ds = [d for d in CRITICAL_TERMS if d in _WORDS(src)]
    res["drug_names"] = None if not ds else all(d in _WORDS(out) for d in ds)
    return res


def hallucination_proxy(src: str, out: str) -> bool:
    """AUTOMATIC PROXY only (not the clinician rate): the output contains a number or a drug name that the source lacks."""
    new_num = set(_NUMRE.findall(out)) - set(_NUMRE.findall(src))
    new_drug = {d for d in CRITICAL_TERMS if d in _WORDS(out)} - {d for d in CRITICAL_TERMS if d in _WORDS(src)}
    return bool(new_num or new_drug)


def edit_ops(src: str, out: str) -> list:
    """Token alignment -> [(kind, n_removed, n_added)], kind in replace / delete / insert."""
    a, b = tokenize(src), tokenize(out)
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal" and any(re.search(r"\w", t) for t in a[i1:i2] + b[j1:j2]):     # ignore punctuation-only edits
            ops.append(({"replace": "replace", "delete": "delete", "insert": "insert"}[tag], i2 - i1, j2 - j1))
    return ops


def edit_type_counts(src: str, out: str, elab_min=4) -> dict:
    """Our own alignment taxonomy, applied identically to references and systems:
    elaboration = an insertion (or replacement that grows the text) adding >= elab_min words net;
    replacement = other replace blocks; deletion; insertion = other insertions."""
    c = {"replacement": 0, "deletion": 0, "insertion": 0, "elaboration": 0}
    for kind, nr, na in edit_ops(src, out):
        if kind == "delete":
            c["deletion"] += 1
        elif kind == "insert":
            c["elaboration" if na >= elab_min else "insertion"] += 1
        else:
            c["elaboration" if na - nr >= elab_min else "replacement"] += 1
    return c


def annotation_label_counts(ann) -> dict:
    """Cross-check for the dataset's own `Annotation` field. Its exact format is NOT assumed: we only count keyword occurrences."""
    s = str(ann or "").lower()
    return {k: len(re.findall(k, s)) for k in ("elaboration", "replacement", "deletion", "insertion")}


def paired_bootstrap(a, b, n=5000, seed=0):
    """Paired bootstrap on per-example scores: mean diff, 95% CI, two-sided p, effect size d_z (diff / sd(diff))."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    return dict(diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                p=float(max(p, 1.0 / n)), dz=float(d.mean() / sd) if sd and sd == sd and sd > 0 else float("nan"))


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family {name: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ---- risk-coverage --------------------------------------------------------------------------------------------
def risk_coverage(conf, err):
    """Selective prediction: keep the most confident outputs, revert the rest to the source (error 0).
    Returns coverage[k], risk[k] = errors among kept / kept, and AURC (mean risk over coverage; lower is better)."""
    conf, err = np.asarray(conf, float), np.asarray(err, float)
    order = np.argsort(-conf, kind="stable")
    e = err[order]
    k = np.arange(1, len(e) + 1)
    risk = np.cumsum(e) / k
    return k / len(e), risk, float(risk.mean())


def risk_at_coverage(conf, err, coverage):
    cov, risk, _ = risk_coverage(conf, err)
    i = int(np.searchsorted(cov, coverage - 1e-12))
    return float(risk[min(i, len(risk) - 1)])


def ece(prob, y, bins=10):
    """Expected calibration error of predicted P(event) against the binary outcome y."""
    prob, y = np.asarray(prob, float), np.asarray(y, float)
    edges, tot = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if m.any():
            tot += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(tot)


class LogReg:
    """Tiny L2 logistic regression (numpy) used to combine signals into one calibrated error probability."""
    def fit(self, X, y, l2=1.0, iters=400, lr=0.1):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.w, self.b = np.zeros(Z.shape[1]), 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            self.w -= lr * (Z.T @ (p - y) / len(y) + l2 * self.w / len(y))
            self.b -= lr * float((p - y).mean())
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return 1 / (1 + np.exp(-(Z @ self.w + self.b)))
