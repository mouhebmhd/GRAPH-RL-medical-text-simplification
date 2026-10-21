"""Reward components: domain Suitability Scorer S(y) and semantic-fidelity signal Sim(y, x).

`FeatureSuitabilityScorer` is the *operationalised* fallback named in the proposal (Table 2) for when
the existing Suitability Scorer is not callable: any object with __call__(text) -> [0,1] can replace it.
`ConceptEncoder` is an offline stand-in for a clinical sentence encoder (swap in sentence-transformers)."""
from __future__ import annotations
import re
import numpy as np
from .lexicon import PLAIN_TERMS, ADJ_REL, CRITICAL_TERMS, NEGATION_CUES, HEDGE_CUES, COND_CUES
from .utils import syllables, WORD_RE, hash_vec

_JARGON_KEYS = sorted(list(PLAIN_TERMS) + list(ADJ_REL), key=len, reverse=True)
_JARGON_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in _JARGON_KEYS) + r")\b", re.I)


class FeatureSuitabilityScorer:
    """S(y) in [0,1] (report x100 for the 'X%' of the proposal). Critical terms/numbers are masked from
    length/jargon penalties because they cannot (and must not) be simplified."""

    def __init__(self, target_grade: float = 6.0, weights=(0.35, 0.15, 0.20, 0.30)):
        self.tg, self.w = target_grade, weights

    def features(self, text: str) -> dict:
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        n_sent = max(1, len(sents))
        words = WORD_RE.findall(text)
        n_words = max(1, len(words))
        syl = [1 if (w[0].isdigit() or w.lower() in CRITICAL_TERMS) else syllables(w) for w in words]
        asl, asw = n_words / n_sent, sum(syl) / n_words
        fk = 0.39 * asl + 11.8 * asw - 15.59
        poly = sum(1 for s in syl if s >= 3) / n_words
        jargon = len(_JARGON_RE.findall(text)) / n_words
        return dict(asl=asl, asw=asw, grade=fk, poly=poly, jargon=jargon)

    def __call__(self, text: str) -> float:
        f = self.features(text)
        c = lambda x: min(1.0, max(0.0, x))
        comps = (1 - c((f["grade"] - self.tg) / 8), 1 - c((f["asl"] - 15) / 20),
                 1 - c(f["poly"] / 0.25), 1 - c(f["jargon"] / 0.08))
        return float(sum(w * x for w, x in zip(self.w, comps)))


_STOP = set("the a an of to and who has have is are was were this that be been by in on with for it as at".split())


class ConceptEncoder:
    """Hashed bag of (concept-normalised) uni+bigrams. Jargon and its approved plain wording map to the
    same concept token, mimicking what a clinical sentence encoder gives; critical terms stay literal."""

    def __init__(self, dim: int = 512, binary: bool = False, bigrams: bool = True):
        self.dim, self.binary, self.bigrams = dim, binary, bigrams
        pairs = []
        for k, (plain, _) in PLAIN_TERMS.items():
            pairs.append((plain, k))
            pairs.append((k, k))
        for adj, (phrase, concept) in ADJ_REL.items():
            pairs += [(adj, concept), (phrase, concept)]
        self.pairs = sorted({(a, b) for a, b in pairs}, key=lambda p: -len(p[0]))
        self.rx = re.compile(r"\b(?:" + "|".join(re.escape(a) for a, _ in self.pairs) + r")\b", re.I)
        self.map = {a: b for a, b in self.pairs}

    def _norm(self, text):
        return self.rx.sub(lambda m: "§" + self.map[m.group(0).lower()].replace(" ", "_"), text.lower())

    def encode(self, text: str) -> np.ndarray:
        toks = [w for w in re.findall(r"§?[a-z0-9_.\-']+", self._norm(text)) if w not in _STOP]
        feats = toks + ([a + "|" + b for a, b in zip(toks, toks[1:])] if self.bigrams else [])
        if self.binary:   # presence only: duplicating a noun phrase when splitting a sentence is not a meaning change
            v = hash_vec(sorted(set(feats)), self.dim)
        else:
            v = np.sqrt(hash_vec(feats, self.dim))  # sublinear tf
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    @staticmethod
    def cos(a, b) -> float:
        return float(np.dot(a, b))


# =====================================================================================================================
# A2 / B2: target-band suitability.  The old S was monotone ("simpler is always better") and clipped, so it saturated
# (67% of invoke-all outputs >= 0.95) and rewarded over-simplification. S_band peaks AT the target and falls off on
# BOTH sides with a smooth Gaussian, so (i) text simpler than the target is penalised, (ii) late-stage orderings keep a
# gradient (no plateau), (iii) the target is explicit and editable rather than implied by clip constants.
# Default targets follow the plain-language guidance of grade ~6-8 for patient materials; they are parameters, not facts:
# anchor them to your clinicians / reference distribution with `.fit(reference_texts)`.
# =====================================================================================================================
class BandSuitabilityScorer(FeatureSuitabilityScorer):
    FEATURES = ("grade", "asl", "poly", "jargon")
    # feature -> (centre, sigma_below, sigma_above).  sigma_below=None: values below the centre are NOT penalised
    # (polysyllable share and jargon density cannot be "too low"). Grade/length are penalised on both sides, but the lower
    # tail is deliberately tolerant: critical terms and numbers are masked to 1 syllable, so short clinical sentences
    # already read low, and loss of content is Sim's job, not S's. A strict lower tail would punish clause splitting.
    DEFAULT_BANDS = {"grade": (6.5, 3.5, 2.5), "asl": (13.0, 9.0, 6.0),
                     "poly": (0.06, None, 0.05), "jargon": (0.0, None, 0.03)}

    def __init__(self, target_grade: float = 6.5, weights=(0.35, 0.15, 0.20, 0.30), bands: dict | None = None):
        super().__init__(target_grade, weights)
        self.bands = {k: tuple(v) for k, v in (bands or self.DEFAULT_BANDS).items()}
        if bands is None:
            c, lo, hi = self.bands["grade"]
            self.bands["grade"] = (target_grade, lo, hi)

    def components(self, text: str) -> dict:
        f = self.features(text)
        out = {}
        for k in self.FEATURES:
            c, lo, hi = self.bands[k]
            d = f[k] - c
            if d < 0 and lo is None:
                out[k] = 1.0
                continue
            out[k] = float(np.exp(-0.5 * (d / (hi if d >= 0 else lo)) ** 2))
        return out

    def __call__(self, text: str) -> float:
        comps = self.components(text)
        return float(sum(w * comps[k] for w, k in zip(self.w, self.FEATURES)))

    def fit(self, texts, min_sigma=None):
        """Anchor the bands to a reference distribution: centre = median, sigma = robust spread (MAD), floored."""
        min_sigma = min_sigma or dict(grade=1.0, asl=3.0, poly=0.03, jargon=0.02)
        feats = [self.features(t) for t in texts if t and t.strip()]
        for k in self.FEATURES:
            v = np.array([f[k] for f in feats])
            c = float(np.median(v))
            sig = max(1.4826 * float(np.median(np.abs(v - c))), min_sigma[k])
            lo = self.bands[k][1]
            self.bands[k] = (0.0 if k == "jargon" else c, None if lo is None else sig, sig)
        return self

    def to_dict(self):
        return dict(weights=list(self.w), bands={k: list(v) for k, v in self.bands.items()})

    @classmethod
    def from_dict(cls, d):
        return cls(weights=tuple(d["weights"]), bands={k: tuple(v) for k, v in d["bands"].items()})


# =====================================================================================================================
# A1: length-invariant, source-anchored, safety-weighted fidelity.
# The old hashed-bigram cosine was length-dominated (a faithful, longer rewrite scored 0.5 against the source) and blind to
# critical edits (dropped doses scored as high as faithful rewrites). ContentEncoder instead computes a weighted F-measure
# over *content* tokens of the source: recall = how much of the source's content survives (concept-normalised, so jargon
# and its approved plain wording match), precision = how much of the output is source content. beta=2 weights recall 4x:
# adding clarifying words is cheap, dropping source content is expensive. Numbers, units, negations, hedges, conditionals
# and critical terms carry `crit_weight` times the weight, so losing them moves Sim a lot. It is a drop-in for
# ConceptEncoder (encode/cos). It cannot know a layman synonym that is absent from the lexicon; for that use an
# embedding/NLI model at evaluation time (the verifier's NLI adapter) - it is too slow to sit inside the reward loop.
# =====================================================================================================================
_SAFETY_TOKENS = set(NEGATION_CUES) | set(HEDGE_CUES) | set(COND_CUES) | set(CRITICAL_TERMS)
_CONNECTIVES = set("however so also then there it they he she his her their them its these those which".split())
_FUNC = _STOP | _CONNECTIVES


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


class ContentEncoder(ConceptEncoder):
    def __init__(self, dim: int = 2048, beta: float = 2.0, crit_weight: float = 3.0):
        super().__init__(dim=dim, binary=True, bigrams=False)
        self.beta, self.crit_weight = beta, crit_weight

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"§?[a-z0-9_.\-']+", self._norm(text)):
            tok = tok.strip(".-'")
            if not tok or tok in _FUNC:
                continue
            crit = tok in _SAFETY_TOKENS or tok[0].isdigit() or tok in UNIT_TOKENS or tok.endswith("n't")
            key = tok if (tok.startswith("§") or crit) else _stem(tok)
            h = int(hash_vec([key], self.dim).argmax())          # one stable slot per token
            v[h] = max(v[h], self.crit_weight if crit else 1.0)
        return v

    def cos(self, out_vec, src_vec) -> float:
        a, b = np.asarray(out_vec), np.asarray(src_vec)
        sa, sb = float(a.sum()), float(b.sum())
        if sa == 0 and sb == 0:
            return 1.0
        if sa == 0 or sb == 0:
            return 0.0
        ov = float(np.minimum(a, b).sum())
        p, r = ov / sa, ov / sb
        if p + r == 0:
            return 0.0
        b2 = self.beta ** 2
        f = (1 + b2) * p * r / (b2 * p + r)
        # critical tokens (numbers, units, negations, hedges, conditionals, drug names): the score is multiplied by
        # their recall (a lost dose/negation) and precision (an invented number/drug), so unsafe edits move Sim a lot
        ca, cb = np.where(a >= self.crit_weight, a, 0.0), np.where(b >= self.crit_weight, b, 0.0)
        sca, scb = float(ca.sum()), float(cb.sum())
        if scb > 0 or sca > 0:
            oc = float(np.minimum(ca, cb).sum())
            f *= (oc / scb if scb > 0 else 1.0) * (oc / sca if sca > 0 else 1.0)
        return float(f)


UNIT_TOKENS = {"mg", "mcg", "g", "kg", "ml", "l", "unit", "units", "iu", "%", "mmhg", "tablet", "tablets", "capsule", "capsules"}
