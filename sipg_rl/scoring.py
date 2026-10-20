"""Reward components: domain Suitability Scorer S(y) and semantic-fidelity signal Sim(y, x).

`FeatureSuitabilityScorer` is the *operationalised* fallback named in the proposal (Table 2) for when
the existing Suitability Scorer is not callable: any object with __call__(text) -> [0,1] can replace it.
`ConceptEncoder` is an offline stand-in for a clinical sentence encoder (swap in sentence-transformers)."""
from __future__ import annotations
import re
import numpy as np
from .lexicon import PLAIN_TERMS, ADJ_REL, CRITICAL_TERMS
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
