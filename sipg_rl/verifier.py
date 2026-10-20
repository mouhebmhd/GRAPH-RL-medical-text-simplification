"""Component 4: post-hoc fidelity verification on critical spans (numbers/dosages, negations,
critical-term list) plus an optional NLI/entailment check."""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from .lexicon import CRITICAL_TERMS, APPROVED_SUBS, NEGATION_CUES, UNITS, HEDGE_CUES, COND_CUES

_NUM_UNIT = re.compile(rf"(\d+(?:\.\d+)?)\s*({UNITS})?\b", re.I)
_WORD = re.compile(r"[a-z]+(?:n't)?|n't", re.I)


@dataclass
class VerifierResult:
    ok: bool
    violations: list = field(default_factory=list)


class FidelityVerifier:
    def __init__(self, critical_terms=None, approved_subs=None, nli=None, nli_threshold=0.5):
        self.terms = set(critical_terms or CRITICAL_TERMS)
        self.subs = approved_subs if approved_subs is not None else APPROVED_SUBS
        self.nli, self.nli_threshold = nli, nli_threshold

    # -- span extraction ---------------------------------------------------------------
    @staticmethod
    def numbers(text):
        return Counter((m.group(1), (m.group(2) or "").lower()) for m in _NUM_UNIT.finditer(text))

    @staticmethod
    def negations(text):
        low = text.lower().replace("n't", " n't")
        return sum(1 for w in re.findall(r"[a-z']+", low) if w in NEGATION_CUES)

    def term_counts(self, text):
        low = text.lower()
        return Counter(w for w in re.findall(r"[a-z]+", low) if w in self.terms or w in self.subs)

    # -- checks -------------------------------------------------------------------------
    def check(self, src: str, out: str, use_nli: bool = True) -> VerifierResult:
        v = []
        sn, on = self.numbers(src), self.numbers(out)
        # every (number, unit) pair must survive; no new numbers may appear. Unit may be absorbed by
        # an adjacent word split, so compare numbers alone for the "no new" direction.
        for (num, unit), c in sn.items():
            if sum(oc for (n2, u2), oc in on.items() if n2 == num and (u2 == unit or not unit)) < c:
                v.append(f"number/dose lost or altered: {num} {unit}".strip())
        src_nums = {n for n, _ in sn}
        for (num, unit) in on:
            if num not in src_nums:
                v.append(f"hallucinated number: {num}")
        if self.negations(out) < self.negations(src):
            v.append("negation cue dropped")
        cw = lambda t, cues: sum(1 for w in re.findall(r"[a-z']+", t.lower()) if w in cues)
        if cw(out, HEDGE_CUES) < cw(src, HEDGE_CUES):
            v.append("hedge dropped (e.g. 'may', 'possible')")
        if cw(out, COND_CUES) < cw(src, COND_CUES):
            v.append("conditional dropped ('if'/'unless')")
        st, ot = self.term_counts(src), self.term_counts(out)
        for term, c in st.items():
            if term in self.subs:  # approved substitution: original or approved plain form
                plain_ok = sum(len(re.findall(re.escape(p), out.lower())) for p in self.subs[term])
                if ot.get(term, 0) + plain_ok < c:
                    v.append(f"critical abbreviation lost: {term}")
            elif ot.get(term, 0) < c:
                v.append(f"critical term lost/substituted: {term}")
        for term in ot:
            if term in self.terms and term not in st:
                v.append(f"hallucinated critical term: {term}")
        if use_nli and self.nli is not None and not v and (st or sn):
            p = self.nli(src, out)  # dict with 'entailment' probability
            if p.get("entailment", 1.0) < self.nli_threshold:
                v.append(f"NLI entailment {p.get('entailment'):.2f} < {self.nli_threshold}")
        return VerifierResult(not v, v)


class HFNLI:
    """Optional adapter around a Hugging Face NLI model (needs `transformers`, network on first use)."""
    def __init__(self, model="microsoft/deberta-large-mnli"):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model, top_k=None)

    def __call__(self, premise, hypothesis):
        res = self.pipe({"text": premise, "text_pair": hypothesis})
        res = res[0] if isinstance(res[0], list) else res
        return {r["label"].lower(): r["score"] for r in res}
