"""Component 4: post-hoc fidelity verification on critical spans (numbers/dosages, negations,
critical-term list) plus an optional NLI/entailment check."""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from .lexicon import CRITICAL_TERMS, APPROVED_SUBS, NEGATION_CUES, UNITS, HEDGE_CUES, COND_CUES
from . import numnorm

from .normalize import number_multiset, fmt   # P0.2: 50mg == 50 mg == 50 milligrams, 4,000 == 4000
_WORD = re.compile(r"[a-z]+(?:n't)?|n't", re.I)


@dataclass
class VerifierResult:
    ok: bool
    violations: list = field(default_factory=list)


class FidelityVerifier:
    def __init__(self, critical_terms=None, approved_subs=None, nli=None, nli_threshold=0.5,
                 nli_scope="all", nli_mode="forward", enabled=True):
        """nli_scope: 'all' (default) checks EVERY rewrite; 'critical' only sentences whose source has a number / drug name
        (the original design - it never sees drift like 'mild' -> 'severe' in a sentence without critical spans).
        nli_mode: 'forward' src=>out ; 'bidirectional' also out=>src (asymmetry = meaning changed; stricter, more false positives)."""
        self.terms = set(critical_terms or CRITICAL_TERMS)
        self.subs = approved_subs if approved_subs is not None else APPROVED_SUBS
        self.nli, self.nli_threshold = nli, nli_threshold
        self.nli_scope, self.nli_mode = nli_scope, nli_mode
        self.enabled = enabled     # Task 3: cfg.verifier.enabled=False -> always pass (ablation only; never ship)

    # -- span extraction ---------------------------------------------------------------
    @staticmethod
    def numbers(text):
        return number_multiset(text)

    @staticmethod
    def negations(text):
        low = text.lower().replace("n't", " n't")
        return sum(1 for w in re.findall(r"[a-z']+", low) if w in NEGATION_CUES)

    def term_counts(self, text):
        low = text.lower()
        return Counter(w for w in re.findall(r"[a-z]+", low) if w in self.terms or w in self.subs)

    # -- checks -------------------------------------------------------------------------
    def check(self, src: str, out: str, use_nli: bool = True) -> VerifierResult:
        if not self.enabled:
            return VerifierResult(True, [])
        v = []
        sn, on = self.numbers(src), self.numbers(out)
        # every (number, unit) pair must survive; no new numbers may appear. Unit may be absorbed by
        # an adjacent word split, so compare numbers alone for the "no new" direction.
        for (num, unit), c in sn.items():
            if sum(oc for (n2, u2), oc in on.items() if n2 == num and (u2 == unit or not unit)) < c:
                v.append(f"number/dose lost or altered: {fmt(num)} {unit}".strip())
        src_nums = {n for n, _ in sn}
        for (num, unit) in on:
            if num not in src_nums:
                v.append(f"hallucinated number: {fmt(num)}")
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
        if use_nli and self.nli is not None and not v and (self.nli_scope == "all" or st or sn):
            e = self.nli(src, out).get("entailment", 1.0)            # dict with 'entailment' probability
            if self.nli_mode == "bidirectional":
                e = min(e, self.nli(out, src).get("entailment", 1.0))
            if e < self.nli_threshold:
                v.append(f"NLI entailment {e:.2f} < {self.nli_threshold} ({self.nli_mode})")
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


class NullVerifier:
    """Ablation 'SIPG-RL minus verifier': accepts every edit and every final output. For evaluation only."""
    def check(self, src, out, use_nli=False):
        return VerifierResult(True, [])


_NLI_CACHE = {}


def default_nli(model="microsoft/deberta-large-mnli", quiet=False):
    """NLI adapter that is ON by default (audit item 5).  Without it the final gate repeats the edit gate's lexical check and
    semantic drift (antonyms, certainty, causality) leaks: 9/10 antonym injections leaked with NLI off vs 1/10 with it on.
    Set SIPG_NLI=off to disable explicitly (unit tests, offline machines).  If the model cannot be loaded this WARNS loudly
    and returns None; it never silently pretends the gate is semantic.  The loaded pipeline is cached per process."""
    import os, warnings
    if os.environ.get("SIPG_NLI", "").lower() in ("off", "0", "none", "false"):
        return None
    if model in _NLI_CACHE:
        return _NLI_CACHE[model]
    try:
        nli = HFNLI(model)
    except Exception as e:      # transformers/torchvision mismatch, no network on first use, ...
        if not quiet:
            warnings.warn(f"NLI model unavailable ({type(e).__name__}: {str(e)[:120]}). The FINAL gate is lexical-only: semantic "
                          "drift will leak. Fix the install, or set SIPG_NLI=off to silence this.", stacklevel=2)
        nli = None
    _NLI_CACHE[model] = nli
    return nli
