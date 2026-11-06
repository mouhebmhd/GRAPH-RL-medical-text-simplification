"""Adversarial self-test for the safety gates (audit items A4 / E1).

Why this exists: the per-edit gate runs the *lexical* verifier with use_nli=False; the final gate re-runs the same
lexical check (+ NLI only if an NLI model is configured). With no NLI the final check is identical to the last
accepted edit's check, so the hard fallback (output := source) can never fire on any data. This module injects
known corruptions through a fake "realizer" and counts where each one is stopped:

    edit_gate   rejected and rolled back by the per-edit lexical check
    final_gate  passed every edit but caught by the final check (needs NLI) -> output reverts to the source
    leaked      reached the output (independent per-mode detector says the corruption is present)

recall = (edit_gate + final_gate) / injected. Run with NLI off and on to measure what the NLI adapter adds (E1).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from .realize import Realizer
from .lexicon import CRITICAL_TERMS

MODES = ("drop_number", "drop_negation", "swap_drug", "hallucinate_number", "antonym_drift", "drop_hedge")
_NUM = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|%|mmhg)?", re.I)
_NEG = re.compile(r"\b(?:no|not|never|without|denies|denied|cannot)\b\s*", re.I)
_HEDGE = re.compile(r"\b(?:possible|possibly|probable|probably|may|might|suspected|likely)\b\s*", re.I)
ANTONYMS = {"increased": "decreased", "decreased": "increased", "improved": "worsened", "worsened": "improved",
            "excluded": "confirmed", "confirmed": "excluded", "absent": "present", "present": "absent",
            "normal": "abnormal", "abnormal": "normal", "elevated": "low", "low": "elevated", "positive": "negative",
            "negative": "positive", "benign": "malignant", "malignant": "benign", "resolved": "worsened"}
_ANT_RX = re.compile(r"\b(" + "|".join(ANTONYMS) + r")\b", re.I)
# Item 6: the old map held 8 drugs, so on Med-EASi (few named drugs per sentence) the injector found nothing: injected = 0.
# Now EVERY critical drug name is swappable, each to a drug from a different class (a wrong-drug error, not a typo).
_DRUG_CLASSES = [
    "warfarin heparin apixaban rivaroxaban clopidogrel aspirin".split(),
    "lisinopril losartan amlodipine metoprolol carvedilol propranolol bisoprolol".split(),
    "metformin insulin levothyroxine prednisone".split(),
    "atorvastatin simvastatin omeprazole furosemide hydrochlorothiazide spironolactone digoxin amiodarone".split(),
    "ibuprofen naproxen acetaminophen paracetamol tramadol oxycodone morphine fentanyl hydrocodone".split(),
    "lorazepam diazepam alprazolam sertraline fluoxetine citalopram escitalopram lithium".split(),
    "gabapentin carbamazepine phenytoin levetiracetam".split(),
    "amoxicillin azithromycin ciprofloxacin methotrexate tacrolimus albuterol salbutamol".split(),
]
_SWAP = {}
for _i, _cls in enumerate(_DRUG_CLASSES):
    _other = _DRUG_CLASSES[(_i + 1) % len(_DRUG_CLASSES)]
    for _j, _d in enumerate(_cls):
        _SWAP[_d] = _other[_j % len(_other)]
_SWAP_RX = re.compile(r"\b(" + "|".join(sorted(_SWAP, key=len, reverse=True)) + r")\b", re.I)

# Sentences used to top up a mode that matches fewer than `min_injected` real sources (tagged synthetic in the report).
SYNTHETIC_SOURCES = [
    "The patient was started on {d} {n} mg daily for the treatment of hypertension.",
    "Do not give {d} to a patient with renal impairment.",
    "She denies chest pain and takes {d} {n} mg twice daily.",
    "{D} may cause dizziness, and the dose of {d} was increased to {n} mg.",
    "The hypertensive patient reports no dyspnea after {d} {n} mg was administered.",
    "Discontinue {d} if nausea or vomiting occurs.",
    "Prior to the procedure, {d} {n} mg was held and the patient was monitored.",
    "He has been taking {d} {n} mg and tolerates it well.",
    "Possible interaction between {d} and the new antibiotic was noted at admission.",
    "Monitoring of renal function is advised in patients receiving {d} {n} mg.",
    "Edema improved after {d} {n} mg was initiated, and blood pressure remained elevated.",
    "The recommended dose of {d} is {n} mg once daily unless dyspnea is present.",
]


def synthetic_sources(k=12):
    ds = sorted(_SWAP)
    out = []
    for i in range(k):
        d = ds[(i * 5) % len(ds)]
        out.append(SYNTHETIC_SOURCES[i % len(SYNTHETIC_SOURCES)].format(d=d, D=d.capitalize(), n=[5, 10, 20, 40, 81, 100][i % 6]))
    return out


def corrupt(mode: str, text: str) -> str:
    """Deterministic corruption of `text`; returns `text` unchanged when the mode does not apply."""
    if mode == "drop_number":
        return _NUM.sub("", text, count=1)
    if mode == "drop_negation":
        return _NEG.sub("", text, count=1)
    if mode == "drop_hedge":
        return _HEDGE.sub("", text, count=1)
    if mode == "swap_drug":
        return _SWAP_RX.sub(lambda m: _SWAP[m.group(0).lower()], text, count=1)
    if mode == "hallucinate_number":
        m = _SWAP_RX.search(text) or re.search(r"\b(?:patient|dose|tablet)s?\b", text, re.I)
        return text if not m else text[:m.end()] + " 500 mg" + text[m.end():]
    if mode == "antonym_drift":
        return _ANT_RX.sub(lambda m: ANTONYMS[m.group(0).lower()], text, count=1)
    raise ValueError(mode)


def applies(mode: str, text: str) -> bool:
    return corrupt(mode, text) != text


def leaked(mode: str, src: str, out: str) -> bool:
    """Independent per-mode detector: is the injected corruption visible in the final output?"""
    low = lambda t: re.findall(r"[a-z]+", t.lower())
    if mode == "drop_number":
        from collections import Counter
        f = lambda t: Counter(re.findall(r"\d+(?:\.\d+)?", t))
        return bool(f(src) - f(out))
    if mode == "hallucinate_number":
        return "500" in out and "500" not in src
    if mode == "drop_negation":
        cnt = lambda t: sum(1 for w in low(t) if w in {"no", "not", "never", "without", "denies", "denied", "cannot"})
        return cnt(out) < cnt(src)
    if mode == "drop_hedge":
        cnt = lambda t: sum(1 for w in low(t) if w in {"possible", "possibly", "probable", "probably", "may", "might", "suspected", "likely"})
        return cnt(out) < cnt(src)
    if mode == "swap_drug":
        return any(d in low(src) and d not in low(out) for d in _SWAP)
    if mode == "antonym_drift":
        return any(w in low(out) and w not in low(src) for w in set(ANTONYMS.values()))
    raise ValueError(mode)


class AdversarialRealizer(Realizer):
    """Corrupts the node's own span in `mode` (a stand-in for a misbehaving LLM). Use with apply_all=True so ROOT fires."""
    def __init__(self, mode):
        self.mode = mode

    def apply(self, graph, node, st):
        if node.kind != "ROOT" or any(i in st.edited for i in node.ids):
            return False
        ids = [i for i in node.ids if i not in st.deleted]
        before = st.render_ids(ids)
        after = corrupt(self.mode, before)
        if after == before:
            return False
        st.replace_span(ids, after)
        return True


@dataclass
class SelfTestRow:
    mode: str
    injected: int = 0
    n_synthetic: int = 0          # how many of `injected` came from the synthetic top-up rather than the real data
    edit_gate: int = 0
    final_gate: int = 0
    leaked: int = 0
    examples: list = field(default_factory=list)
    leaks: list = field(default_factory=list)       # every leaked case: source, output, NLI probabilities

    @property
    def recall(self):
        return (self.edit_gate + self.final_gate) / self.injected if self.injected else float("nan")


def selftest(make_system, sources, modes=MODES, max_per_mode=50, min_injected=9, nli=None):
    """make_system(realizer) -> object with .simplify(text, apply_all=True) (an SIPGRL built on that realizer, NLI as configured).
    A mode that applies to fewer than `min_injected` real sources is topped up from synthetic_sources() (tagged in
    row.n_synthetic) so that a mode can never silently report injected = 0.  Leaks are kept in row.leaks with the NLI
    probabilities (when `nli` is given) so a residual leak can be inspected, not just counted."""
    rows = []
    for mode in modes:
        row, sysm = SelfTestRow(mode), make_system(AdversarialRealizer(mode))
        real = [s for s in sources if applies(mode, s)][:max_per_mode]
        topup = [s for s in synthetic_sources(24) if applies(mode, s)][:max(0, min_injected - len(real))]
        for src in real + topup:
            res = sysm.simplify(src, apply_all=True)
            row.injected += 1
            row.n_synthetic += int(src in topup)
            rec = res.sentences
            row.edit_gate += int(any(r["rejected"] for r in rec))
            row.final_gate += int(any(r["final_rejected"] for r in rec))
            lk = leaked(mode, src, res.text)
            row.leaked += int(lk)
            if lk:
                probs = None
                if nli is not None:
                    try:
                        probs = {k: round(float(v), 3) for k, v in nli(src, res.text).items()}
                    except Exception as e:
                        probs = {"error": str(e)[:80]}
                row.leaks.append(dict(source=src, output=res.text, nli=probs, synthetic=src in topup))
                if len(row.examples) < 3:
                    row.examples.append((src, res.text))
        rows.append(row)
    return rows
