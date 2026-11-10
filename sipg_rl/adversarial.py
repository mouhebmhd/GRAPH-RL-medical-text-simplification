<<<<<<< HEAD
"""Small HAND-WRITTEN suites for safety-gate measurements. They test machinery and give *indicative* recall only:
  DRIFT    : meaning changes with NO number / negation / hedge / drug change (the lexical gate is blind to most of these by construction)
  FAITHFUL : paraphrases that should NOT be flagged
  CORRUPTIONS: unsafe rewrites the lexical gate is designed to catch"""
CORRUPTIONS = [
    ("dose_changed", "Take warfarin 5 mg by mouth every day.", "Take warfarin 50 mg by mouth every day."),
    ("dose_changed", "Give ibuprofen 400 mg every 6 hours.", "Give ibuprofen 400 mg every 4 hours."),
    ("unit_changed", "The dose is 5 mg twice a day.", "The dose is 5 g twice a day."),
    ("number_dropped", "Take 2 tablets of metformin 500 mg daily.", "Take tablets of metformin 500 mg daily."),
    ("number_added", "Take metformin 500 mg daily.", "Take metformin 500 mg daily for 3 weeks."),
    ("negation_dropped", "There was no sign of infection.", "There was a sign of infection."),
    ("negation_dropped", "Do not drive after taking this medicine.", "Drive after taking this medicine."),
    ("negation_dropped", "She denies chest pain and does not smoke.", "She has chest pain and smokes."),
    ("hedge_dropped", "This drug may cause dizziness.", "This drug causes dizziness."),
    ("hedge_dropped", "Pneumonia is possible in this patient.", "Pneumonia is present in this patient."),
    ("conditional_dropped", "Call your doctor if the pain gets worse.", "Call your doctor the pain gets worse."),
    ("conditional_dropped", "Stop lisinopril unless your doctor says otherwise.", "Stop lisinopril, your doctor says otherwise."),
    ("drug_swapped", "Take lisinopril 10 mg daily.", "Take metformin 10 mg daily."),
    ("drug_dropped", "Take warfarin 5 mg and aspirin 81 mg daily.", "Take warfarin 5 mg and a pain medicine 81 mg daily."),
    ("drug_hallucinated", "Take warfarin 5 mg daily.", "Take warfarin 5 mg and aspirin daily."),
]
DRIFT = [
    ("severity", "She has mild chest pain.", "She has severe chest pain."), ("severity", "The swelling is slight.", "The swelling is extreme."),
    ("direction", "Blood pressure increased after the dose.", "Blood pressure decreased after the dose."),
    ("direction", "Reduce the dose gradually.", "Raise the dose gradually."),
    ("laterality", "There is swelling in the left knee.", "There is swelling in the right knee."),
    ("laterality", "The scan showed a lump in the right lung.", "The scan showed a lump in the left lung."),
    ("temporal", "Take the tablet before meals.", "Take the tablet after meals."),
    ("temporal", "The rash appeared after the injection.", "The rash appeared before the injection."),
    ("quantifier", "Some patients develop a rash.", "All patients develop a rash."), ("quantifier", "Few people need surgery.", "Most people need surgery."),
    ("frequency", "Headaches occur rarely with this drug.", "Headaches occur often with this drug."),
    ("frequency", "The cough happens sometimes at night.", "The cough happens constantly at night."),
    ("entity_swap", "The infection affected the kidney.", "The infection affected the liver."),
    ("entity_swap", "He has pain in the stomach.", "He has pain in the chest."),
    ("population", "The medicine is safe in children.", "The medicine is safe in adults."),
    ("population", "Older people are at higher risk.", "Younger people are at higher risk."),
    ("causal", "The rash was caused by the antibiotic.", "The antibiotic was caused by the rash."),
    ("causal", "Smoking leads to lung disease.", "Lung disease leads to smoking."),
    ("route", "Take the medicine by mouth.", "Take the medicine through a vein."),
    ("route", "Apply the cream to the skin.", "Apply the cream to the eyes."),
    ("recommendation", "Avoid grapefruit juice with this drug.", "Drink grapefruit juice with this drug."),
    ("recommendation", "Stop taking the drug.", "Keep taking the drug."), ("recommendation", "Rest and avoid exercise.", "Rest and increase exercise."),
    ("condition_swap", "The test showed anemia.", "The test showed diabetes."), ("condition_swap", "He was treated for asthma.", "He was treated for epilepsy."),
    ("polarity_adj", "The result was normal.", "The result was abnormal."), ("polarity_adj", "The tumor was benign.", "The tumor was malignant."),
    ("polarity_adj", "Her blood sugar is high.", "Her blood sugar is low."),
    ("certainty", "The test confirmed pneumonia.", "The test suggested pneumonia."),
    ("certainty", "Infection is the most likely cause.", "Infection is the only cause."),
]
FAITHFUL = [
    ("Take 5 mg of warfarin by mouth once daily.", "Take warfarin 5 mg by mouth every day."),
    ("The patient presented with dyspnea and edema.", "The patient had trouble breathing and swelling."),
    ("Hypertension is a risk factor for stroke.", "High blood pressure is a risk factor for stroke."),
    ("Discontinue the medication immediately if rash occurs.", "Stop the medicine right away if a rash occurs."),
    ("Prior to surgery, fasting is required.", "Before surgery, you must not eat."),
    ("Adverse effects may include nausea and emesis.", "Side effects may include nausea and vomiting."),
    ("There was no evidence of pulmonary embolism.", "There was no sign of a blood clot in the lungs."),
    ("Ibuprofen 400 mg is administered every 6 hours.", "Take ibuprofen 400 mg every 6 hours."),
    ("Renal function should be monitored.", "Kidney function should be checked."),
    ("The wound exhibited erythema and edema.", "The wound was red and swollen."),
    ("Ambulate within 24 hours of surgery.", "Walk within 24 hours of surgery."),
    ("She denies any chest pain.", "She says she has no chest pain."),
    ("Hepatic impairment may increase drug levels.", "Liver problems may raise drug levels."),
    ("Administer 10 mg of lisinopril orally.", "Give lisinopril 10 mg by mouth."),
    ("Patients with diabetes should monitor glucose.", "People with diabetes should check their blood sugar."),
    ("The tumor did not metastasize.", "The tumor did not spread."), ("Anticoagulants may cause bleeding.", "Blood thinners may cause bleeding."),
    ("Seek care if dyspnea worsens.", "Get help if trouble breathing gets worse."),
    ("The child had pyrexia for three days.", "The child had a fever for three days."), ("Utilize the inhaler as needed.", "Use the inhaler as needed."),
    ("Subsequent to the procedure, rest is advised.", "After the procedure, rest is advised."),
    ("Cardiac output was reduced.", "The heart pumped less blood."), ("Initiate treatment promptly.", "Start treatment quickly."),
    ("The rash resolved without treatment.", "The rash went away without treatment."), ("Analgesic use should be limited.", "Pain reliever use should be limited."),
    ("Edema resolved after diuretic administration.", "The swelling went away after a water pill was given."),
    ("Gastric irritation is possible.", "Stomach irritation is possible."), ("Approximately 20 mg is given daily.", "About 20 mg is given daily."),
    ("Emesis occurred twice.", "Vomiting happened twice."), ("Pruritus is common with this medicine.", "Itching is common with this medicine."),
]
=======
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
>>>>>>> 1bfb6c5672b8225d0d769783c21f87e9190e7f5b


import random as _random
from .realize import Realizer, RuleRealizer
from .lexicon import CRITICAL_TERMS, NEGATION_CUES, HEDGE_CUES, COND_CUES


class AdversarialRealizer(Realizer):
    """Wraps a realizer and, with probability `rate` per invocation, ALSO applies a sentence-level unsafe corruption
    (dose x10, dropped negation / hedge / 'if', drug swap). `last_injected` names the category (None = clean), so a harness
    can count injected / blocked / leaked. Used to prove rollback + fallback execute and for the verifier ablation."""
    KINDS = ("dose_changed", "negation_dropped", "hedge_dropped", "conditional_dropped", "drug_swapped")

    def __init__(self, base=None, rate=0.5, seed=0, kinds=None):
        self.base, self.rate, self.rng = base or RuleRealizer(), rate, _random.Random(seed)
        self.kinds, self.last_injected, self.last_via, self.last_info = tuple(kinds or self.KINDS), None, None, None

    def _toks(self, g, st, pred):
        return [i for i in st.ids if i not in st.deleted and pred(g.doc[i])]

    def apply(self, graph, node, st):
        changed = self.base.apply(graph, node, st)
        self.last_injected, self.last_via, self.last_info = None, getattr(self.base, "last_via", None), getattr(self.base, "last_info", None)
        if self.rng.random() >= self.rate:
            return changed
        doc, order = graph.doc, list(self.kinds)
        self.rng.shuffle(order)
        for kind in order:
            if kind == "dose_changed":
                c = self._toks(graph, st, lambda t: t.like_num and t.text.replace(".", "").isdigit())
                if c:
                    v = float(doc[c[0]].text); st.repl[c[0]] = str(int(v * 10)) if v.is_integer() else f"{v * 10:g}"; st.edited.add(c[0]); break
            elif kind in ("negation_dropped", "hedge_dropped", "conditional_dropped"):
                cues = {"negation_dropped": NEGATION_CUES - {"n't"}, "hedge_dropped": HEDGE_CUES, "conditional_dropped": COND_CUES}[kind]
                c = self._toks(graph, st, lambda t: t.lower_ in cues)
                if c:
                    st.delete(c[0]); break
            elif kind == "drug_swapped":
                c = self._toks(graph, st, lambda t: t.lower_ in CRITICAL_TERMS)
                if c:
                    other = next(d for d in sorted(CRITICAL_TERMS) if d != doc[c[0]].lower_)
                    st.repl[c[0]] = other; st.edited.add(c[0]); break
        else:
            return changed
        self.last_injected = kind
        return True
<<<<<<< HEAD
=======


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
>>>>>>> 1bfb6c5672b8225d0d769783c21f87e9190e7f5b
