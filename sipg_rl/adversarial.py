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
