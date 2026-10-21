"""Small seed resources.

IMPORTANT: these are *stand-ins* for (a) the instruction model that realises local prompts and
(b) the clinician-curated critical-term list. They are not a knowledge base the policy reads;
the policy only sees the syntax-induced graph. Extend/replace via data files for real use.
"""
import os

# surface (lowercase) -> (plain wording, category)   category in {term, nom, abbr}
PLAIN_TERMS = {
    "hypertension": ("high blood pressure", "term"),
    "hypotension": ("low blood pressure", "term"),
    "myocardial infarction": ("heart attack", "term"),
    "hypertensive urgency": ("very high blood pressure that needs quick treatment", "term"),
    "hypertensive crisis": ("dangerously high blood pressure", "term"),
    "hypertensive emergency": ("dangerously high blood pressure that is harming the body", "term"),
    "diabetic ketoacidosis": ("a dangerous build-up of acid in the blood caused by diabetes", "term"),
    "dyspnea": ("trouble breathing", "term"),
    "edema": ("swelling", "term"),
    "pyrexia": ("fever", "term"),
    "tachycardia": ("fast heartbeat", "term"),
    "bradycardia": ("slow heartbeat", "term"),
    "emesis": ("vomiting", "term"),
    "syncope": ("fainting", "term"),
    "pruritus": ("itching", "term"),
    "erythema": ("redness", "term"),
    "cephalalgia": ("headache", "term"),
    "hyperglycemia": ("high blood sugar", "term"),
    "hypoglycemia": ("low blood sugar", "term"),
    "hyperlipidemia": ("high cholesterol", "term"),
    "analgesic": ("pain reliever", "term"),
    "antipyretic": ("fever reducer", "term"),
    "anticoagulant": ("blood thinner", "term"),
    "diuretic": ("water pill", "term"),
    "antihypertensive": ("blood pressure medicine", "term"),
    "exacerbation": ("flare-up", "term"),
    "exacerbated": ("made worse", "term"),
    "etiology": ("cause", "term"),
    "prognosis": ("outlook", "term"),
    "renal": ("kidney", "term"),
    "hepatic": ("liver", "term"),
    "pulmonary": ("lung", "term"),
    "cardiac": ("heart", "term"),
    "gastric": ("stomach", "term"),
    "cutaneous": ("skin", "term"),
    "acute": ("sudden", "term"),
    "chronic": ("long-lasting", "term"),
    "malignant": ("cancerous", "term"),
    "contraindicated": ("not safe to use", "term"),
    "ambulated": ("walked", "term"),
    "ambulate": ("walk", "term"),
    "administered": ("given", "term"),
    "administer": ("give", "term"),
    "discontinued": ("stopped", "term"),
    "discontinue": ("stop", "term"),
    "initiate": ("start", "term"),
    "utilize": ("use", "term"),
    "demonstrated": ("showed", "term"),
    "exhibited": ("showed", "term"),
    "approximately": ("about", "term"),
    "prior to": ("before", "term"),
    "subsequent to": ("after", "term"),
    "intravenously": ("through a vein", "term"),
    "orally": ("by mouth", "term"),
    "administration": ("giving", "nom"),
    "discontinuation": ("stopping", "nom"),
    "initiation": ("starting", "nom"),
    "monitoring": ("checking", "nom"),
    "ambulation": ("walking", "nom"),
    "utilization": ("use", "nom"),
    # critical frequency / route abbreviations: substitution allowed ONLY to these approved forms
    "bid": ("twice a day", "abbr"),
    "tid": ("three times a day", "abbr"),
    "qid": ("four times a day", "abbr"),
    "prn": ("as needed", "abbr"),
    "qhs": ("at bedtime", "abbr"),
    "po": ("by mouth", "abbr"),
}

# C2: broader deterministic lexical coverage. Each pair is a widely used plain-language equivalent; a clinician should
# still review additions (see tests/test_lexicon_safety.py: nothing with a drug name, number, unit or code may enter).
PLAIN_TERMS.update({
    "cerebrovascular accident": ("stroke", "term"),
    "epistaxis": ("nosebleed", "term"),
    "alopecia": ("hair loss", "term"),
    "dysphagia": ("trouble swallowing", "term"),
    "tinnitus": ("ringing in the ears", "term"),
    "hematuria": ("blood in the urine", "term"),
    "dysuria": ("painful urination", "term"),
    "hemoptysis": ("coughing up blood", "term"),
    "myalgia": ("muscle pain", "term"),
    "arthralgia": ("joint pain", "term"),
    "xerostomia": ("dry mouth", "term"),
    "diaphoresis": ("heavy sweating", "term"),
    "metastasis": ("spread of cancer to other parts of the body", "term"),
    "hepatotoxicity": ("liver damage", "term"),
    "nephrotoxicity": ("kidney damage", "term"),
    "hyperkalemia": ("high potassium in the blood", "term"),
    "hypokalemia": ("low potassium in the blood", "term"),
    "hyponatremia": ("low sodium in the blood", "term"),
    "hypernatremia": ("high sodium in the blood", "term"),
    "anemia": ("low red blood cell count", "term"),
    "thrombocytopenia": ("low platelet count", "term"),
    "leukocytosis": ("high white blood cell count", "term"),
    "neutropenia": ("a low number of infection-fighting white blood cells", "term"),
    "dyslipidemia": ("unhealthy levels of fats such as cholesterol in the blood", "term"),
    "ischemia": ("reduced blood supply", "term"),
    "myocardial ischemia": ("reduced blood flow to the heart", "term"),
    "orthostatic hypotension": ("a drop in blood pressure when standing up", "term"),
    "peripheral edema": ("swelling in the arms or legs", "term"),
    "pedal edema": ("swelling of the feet", "term"),
    "bilateral": ("on both sides", "term"),
    "bilaterally": ("on both sides", "term"),
    "unilateral": ("on one side", "term"),
    "idiopathic": ("of unknown cause", "term"),
    "asymptomatic": ("without symptoms", "term"),
    "symptomatic": ("having symptoms", "term"),
    "prophylaxis": ("prevention", "term"),
    "prophylactic": ("preventive", "term"),
    "adverse effects": ("side effects", "term"),
    "adverse events": ("side effects", "term"),
    "adverse reactions": ("harmful reactions", "term"),
    "efficacious": ("effective", "term"),
    "efficacy": ("how well it works", "term"),
    "ameliorated": ("improved", "term"),
    "ameliorate": ("improve", "term"),
    "exacerbate": ("make worse", "term"),
    "commence": ("start", "term"),
    "terminate": ("end", "term"),
    "ascertain": ("find out", "term"),
    "concurrently": ("at the same time", "term"),
    "subsequently": ("later", "term"),
    "sufficient": ("enough", "term"),
    "insufficient": ("not enough", "term"),
    "adequate": ("enough", "term"),
    "elevated": ("high", "term"),
    "diminished": ("reduced", "term"),
    "dermatitis": ("skin inflammation", "term"),
    "gastritis": ("inflammation of the stomach lining", "term"),
    "hepatitis": ("inflammation of the liver", "term"),
    "nephritis": ("inflammation of the kidneys", "term"),
    "bronchitis": ("inflammation of the airways in the lungs", "term"),
    "nocturia": ("waking at night to urinate", "term"),
})

# adjective -> (condition phrase, concept key). Only applied to person nouns (safety).
ADJ_REL = {
    "hypertensive": ("high blood pressure", "hypertension"),
    "hypotensive": ("low blood pressure", "hypotension"),
    "diabetic": ("diabetes", "diabetes"),
    "asthmatic": ("asthma", "asthma"),
    "febrile": ("a fever", "pyrexia"),
    "hyperlipidemic": ("high cholesterol", "hyperlipidemia"),
    "epileptic": ("epilepsy", "epilepsy"),
    "anemic": ("anemia", "anemia"),
}
PERSON_NOUNS = {"patient", "patients", "man", "men", "woman", "women", "male", "female", "adult",
                "adults", "child", "children", "individual", "person", "boy", "girl", "participant"}

CRITICAL_TERMS = set("""
warfarin heparin lisinopril metformin insulin digoxin amiodarone atorvastatin simvastatin metoprolol
amlodipine losartan furosemide aspirin clopidogrel apixaban rivaroxaban ibuprofen acetaminophen
paracetamol naproxen prednisone amoxicillin azithromycin ciprofloxacin levothyroxine omeprazole
gabapentin tramadol oxycodone morphine fentanyl hydrocodone lorazepam diazepam alprazolam sertraline
fluoxetine citalopram escitalopram lithium carbamazepine phenytoin levetiracetam albuterol salbutamol
hydrochlorothiazide spironolactone carvedilol propranolol bisoprolol methotrexate tacrolimus
""".split())
# Extend to the ~200-term clinician list without code changes:
_extra = os.environ.get("SIPG_CRITICAL_TERMS")
if _extra and os.path.exists(_extra):
    CRITICAL_TERMS |= {l.strip().lower() for l in open(_extra) if l.strip()}

# critical abbreviation -> approved plain substitutes (any one satisfies the verifier)
APPROVED_SUBS = {k: {v[0]} for k, v in PLAIN_TERMS.items() if v[1] == "abbr"}

NEGATION_CUES = {"no", "not", "never", "without", "none", "denies", "denied", "deny", "negative",
                 "absent", "absence", "cannot", "unable", "neither", "nor", "n't"}
UNITS = r"(?:mg|mcg|g|kg|ml|l|units?|iu|tablets?|capsules?|pills?|drops?|puffs?|%|mmhg|hours?|days?|weeks?|months?)"

# ---- source-side context cues (NegEx/ConText style) -------------------------------------------------
HEDGE_CUES = {"possible", "possibly", "probable", "probably", "suspected", "suspect", "likely", "unlikely",
              "may", "might", "could", "appears", "apparent", "questionable", "presumed", "potential"}
COND_CUES = {"if", "unless"}
HISTORICAL_CUES = {"history", "previous", "previously", "prior", "past", "resolved"}
OTHER_EXPERIENCER = {"mother", "father", "sister", "brother", "family", "familial", "parent"}
SCOPE_BREAKERS = {"but", "however", "although", "though", "except", "because", "while"}


# ---- D3: lexicon safety gate -------------------------------------------------------------------------------------------
# Nothing that names a drug, carries a number/unit, or looks like a code may be a replaceable key or a replacement, no
# matter how often a miner finds it aligned. Run at import and by any lexicon miner before it writes an entry.
import re as _re
_UNIT_WORDS = {"mg", "mcg", "g", "kg", "ml", "l", "iu", "unit", "units", "tablet", "tablets", "capsule", "capsules", "mmhg"}
_CODE_LIKE = _re.compile(r"\d|\b(?:icd|cpt|snomed|loinc|rxnorm|atc)\b|\b[a-z]{1,4}-?\d{2,}\b|%", _re.I)


def lexicon_entry_problem(key: str, plain: str, category: str = "term"):
    """Return a reason string if (key -> plain) must NOT be a replaceable entry, else None."""
    for side, text in (("key", key), ("replacement", plain)):
        words = set(_re.findall(r"[a-z]+", text.lower()))
        if _CODE_LIKE.search(text):
            return f"{side} contains a number, percentage or code-like token: {text!r}"
        if words & _UNIT_WORDS:
            return f"{side} contains a dose unit: {text!r}"
        if category != "abbr" and words & CRITICAL_TERMS:
            return f"{side} contains a critical drug term ({sorted(words & CRITICAL_TERMS)}): {text!r}"
    if category != "abbr" and key.lower() in CRITICAL_TERMS:
        return f"key is a critical term: {key!r}"
    return None


def assert_lexicon_safe(terms=None):
    bad = {k: r for k, (p, c) in (terms or PLAIN_TERMS).items() if (r := lexicon_entry_problem(k, p, c))}
    if bad:
        raise ValueError(f"unsafe lexicon entries: {bad}")


assert_lexicon_safe()
