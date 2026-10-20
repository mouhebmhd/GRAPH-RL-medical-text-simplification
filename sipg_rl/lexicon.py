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
