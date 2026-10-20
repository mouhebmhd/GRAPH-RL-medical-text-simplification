"""Data helpers: a synthetic clinical-sentence generator for pipeline pilots, and loaders for the
public corpora named in the proposal (Med-EASi / ClinicalEase / Cochrane-auto) once downloaded."""
from __future__ import annotations
import json, random

DRUGS = ["lisinopril", "metformin", "warfarin", "atorvastatin", "furosemide", "amlodipine", "metoprolol",
         "prednisone", "digoxin", "gabapentin", "levothyroxine", "omeprazole", "clopidogrel"]
DOSES = ["10 mg", "500 mg", "5 mg", "2.5 mg", "40 mg", "81 mg", "0.25 mg", "20 mg", "100 mcg"]
FREQ = ["daily", "BID", "TID", "PRN", "twice daily", "every 8 hours", "at bedtime"]
SYMPTOMS = ["chest pain", "nausea", "dizziness", "fever", "headache", "shortness of breath", "swelling"]
CONDS = ["asthma", "diabetes", "heart failure", "kidney disease", "high cholesterol", "atrial fibrillation"]
PROCS = ["surgery", "the procedure", "the scan", "discharge"]

TEMPLATES = [
    "The hypertensive patient, who takes {d1} {m1} {f}, reports no {s1}.",
    "Discontinue {d1} if {s1} or {s2} occurs, because the drug is contraindicated in patients with renal impairment.",
    "The patient was administered {d1} {m1} {f} and ambulated without dyspnea.",
    "Administration of {d1} {m1} should be discontinued prior to {p}, and the patient should not utilize {d2} until further notice.",
    "She has a history of {c}, which was exacerbated by {d1} {m1}, and she denies any {s1}.",
    "The diabetic patient presented with {s1} and edema, although laboratory results were negative for infection.",
    "Monitoring of blood pressure is necessary because {d1} may cause hypotension in elderly patients.",
    "Take {d1} {m1} {f} for the treatment of {c}, and do not stop it without consulting your physician.",
    "The patient, who has chronic {c}, should initiate {d1} {m1} daily and avoid {d2}.",
    "Myocardial infarction was ruled out; however, the patient reported cephalalgia and pruritus that persisted for 3 days.",
    "An anticoagulant such as {d1} {m1} is administered orally and should be monitored closely.",
    "No evidence of pulmonary edema was observed on examination of the asthmatic patients, who demonstrated improved lung function.",
    "The patient exhibited tachycardia and syncope after {d1} {m1} was administered intravenously.",
    "Etiology of the {s1} remains unclear, and the prognosis is not favorable without {p}.",
    "A febrile child, who received {d1} {m1} {f}, showed no erythema and had no emesis.",
    "Hepatic function was normal, but the patient should not exceed {m1} of {d1} per day.",
    "The hyperlipidemic man, who stopped {d1} {m1} last week, has acute {s1} and denies {s2}.",
    "Prior to initiation of {d1}, the patient had no cutaneous reaction and was not taking {d2}.",
]


def synth(n, seed=0):
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        d1, d2 = rng.sample(DRUGS, 2)
        s1, s2 = rng.sample(SYMPTOMS, 2)
        s = rng.choice(TEMPLATES).format(d1=d1, d2=d2, m1=rng.choice(DOSES), f=rng.choice(FREQ), s1=s1, s2=s2,
                                         c=rng.choice(CONDS), p=rng.choice(PROCS))
        if s not in seen or len(seen) > 2000:
            seen.add(s)
            out.append(s)
    return out


def load_parallel_jsonl(path, src_key="source", tgt_key="target"):
    """Med-EASi / ClinicalEase / Cochrane-auto style JSONL -> list of (complex, simple) pairs."""
    pairs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pairs.append((r[src_key], r.get(tgt_key)))
    return pairs
