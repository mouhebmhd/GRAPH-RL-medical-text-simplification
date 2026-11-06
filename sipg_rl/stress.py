"""D1: critical-span stress set. 120 deterministic items, 20 per category, each also containing simplifiable jargon so that
systems actually edit the sentence. Preservation is checked with the independent detectors in metrics.critical_report plus
exact-span checks for numbers / units / drug names. Baselines that rewrite freely should show failures; a verified system
should show none."""
from __future__ import annotations
from dataclasses import dataclass
from . import metrics as M

DRUGS = ["warfarin", "lisinopril", "metformin", "digoxin", "furosemide", "atorvastatin", "amiodarone", "prednisone", "ibuprofen", "insulin"]
JARGON = ["dyspnea", "hypertension", "edema", "pruritus", "syncope", "nausea", "tachycardia", "epistaxis", "myalgia", "dysphagia"]
NUMS = ["2.5", "10", "0.125", "500", "81", "40", "5", "25", "1.5", "20"]
UNITS = ["mg", "mcg", "ml", "units", "mg", "mmHg", "%", "mg", "g", "mcg"]


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    text: str
    must_keep: tuple = ()      # exact lowercase spans that must survive


def build_stress_set() -> list:
    items = []
    for i in range(20):
        d, j, n, u = DRUGS[i % 10], JARGON[(i * 3) % 10], NUMS[i % 10], UNITS[i % 10]
        items.append(Item(f"num{i}", "numbers", f"Administer {d} {n} {u} twice daily and monitor for {j}.", (n,)))
        items.append(Item(f"unit{i}", "units", f"The recommended dose of {d} is {n} {u} when {j} is present.", (u.lower(),)))
        neg = ["no", "not", "denies", "without", "never"][i % 5]
        negtxt = {"no": f"The patient has no {j} or {JARGON[(i + 1) % 10]}.",
                  "not": f"Do {neg} give {d} to a patient with {j}.".replace("Do not give", "Do not give"),
                  "denies": f"The patient denies {j} and {JARGON[(i + 2) % 10]}.",
                  "without": f"She was discharged without {j} after treatment.",
                  "never": f"He has never experienced {j} while taking {d}."}[neg]
        items.append(Item(f"neg{i}", "negations", negtxt, ()))
        hed = ["possible", "may", "suspected", "likely", "probable"][i % 5]
        hedtxt = {"possible": f"Possible {j} was noted on examination.", "may": f"The symptoms may represent {j} related to {d}.",
                  "suspected": f"Suspected {j} requires further review.", "likely": f"The episode was likely caused by {j}.",
                  "probable": f"Probable {j} was documented at admission."}[hed]
        items.append(Item(f"hed{i}", "hedges", hedtxt, ()))
        cnd = ["if", "unless", "if", "unless", "if"][i % 5]
        cndtxt = (f"Discontinue {d} if {j} occurs." if cnd == "if" else f"Continue {d} unless {j} develops.")
        items.append(Item(f"cond{i}", "conditionals", cndtxt, ()))
        items.append(Item(f"drug{i}", "drug_names", f"The patient with {j} was started on {d} after consultation.", (d,)))
    return items


def check_item(item: Item, output: str) -> bool:
    ok = M.critical_report(item.text, output).get(item.category)
    low = output.lower()
    return bool(ok) and all(s in low for s in item.must_keep)
