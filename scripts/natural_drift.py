"""Task 17 - natural drift rate: bidirectional NLI on EVERY output; asymmetric entailment (src => out but out =/=> src) = potential drift.
Needs an NLI model (external download), so it is skipped without --nli.
    python scripts/natural_drift.py --nli microsoft/deberta-large-mnli [--jsonl your.jsonl] [--apply-all]
Writes eval/natural_drift_cases.csv (flagged cases, for later inspection) and prints the rate. The rate itself needs no human."""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import numpy as np
from sipg_rl.pipeline import SIPGRL, default_parts
from sipg_rl import metrics as M

ap = add_common(argparse.ArgumentParser()); ap.add_argument("--nli"); ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--apply-all", action="store_true")
a = ap.parse_args()
if not a.nli:
    sys.exit("SKIPPED: natural-drift detection needs an NLI model. Re-run with --nli microsoft/deberta-large-mnli")
from sipg_rl.verifier import HFNLI
nli = HFNLI(a.nli)
m = SIPGRL(default_parts(), checkpoint=ckpt_or_none(a.ckpt))
os.makedirs(a.out, exist_ok=True)
rows, flagged = [], []
for s in sentences(a):
    out = m.simplify(s, apply_all=a.apply_all).text
    r = M.nli_entailment(s, out, nli)
    rows.append(r); 
    if r["fwd"] >= a.thr and r["bwd"] < a.thr:
        flagged.append((s, out, r["fwd"], r["bwd"]))
print(f"{len(rows)} outputs | asymmetric entailment (potential drift): {len(flagged)} = {100*len(flagged)/len(rows):.1f}% | mean fwd {np.mean([r['fwd'] for r in rows]):.3f} bwd {np.mean([r['bwd'] for r in rows]):.3f}")
with open(os.path.join(a.out, "natural_drift_cases.csv"), "w", newline="", encoding="utf8") as f:
    w = csv.writer(f); w.writerow(["source", "output", "fwd", "bwd"]); w.writerows(flagged)
print(f"flagged cases -> {a.out}/natural_drift_cases.csv")
