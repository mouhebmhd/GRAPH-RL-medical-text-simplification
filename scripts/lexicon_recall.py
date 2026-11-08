"""Task 16 - mined-lexicon recall: for every jargon span a REFERENCE replaced in held-out rows, is the reference's layman choice in the
miner's top-1/3/5?  Broken down by node type (LEX/NOM/AMOD/CPLX/other). Needs parallel text you supply.
    python scripts/lexicon_recall.py --train train.jsonl --test test.jsonl
    python scripts/lexicon_recall.py --train all.jsonl --cv 5            # k-fold on one file
Low top-5 recall = the graph has nothing useful to fire on: it caps the layman-hit% any lexical system can reach."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from mine_lexicon import rows as load_rows
from sipg_rl.mined import MinedLexicon, Blocklist, pairs_from_row
from sipg_rl.pipeline import default_parts
from sipg_rl.audit_stats import wilson

ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--test"); ap.add_argument("--cv", type=int, default=0)
a = ap.parse_args()
tr = list(load_rows(a.train))
if a.cv:
    idx = np.random.default_rng(0).permutation(len(tr)); folds = [([tr[i] for i in idx if i % a.cv != k], [tr[i] for i in idx if i % a.cv == k]) for k in range(a.cv)]
elif a.test:
    folds = [(tr, list(load_rows(a.test)))]
else:
    sys.exit("give --test or --cv")
P = default_parts(); B = P["builder"]
KINDS = ["LEX", "NOM", "AMOD", "CPLX", "other"]
tot = {k: dict(n=0, cov=0, t1=0, t3=0, t5=0) for k in KINDS}
for train, test in folds:
    lex, bl = MinedLexicon(), Blocklist()
    for s, r, et, lt in train: lex.add_row(s, r, et, lt, bl)
    for s, r, et, lt in test:
        kinds = {n.text.lower(): n.kind for g in [B.build_text(s)] for n in g.nodes if n.kind in ("LEX", "NOM", "AMOD", "CPLX")}
        for js, ls in pairs_from_row(s, r, et, lt):
            k = kinds.get(js, "other"); k = k if k in tot else "other"
            ranked = lex.top(js, 5); t = tot[k]
            t["n"] += 1; t["cov"] += bool(ranked); t["t1"] += ls in ranked[:1]; t["t3"] += ls in ranked[:3]; t["t5"] += ls in ranked[:5]
print(f"{'node type':10}{'n':>5}{'any cand':>10}{'top-1':>9}{'top-3':>9}{'top-5':>9}   (95% CI on top-5)")
agg = dict(n=0, cov=0, t1=0, t3=0, t5=0)
for k, t in list(tot.items()) + [("ALL", None)]:
    if t is None:
        t = agg
    else:
        for x in agg: agg[x] += t[x]
    n = t["n"]
    if n == 0: print(f"{k:10}{0:5d}   n/a"); continue
    lo, hi = wilson(t["t5"], n)
    print(f"{k:10}{n:5d}{100*t['cov']/n:9.1f}%{100*t['t1']/n:8.1f}%{100*t['t3']/n:8.1f}%{100*t['t5']/n:8.1f}%   [{100*lo:.1f}, {100*hi:.1f}]")
