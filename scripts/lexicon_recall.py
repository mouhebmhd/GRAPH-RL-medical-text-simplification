<<<<<<< HEAD
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
=======
"""Mined-lexicon recall diagnostic (audit item 10): does the reference's own plain wording appear in the mined top-1/3/5?

  python scripts/lexicon_recall.py --lexicon data/lexicon_mined.json --split validation   # a HELD-OUT split, not the one mined from
Reports coverage and exact/soft recall@1/3/5 overall and per node type (LEX, NOM, AMOD, CPLX, OTHER).  If soft recall@5 < 30%
the alignment itself is the problem: rework mining before touching the realizers.
"""
import argparse, json, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from sipg_rl import mining


def node_kind_fn(nlp, builder):
    cache = {}
    from sipg_rl.lexicon import PLAIN_TERMS, ADJ_REL
    def kind_of(ex, phrase):
        k = (ex["id"], phrase)
        if k in cache:
            return cache[k]
        kind = "OTHER"
        try:
            doc = nlp(ex["src"])
            g = builder.build(next(doc.sents)) if doc else None
            low = phrase.lower()
            hit = [n for n in g.nodes if n.kind not in ("ROOT", "RELCL", "CONJ_CL", "ADVCL", "CCOMP", "ACL") and low in n.text.lower()]
            if hit:
                kind = hit[0].kind
            elif low in ADJ_REL:
                kind = "AMOD"
        except Exception:
            pass
        cache[k] = kind
        return kind
    return kind_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="data/lexicon_mined.json")
    ap.add_argument("--dataset", default="cbasu/Med-EASi"); ap.add_argument("--jsonl", default=None)
    ap.add_argument("--split", default="validation"); ap.add_argument("--src-field", default="Expert"); ap.add_argument("--ref-field", default="Simple")
    ap.add_argument("--min-words", type=int, default=5); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="eval/lexicon_recall.json")
    a = ap.parse_args()
    d = json.load(open(a.lexicon, encoding="utf8"))
    if d["meta"].get("split") != "train":
        sys.exit("lexicon was not mined from the train split: refusing to evaluate (leakage)")
    if a.split == "train":
        print("WARNING: evaluating recall on the TRAIN split the lexicon was mined from - optimistic, use a held-out split")
    from eval_suite import load_rows
    ns = types.SimpleNamespace(dataset=a.dataset, jsonl=a.jsonl, split=a.split, src_field=a.src_field, ref_field=a.ref_field,
                               min_words=a.min_words, shuffle=False, seed=0, limit=a.limit)
    rows = load_rows(ns, a.split)
    import spacy
    from sipg_rl.graph import GraphBuilder
    from sipg_rl.prompts import PromptInstantiator
    nlp = spacy.load("en_core_web_sm")
    rec = mining.recall_at_k(rows, d["entries"], kind_of=node_kind_fn(nlp, GraphBuilder(nlp, PromptInstantiator())))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"{'type':6} {'n':>5} {'covered':>8} {'ex@1':>6} {'ex@3':>6} {'ex@5':>6} {'soft@1':>7} {'soft@3':>7} {'soft@5':>7}")
    for k in ["ALL", "LEX", "NOM", "AMOD", "CPLX", "OTHER"]:
        if k in rec:
            r = rec[k]
            print(f"{k:6} {r['n']:5d} " + " ".join(f"{100*r[c]:7.1f}" for c in ("covered", "exact@1", "exact@3", "exact@5", "soft@1", "soft@3", "soft@5")))
    s5 = rec["ALL"]["soft@5"]
    print("\nVERDICT:", "soft recall@5 < 30%: REWORK THE ALIGNMENT BEFORE TOUCHING THE REALIZERS" if s5 < 0.30 else
          f"soft recall@5 = {100*s5:.1f}% >= 30%: alignment is usable; realizer work can proceed")


if __name__ == "__main__":
    main()
>>>>>>> 1bfb6c5672b8225d0d769783c21f87e9190e7f5b
