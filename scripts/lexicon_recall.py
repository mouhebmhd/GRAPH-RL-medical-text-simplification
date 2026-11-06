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
