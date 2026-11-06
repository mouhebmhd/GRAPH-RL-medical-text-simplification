"""Graph-truncation diagnostic (audit item 14): how often does the node cap clip a sentence, and what is lost?

  python scripts/graph_stats.py --split test                    # Med-EASi (needs `datasets`), reports caps 24/32/40/48/64
  python scripts/graph_stats.py --jsonl file.jsonl --src-field Expert

The graph builder lists clause nodes first and TERM nodes (LEX/NOM/AMOD/CPLX) after, then cuts at N_MAX, so clipping removes
term nodes (the lexical edits) from the END of the sentence.  Output: share of sentences clipped, mean/95th-percentile nodes lost,
the node kinds lost, and the share of lexicon-hit (LEX/NOM) nodes lost, for several caps.  Decide the cap from this table; if you
pre-split instead, the rule must be documented next to the results.
"""
import argparse, collections, os, sys, types, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cbasu/Med-EASi"); ap.add_argument("--jsonl", default=None)
    ap.add_argument("--split", default="test"); ap.add_argument("--src-field", default="Expert"); ap.add_argument("--ref-field", default="Simple")
    ap.add_argument("--min-words", type=int, default=5); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--caps", default="24,32,40,48,64"); ap.add_argument("--out", default="eval/graph_stats.json")
    a = ap.parse_args()
    os.environ["SIPG_NMAX"] = "4096"            # build UNCLIPPED graphs, then apply each cap analytically
    from eval_suite import load_rows
    import spacy
    from sipg_rl.graph import GraphBuilder
    from sipg_rl.prompts import PromptInstantiator
    ns = types.SimpleNamespace(dataset=a.dataset, jsonl=a.jsonl, split=a.split, src_field=a.src_field, ref_field=a.ref_field,
                               min_words=a.min_words, shuffle=False, seed=0, limit=a.limit)
    rows = load_rows(ns, a.split, need_ref=False)
    nlp = spacy.load("en_core_web_sm"); b = GraphBuilder(nlp, PromptInstantiator())
    graphs = []
    for ex in rows:
        for sent in nlp(ex["src"]).sents:
            if sent.text.strip():
                g = b.build(sent); graphs.append([n.kind for n in g.nodes])
    sizes = np.array([len(k) for k in graphs])
    print(f"{len(graphs)} sentences | nodes per sentence: mean {sizes.mean():.1f}, median {np.median(sizes):.0f}, p95 {np.percentile(sizes, 95):.0f}, max {sizes.max()}")
    res = {}
    print(f"{'cap':>4} {'clipped %':>10} {'nodes lost mean':>16} {'lost p95':>9} {'LEX/NOM lost %':>15}  kinds lost")
    for cap in map(int, a.caps.split(",")):
        lost = np.maximum(0, sizes - cap)
        kinds = collections.Counter(k for ks in graphs for k in ks[cap:])
        lex_total = sum(1 for ks in graphs for k in ks if k in ("LEX", "NOM"))
        lex_lost = sum(1 for ks in graphs for k in ks[cap:] if k in ("LEX", "NOM"))
        res[cap] = dict(clipped=float((lost > 0).mean()), lost_mean=float(lost.mean()), lex_lost=lex_lost / max(1, lex_total), kinds=dict(kinds))
        print(f"{cap:4d} {100*(lost > 0).mean():10.1f} {lost.mean():16.2f} {np.percentile(lost, 95):9.0f} {100*lex_lost/max(1, lex_total):15.1f}  {dict(kinds)}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(n_sentences=len(graphs), caps=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
