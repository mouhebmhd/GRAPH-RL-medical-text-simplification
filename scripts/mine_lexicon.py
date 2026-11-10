<<<<<<< HEAD
"""Mine a jargon->layman lexicon from a PARALLEL JSONL you supply (keys Expert/Simple, optional expert_terms/layman_terms), enforce the
blocklist, run the independent audit, write lexicon_mined.json. Then:  SIPG_MINED_LEXICON=lexicon_mined.json python ...   (Task 27)
    python scripts/mine_lexicon.py --jsonl train.jsonl --out lexicon_mined.json --min-count 2"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sipg_rl.mined import MinedLexicon, Blocklist, audit_lexicon
from sipg_rl import metrics as M


def rows(path):
    k = lambda r, *n: next((r[x] for x in n if x in r and r[x]), None)
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            if k(r, "Expert", "source") and k(r, "Simple", "target"):
                yield (k(r, "Expert", "source"), k(r, "Simple", "target"), M.parse_terms(r.get("expert_terms")) or None, M.parse_terms(r.get("layman_terms")) or None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jsonl", required=True); ap.add_argument("--out", default="lexicon_mined.json")
    ap.add_argument("--min-count", type=int, default=2); ap.add_argument("--min-share", type=float, default=0.5)
    a = ap.parse_args()
    lex, bl, n = MinedLexicon(), Blocklist(), 0
    for s, r, et, lt in rows(a.jsonl):
        lex.add_row(s, r, et, lt, bl); n += 1
    lex.save(a.out)
    rules = lex.export_rules(a.min_count, a.min_share)
    bad = audit_lexicon({j: lex.ranked(j) for j in lex.cands})
    print(f"{n} rows -> {len(lex.cands)} candidates, {len(rules)} usable | blocked: {dict(lex.rejected) or 'nothing'} | AUDIT violations: {len(bad)}")
    if bad: sys.exit(f"FAIL: {bad[:3]}")
    print("wrote", a.out)
=======
"""Mine a plain-wording lexicon from the TRAIN split (audit item 7).

  python scripts/mine_lexicon.py --split train --out data/lexicon_mined.json [--min-count 2] [--jsonl file.jsonl]
  # then:  set SIPG_MINED_LEXICON=data/lexicon_mined.json   (or leave it at data/lexicon_mined.json: it is picked up automatically)

Refuses to mine from any split other than 'train' (never the test split).  Every entry is passed through the D3 safety gate
(no drug names / numbers / doses / codes); rejected pairs are listed in the file's meta.
"""
import argparse, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from sipg_rl import mining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cbasu/Med-EASi"); ap.add_argument("--jsonl", default=None)
    ap.add_argument("--split", default="train"); ap.add_argument("--src-field", default="Expert"); ap.add_argument("--ref-field", default="Simple")
    ap.add_argument("--min-words", type=int, default=5); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=2); ap.add_argument("--min-share", type=float, default=0.4)
    ap.add_argument("--out", default="data/lexicon_mined.json")
    a = ap.parse_args()
    if a.split != "train":
        sys.exit(f"refusing to mine from split '{a.split}': the lexicon must be built from the TRAIN split only (no test leakage)")
    from eval_suite import load_rows
    ns = types.SimpleNamespace(dataset=a.dataset, jsonl=a.jsonl, split="train", src_field=a.src_field, ref_field=a.ref_field,
                               min_words=a.min_words, shuffle=False, seed=0, limit=a.limit)
    rows = load_rows(ns, "train")
    entries, rep = mining.mine(rows, "train", a.min_count, a.min_share)
    mining.save(entries, rep, a.out)
    print(f"mined {rep['n_entries']} entries from {rep['n_rows']} train rows "
          f"({rep['n_candidate_keys']} candidate expert phrases, {rep['n_unsafe_dropped']} dropped as unsafe, "
          f"{rep['n_below_threshold']} below count/share threshold) -> {a.out}")
    for k, p, why in rep["unsafe_examples"][:5]:
        print(f"  unsafe, dropped: {k!r} -> {p!r}: {why}")


if __name__ == "__main__":
    main()
>>>>>>> 1bfb6c5672b8225d0d769783c21f87e9190e7f5b
