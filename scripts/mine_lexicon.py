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
