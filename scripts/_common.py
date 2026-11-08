import json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def sentences(a):
    """Sentences to test on: --jsonl (Expert/source column, split into sentences) or synthetic templates."""
    from sipg_rl.data import synth
    if getattr(a, "jsonl", None):
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        k = lambda r, *n: next((r[x] for x in n if x in r and r[x]), None)
        rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
        out = [s.text.strip() for r in rows for s in nlp(k(r, "Expert", "source") or "").sents if len(s.text.split()) >= a.min_words]
        return out[:a.n]
    return synth(a.n, a.seed)


def add_common(ap):
    ap.add_argument("--jsonl", default=None, help="your own parallel/source text (optional); default = synthetic template sentences")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "sipg_policy_v3.pt")); ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--out", default="eval")
    return ap


def ckpt_or_none(p):
    return p if p and os.path.exists(p) else None
