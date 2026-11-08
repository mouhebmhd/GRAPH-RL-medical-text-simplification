"""Test data selector. Default: the 24-pair hand-written TOY fixture. For the real acceptance run set
SIPG_MEDEASI_JSONL=/path/to/medeasi_test.jsonl (Expert/Simple keys) - tests that guard Med-EASi behaviour then use it."""
import json, os

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mini_medeasi.jsonl")
IS_REAL = bool(os.environ.get("SIPG_MEDEASI_JSONL"))


def load_pairs():
    path = os.environ.get("SIPG_MEDEASI_JSONL") or FIXTURE
    rows = [json.loads(l) for l in open(path) if l.strip()]
    k = lambda r, *n: next((r[x] for x in n if x in r and r[x]), None)
    return [(k(r, "Expert", "source"), k(r, "Simple", "target")) for r in rows if k(r, "Expert", "source") and k(r, "Simple", "target")]
