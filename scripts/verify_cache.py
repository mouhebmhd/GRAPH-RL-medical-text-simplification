"""LLM cache verification (audit item 15): `cache_hits = 0` over 388 calls - is the cache broken, or just never exercised?

  python scripts/verify_cache.py                          # offline self-test against a stand-in server (proves the mechanism)
  python scripts/verify_cache.py --live                   # same checks against your real Ollama model
  python scripts/verify_cache.py --file runs/llm_cache.jsonl   # analyse an existing cache file

The cache key is (prompt, span, FULL CURRENT SENTENCE).  Over one pass of distinct sentences that key is almost never repeated,
so 0 hits there is expected, not a bug; hits appear when the same sentence state is queried again (another seed, --resume, a second
system).  The file analysis therefore reports how many calls WOULD be shared under a looser key (prompt, span) so the cost of the
strict key is visible instead of silent.
"""
import argparse, collections, json, os, sys, tempfile, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sipg_rl.llm import OpenAICompatLLM


def selftest(live, url, model):
    ok = True
    d = tempfile.mkdtemp(); path = os.path.join(d, "cache.jsonl")
    srv = None
    if not live:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
        import fake_llm
        srv = fake_llm.serve(0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url, model = f"http://127.0.0.1:{srv.server_address[1]}/v1", "fake-model:1b"
    llm = OpenAICompatLLM(url, model, cache_path=path)
    args = ("Replace the span with plain wording.", "dyspnea", "The patient has dyspnea.")
    a = llm(*args); b = llm(*args)
    s = llm.stats
    print(f"1) same call twice in one process : calls={s['calls']} hits={s['cache_hits']}  (expect calls=1 hits=1)  same answer: {a == b}")
    ok &= s["calls"] == 1 and s["cache_hits"] == 1 and a == b
    llm2 = OpenAICompatLLM(url, model, cache_path=path)
    llm2(*args)
    print(f"2) new process, persistent file   : calls={llm2.stats['calls']} hits={llm2.stats['cache_hits']}  (expect calls=0 hits=1)")
    ok &= llm2.stats["calls"] == 0 and llm2.stats["cache_hits"] == 1
    llm2("Replace the span with plain wording.", "dyspnea", "She had dyspnea yesterday.")
    print(f"3) same span, different sentence  : calls={llm2.stats['calls']} hits={llm2.stats['cache_hits']}  (expect a NEW call: the key includes the sentence)")
    ok &= llm2.stats["calls"] == 1
    print("cache mechanism:", "WORKS" if ok else "BROKEN")
    return ok


def analyse(path):
    keys = collections.Counter(); loose = collections.Counter(); n = 0
    for line in open(path, encoding="utf8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = r["key"]
        if k and k[0] == "explain":
            continue
        n += 1; keys[tuple(k)] += 1; loose[(k[0], k[1])] += 1
    dup = sum(c - 1 for c in keys.values())
    print(f"{path}: {n} cached calls | distinct exact keys {len(keys)} | duplicate lines {dup} (should be 0: a duplicate means a hit was missed)")
    print(f"  under a looser key (prompt, span) only {len(loose)} distinct -> {n - len(loose)} calls ({100*(n-len(loose))/max(1,n):.0f}%) share a span+prompt with another call")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true"); ap.add_argument("--file", default=None)
    ap.add_argument("--llm-url", default="http://localhost:11434/v1"); ap.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    a = ap.parse_args()
    if a.file:
        analyse(a.file)
    else:
        sys.exit(0 if selftest(a.live, a.llm_url, a.llm_model) else 1)
