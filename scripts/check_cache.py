"""Task 23 - is the LLM cache wired? Runs a stand-in OpenAI-compatible server in-process (NO real model) and shows:
  (1) repeated lexical spans across DIFFERENT sentences hit the cache (lexical requests are cached per span, not per sentence),
  (2) the persistent cache file makes a second run cost zero server calls.   python scripts/check_cache.py"""
import os, sys, tempfile, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import fake_llm as F
from sipg_rl.llm import OpenAICompatLLM
from sipg_rl.pipeline import SIPGRL, default_parts
from sipg_rl.realize import HybridRealizer, RuleRealizer

srv = F.serve(0); threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
sents = ["The patient has cardiomyopathy.", "She was treated for cardiomyopathy last year.", "He denies cardiomyopathy.", "Cardiomyopathy is rare in children."]
path = os.path.join(tempfile.mkdtemp(), "cache.jsonl")
for run in (1, 2):
    F.CALLS.clear()
    llm = OpenAICompatLLM(url, "fake-model:1b", cache_path=path)
    m = SIPGRL(default_parts(realizer=HybridRealizer(llm, RuleRealizer())), apply_all=False) if False else SIPGRL(default_parts(realizer=HybridRealizer(llm, RuleRealizer())))
    for s in sents: m.simplify(s, apply_all=True)
    st = llm.stats
    print(f"run {run}: llm.calls={st['calls']} cache_hits={st['cache_hits']} server_requests={len(F.CALLS)} cache_file_lines={sum(1 for _ in open(path))}")
print("\nrun 1: the same span in 4 sentences -> 1 server request + hits.  run 2: loaded from disk -> 0 server requests.")
