"""Task 5 - {strong, weak} x {verifier on, off} -> eval/2x2.json   (python scripts/run_2x2.py [--strong llm ...])
Expected pattern (the orthogonality the paper relies on): both ON cells preserve critical spans; OFF cells differ: weak-OFF stays
safe-by-construction, strong-OFF leaks. tests/test_2x2_orthogonality.py asserts it."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
from sipg_rl.twobytwo import summarize, run_cell, fmt_table

ap = add_common(argparse.ArgumentParser())
ap.add_argument("--strong", default="stand-in", choices=["stand-in", "llm"])
ap.add_argument("--llm-url", default="http://localhost:11434/v1"); ap.add_argument("--llm-model", default="qwen2.5:7b-instruct")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
S, llm, kind = sentences(a), None, "strong"
if a.strong == "llm":
    from sipg_rl.llm import OpenAICompatLLM
    llm = OpenAICompatLLM(a.llm_url, a.llm_model, cache_path=os.path.join(a.out, "llm_cache.jsonl"))
    if not llm.health()["reachable"]:
        sys.exit("LLM not reachable; omit --strong llm for the offline stand-in")
    kind = "strong_llm"
cells = {}
for k in (kind, "weak"):
    for on in (True, False):
        cells[f"{'strong' if k != 'weak' else 'weak'}|verifier_{'on' if on else 'off'}"] = summarize(
            run_cell(S, k, on, ckpt=ckpt_or_none(a.ckpt), llm=llm, rate=a.rate, seed=a.seed))
print(f"2x2 on {len(S)} sentences\n"); print(fmt_table(cells))
json.dump(cells, open(os.path.join(a.out, "2x2.json"), "w"), indent=1, default=float)
print(f"\nwrote {a.out}/2x2.json")
