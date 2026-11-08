"""Task 1 - Table 2: strong realizer WITHOUT verifier/fallback vs WITH, scored by the same deterministic post-hoc checker.
    python scripts/ablate_verifier.py                      # offline: strong = unsafe stand-in realizer, synthetic sentences
    python scripts/ablate_verifier.py --strong llm --llm-model qwen2.5:7b-instruct   # real LLM via Ollama (hybrid realizer)
Writes eval/table2.json and eval/table2.md."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
from sipg_rl.twobytwo import summarize, run_cell, fmt_table

ap = add_common(argparse.ArgumentParser())
ap.add_argument("--strong", default="stand-in", choices=["stand-in", "llm"])
ap.add_argument("--llm-url", default="http://localhost:11434/v1"); ap.add_argument("--llm-model", default="qwen2.5:7b-instruct")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
S, llm = sentences(a), None
kind = "strong"
if a.strong == "llm":
    from sipg_rl.llm import OpenAICompatLLM
    llm = OpenAICompatLLM(a.llm_url, a.llm_model, cache_path=os.path.join(a.out, "llm_cache.jsonl"))
    h = llm.health()
    if not h["reachable"]:
        sys.exit(f"LLM not reachable ({h['error']}); omit --strong llm to use the offline stand-in")
    kind = "strong_llm"
cells = {f"{'strong realizer' if kind=='strong' else 'LLM (hybrid)'} + verifier {'ON' if on else 'OFF'}":
         summarize(run_cell(S, kind, on, ckpt=ckpt_or_none(a.ckpt), llm=llm, rate=a.rate, seed=a.seed)) for on in (True, False)}
tbl = fmt_table(cells)
print(f"TABLE 2 ({len(S)} sentences; every cell scored by checks.check_all; preservation = % of sentences containing the category that keep it)\n")
print(tbl)
if kind == "strong":
    print("\nNOTE: 'strong realizer' here is an OFFLINE STAND-IN that injects hazards (rate %.2f); it measures the gate, not a real LLM." % a.rate)
json.dump(cells, open(os.path.join(a.out, "table2.json"), "w"), indent=1, default=float)
open(os.path.join(a.out, "table2.md"), "w").write("```\n" + tbl + "\n```\n")
