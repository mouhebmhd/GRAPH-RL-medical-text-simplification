"""Run EVERY offline, human-free item in one go and write eval/summary.md + manifest.json.
    python scripts/run_all_offline.py [--out eval] [--n 200] [--jsonl your.jsonl] [--quick]
Skipped automatically (need an external model): NLI-on rows, natural-drift rate, real-LLM arms. Pass --nli/--llm flags to the individual scripts."""
import argparse, os, subprocess, sys, time
here = os.path.dirname(os.path.abspath(__file__)); py = sys.executable
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="eval"); ap.add_argument("--n", type=int, default=200); ap.add_argument("--jsonl", default=None)
ap.add_argument("--quick", action="store_true", help="skip the pytest run"); a = ap.parse_args()
os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
J = ["--jsonl", a.jsonl] if a.jsonl else []
steps = [
    ("table2 (verifier ablation)", ["ablate_verifier.py", "--n", str(a.n), "--out", a.out] + J),
    ("2x2", ["run_2x2.py", "--n", str(a.n), "--out", a.out] + J),
    ("nli ablation (offline row)", ["ablate_nli.py", "--n", str(a.n), "--out", a.out] + J),
    ("injection harness", ["run_injection.py", "--base", "synth", "--n", "300"]),
    ("graph truncation", ["measure_truncation.py", "--n", "500", "--out", a.out] + J),
    ("cache check", ["check_cache.py"]),
    ("reward sanity", ["reward_sanity.py"]),
]
if not a.quick:
    steps.append(("tests", ["-m", "pytest", os.path.join(here, "..", "tests"), "-q"]))
rows = []
for name, cmd in steps:
    t0 = time.time()
    full = [py, os.path.join(here, cmd[0])] + cmd[1:] if not cmd[0].startswith("-") else [py] + cmd
    p = subprocess.run(full, capture_output=True, text=True, cwd=os.path.join(here, ".."))
    open(os.path.join(a.out, "logs", name.split()[0] + ".txt"), "w").write(p.stdout + p.stderr)
    rows.append((name, "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}", time.time() - t0))
    print(f"{rows[-1][1]:12} {name} ({rows[-1][2]:.0f}s)", flush=True)
subprocess.run([py, os.path.join(here, "reproduce.py"), "--out", a.out], cwd=os.path.join(here, ".."))
with open(os.path.join(a.out, "summary.md"), "w") as f:
    f.write("| step | status | s |\n|---|---|---|\n" + "\n".join(f"| {n} | {s} | {t:.0f} |" for n, s, t in rows) + "\n")
print(f"\nsummary -> {a.out}/summary.md  | logs -> {a.out}/logs/ | tables -> {a.out}/table2.md, 2x2.json")
sys.exit(1 if any(s != "ok" for _, s, _ in rows) else 0)
