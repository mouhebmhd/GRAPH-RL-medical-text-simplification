"""Task 20 - multi-seed runner: (optionally) train one policy per seed, then evaluate every system per seed and report mean +- std and a
t-based cross-seed CI for the learned components (run_eval prints the across-seed table when --seeds has >1 value).
    python scripts/run_multiseed.py --seeds 0,1,2,3,4 --train --jsonl medeasi_test.jsonl --limit 200
Anything else is forwarded to run_eval.py."""
import argparse, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--train", action="store_true")
ap.add_argument("--iters", type=int, default=120); ap.add_argument("--train-jsonl", default=None); ap.add_argument("--out", default="eval_out")
a, rest = ap.parse_known_args()
pat = "sipg_policy_v3_seed{seed}.pt"
if a.train:
    cmd = [sys.executable, os.path.join(here, "train_seeds.py"), "--seeds", a.seeds, "--iters", str(a.iters), "--out-pattern", pat]
    if a.train_jsonl: cmd += ["--jsonl", a.train_jsonl]
    print("$", " ".join(cmd)); subprocess.check_call(cmd)
cmd = [sys.executable, os.path.join(here, "run_eval.py"), "--seeds", a.seeds, "--ckpt-pattern", pat, "--out", a.out] + rest
print("$", " ".join(cmd)); sys.exit(subprocess.call(cmd))
