"""Task 25 - how often does the N_MAX node cap clip sentences?   python scripts/measure_truncation.py [--jsonl your.jsonl] [--n 2000]
Reports the share of sentences whose graph has more than N_MAX nodes (clipped), nodes dropped, and what a larger cap would fix."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import numpy as np
from sipg_rl.pipeline import default_parts
from sipg_rl.graph import N_MAX

ap = add_common(argparse.ArgumentParser()); a = ap.parse_args()
a.n = max(a.n, 1)
P = default_parts(); B = P["builder"]
tot = np.array([getattr(B.build_text(s), "n_total", 0) for s in sentences(a)])
print(f"{len(tot)} sentences | current cap N_MAX={N_MAX}")
print(f"clipped sentences: {int((tot > N_MAX).sum())} = {100*np.mean(tot > N_MAX):.2f}% | mean nodes {tot.mean():.1f} | p95 {np.percentile(tot, 95):.0f} | max {tot.max()}")
print("nodes dropped in clipped sentences: " + (f"mean {np.mean(tot[tot > N_MAX] - N_MAX):.1f}" if (tot > N_MAX).any() else "none"))
for cap in (24, 32, 40, 64):
    print(f"  cap {cap:3d}: would clip {100*np.mean(tot > cap):5.2f}% of sentences")
