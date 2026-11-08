"""Task 6 - NLI on/off for ANTONYM drift (meaning flips with no number/negation/hedge/drug change), edit gate vs final gate.
  edit gate  = the per-edit verifier call (lexical only: NLI is never run per edit - it is the cheap gate)
  final gate = the last check on the assembled output (the only place NLI runs)
  python scripts/ablate_nli.py                                   # offline: reports the NLI-OFF row (the lexical blind spot)
  python scripts/ablate_nli.py --nli microsoft/deberta-large-mnli # also the NLI-ON rows (needs `transformers` + a model download)"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import numpy as np
from sipg_rl.inject import inject_antonym
from sipg_rl.verifier import FidelityVerifier

ap = add_common(argparse.ArgumentParser())
ap.add_argument("--nli", default=None); ap.add_argument("--thr", type=float, nargs="+", default=[0.3, 0.5, 0.7])
a = ap.parse_args()
S = sentences(a)
ex = [dict(src=s, out=s) for s in S]
pert, n = inject_antonym(ex)
pert = [p for p in pert if p["injected"]]
print(f"{len(pert)} antonym-drift outputs injected (e.g. {pert[0]['src'][:60]!r} -> {pert[0]['out'][:60]!r})\n")
off = FidelityVerifier()
catch = lambda v, use: np.mean([not v.check(p["src"], p["out"], use_nli=use).ok for p in pert])
print(f"{'gate':44}{'drift caught':>14}{'leaked':>9}")
print(f"{'edit gate (lexical)  / NLI off':44}{100*catch(off, False):13.1f}%{100*(1-catch(off, False)):8.1f}%")
print(f"{'final gate (lexical) / NLI off':44}{100*catch(off, True):13.1f}%{100*(1-catch(off, True)):8.1f}%")
if not a.nli:
    print("\nNLI-ON rows skipped (no --nli model). The NLI-off rows are the measured baseline: the lexical gate cannot see antonym drift.")
else:
    from sipg_rl.verifier import HFNLI
    nli = HFNLI(a.nli)
    for mode in ("forward", "bidirectional"):
        for t in a.thr:
            v = FidelityVerifier(nli=nli, nli_threshold=t, nli_mode=mode)
            print(f"{'final gate + NLI ' + mode + ' thr=' + str(t):44}{100*catch(v, True):13.1f}%{100*(1-catch(v, True)):8.1f}%")
