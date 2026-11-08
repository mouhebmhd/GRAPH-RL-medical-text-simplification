"""P2.1: ONE figure, three curves: risk (error rate on the accepted set) vs coverage, plus AURC / AUROC / ECE.
Curves: (1) combined uncertainty signal (logistic fit of correctness on the available signals, fit on one half, scored on the other),
        (2) decoder log-prob alone, (3) fidelity-to-source alone (BERTScore if you pass --fidelity bertscore).
Input records (JSONL, one per edited span or output):  {"correct": 0|1, "fid": F1, "logprob": mean token logprob (optional),
"entropy": policy entropy (optional), "nli": min bidirectional entailment (optional)}.  `correct` should come from clinician labels
or, until you have them, from injected hazards.
  python scripts/risk_coverage.py --records records.jsonl --out risk_coverage.png
  python scripts/risk_coverage.py --demo --out risk_coverage_demo.png     # HAND-WRITTEN suite (75 pairs): machinery demo, NOT a result"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sipg_rl.selective import risk_coverage, aurc, ece, auroc, fit_logistic, reliability


def demo_records(fid):
    from sipg_rl.adversarial import CORRUPTIONS, DRIFT, FAITHFUL
    recs = [dict(correct=1, fid=fid.score(s, o)["f1"], kind="faithful") for s, o in FAITHFUL]
    recs += [dict(correct=0, fid=fid.score(s, o)["f1"], kind=c) for c, s, o in DRIFT + CORRUPTIONS]
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records"); ap.add_argument("--demo", action="store_true"); ap.add_argument("--out", default="risk_coverage.png")
    ap.add_argument("--fidelity", default="soft", choices=["soft", "bertscore"]); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from sipg_rl.fidelity import make_fidelity
    fid = make_fidelity(a.fidelity)
    recs = demo_records(fid) if a.demo else [json.loads(l) for l in open(a.records) if l.strip()]
    y = np.array([r["correct"] for r in recs], int)
    rng = np.random.default_rng(a.seed); idx = rng.permutation(len(recs)); half = len(idx) // 2
    fit_i, ev_i = idx[:half], idx[half:]
    feats = [k for k in ("fid", "logprob", "entropy", "nli") if all(r.get(k) is not None for r in recs)]
    X = np.array([[r[k] if k != "logprob" else np.exp(r[k]) for k in feats] for r in recs], float)
    sig = {}
    pred = fit_logistic(X[fit_i], y[fit_i]); sig["combined (" + "+".join(feats) + ")"] = pred(X[ev_i])
    if "logprob" in feats:
        sig["decoder log-prob alone"] = X[ev_i, feats.index("logprob")]
    sig[f"fidelity to source alone ({a.fidelity})"] = X[ev_i, feats.index("fid")]
    ye = y[ev_i]
    print(f"{len(recs)} records ({int((y==0).sum())} wrong / {int((y==1).sum())} right); fit on {len(fit_i)}, evaluated on {len(ev_i)} (error rate if accept all: {100*(1-ye.mean()):.1f}%)\n")
    print(f"{'signal':48} {'AURC':>6} {'AUROC':>6} {'ECE':>6}   risk@cov 0.5 / 0.8 / 1.0")
    curves = {}
    for name, s in sig.items():
        cv = risk_coverage(s, ye, np.linspace(0.1, 1.0, 19)); curves[name] = cv
        at = lambda c: min(cv, key=lambda p: abs(p[0] - c))[1]
        sc = np.clip(s, 0, 1)
        print(f"{name:48} {aurc(cv):6.3f} {auroc(s, ye):6.3f} {ece(sc, ye):6.3f}   {100*at(.5):5.1f}% / {100*at(.8):5.1f}% / {100*at(1.0):5.1f}%")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name, cv in curves.items():
            ax[0].plot([c for c, _ in cv], [100 * r for _, r in cv], marker="o", ms=3, label=name)
        ax[0].set_xlabel("coverage"); ax[0].set_ylabel("error rate on accepted set (%)"); ax[0].set_title("Risk-coverage"); ax[0].legend(fontsize=7)
        rel = reliability(np.clip(next(iter(sig.values())), 0, 1), ye, 8)
        ax[1].plot([0, 1], [0, 1], "k--", lw=1); ax[1].plot([b["conf"] for b in rel], [b["acc"] for b in rel], marker="o")
        ax[1].set_xlabel("confidence"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Reliability (combined signal)")
        plt.tight_layout(); plt.savefig(a.out, dpi=130); print("\nfigure:", a.out)
    except Exception as e:
        print("plot skipped:", e)
    if a.demo:
        print("\nDEMO ONLY: 75 hand-written pairs, 'fid' from the offline SoftTokenFidelity, no decoder log-prob (rule realizer). It validates the code path, not the signal.")


if __name__ == "__main__":
    main()
