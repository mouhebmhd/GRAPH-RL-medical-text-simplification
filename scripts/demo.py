"""Simplify free text with a trained policy and print the audit trail (the traversal path)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sipg_rl.pipeline import SIPGRL

DEFAULT = ("The hypertensive patient, who takes lisinopril 10 mg BID, reports no chest pain. "
           "Discontinue warfarin 5 mg if dyspnea occurs, because the drug is contraindicated in patients with renal impairment.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default=DEFAULT)
    ap.add_argument("--ckpt", default="sipg_policy.pt")
    a = ap.parse_args()
    m = SIPGRL(checkpoint=a.ckpt if os.path.exists(a.ckpt) else None)
    res = m.simplify(a.text)
    print("OUTPUT:", res.text, "\n")
    for s in res.sentences:
        print(f"SOURCE : {s['source']}\nOUTPUT : {s['output']}")
        print(f"S: {s['S_before']*100:.0f}% -> {s['S_after']*100:.0f}% | verified={s['verified']} {s['violations']}")
        for t in s["trail"]:
            print(f"   step {t['step']:2d} INVOKE node {t['node']} [{t['kind']}] -> {t.get('event')}  (r={t['reward']:+.3f})")
        print()
