"""P0.3 / A4: inject hazards and count caught vs leaked at the edit gate. A zero-injection run RAISES (never silent).

    python scripts/run_injection.py --base synth --n 300                       # machinery check (synthetic sentences)
    python scripts/run_injection.py --jsonl medeasi_test.jsonl --base source   # real sentences; gate(src, perturbed source)
    python scripts/run_injection.py --jsonl medeasi_test.jsonl --base reference
    python scripts/run_injection.py ... --nli microsoft/deberta-large-mnli     # add the NLI gate (recall delta)
`caught` = rejected by the gate; `leaked` = accepted although unsafe. The antonym row is the KNOWN lexical blind spot."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sipg_rl.verifier import FidelityVerifier, HFNLI
from sipg_rl.inject import INJECTORS, gate_report
from sipg_rl.data import synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl"); ap.add_argument("--base", default="synth", choices=["synth", "source", "reference"])
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--nli", default=None); ap.add_argument("--nli-threshold", type=float, default=0.5)
    a = ap.parse_args()
    if a.base == "synth":
        ex = [dict(src=s, out=s) for s in synth(a.n, 0)]
    else:
        rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
        k = lambda r, *n: next((r[x] for x in n if x in r and r[x]), None)
        pairs = [(k(r, "Expert", "source"), k(r, "Simple", "target")) for r in rows]
        ex = [dict(src=x, out=(x if a.base == "source" else y)) for x, y in pairs if x and y][:a.n]
    for e in ex:
        e["orig_out"] = e["out"]
    ver, gate = FidelityVerifier(), None
    if a.nli:
        nli = HFNLI(a.nli)
        gate = lambda s, o: min(nli(s, o).get("entailment", 0), nli(o, s).get("entailment", 0)) < a.nli_threshold
    print(f"{len(ex)} base examples ({a.base}); gate = lexical verifier" + (f" + NLI<{a.nli_threshold}" if a.nli else ""))
    print(f"{'injector':18}{'injected':>9}{'excluded':>9}{'caught':>8}{'leaked':>8}{'recall':>9}")
    for name, fn in INJECTORS.items():
        try:
            pert, n = fn([dict(e) for e in ex])
        except (RuntimeError, ValueError) as err:
            print(f"{name:18} RAISED: {err}")
            continue
        for p, e in zip(pert, ex):
            p["orig_out"] = e["orig_out"]
        r = gate_report(pert, ver, gate)
        print(f"{name:18}{r['injected']:9d}{r['excluded_base_rejected']:9d}{r['caught']:8d}{r['leaked']:8d}{100*r['recall']:8.1f}%")


if __name__ == "__main__":
    main()
