"""Evaluate the Health Literacy module: for each profile preset, simplify the same inputs and report readability
targets (FKGL in band, RVS, RTS), the verifier pass rate, and a verifier-blindness check.

  python scripts/profile_eval.py                                   # built-in sentences, rules realizer, all presets
  python scripts/profile_eval.py --jsonl medeasi_test.jsonl --src-field Expert --limit 100 --out eval_profile
  python scripts/profile_eval.py --presets low_literacy,balanced --llm-model qwen2.5:7b-instruct --revise
"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from statistics import mean
from sipg_rl.pipeline import SIPGRL, default_parts
from sipg_rl.patient_profile import PRESETS, profile_report, PatientProfile
from sipg_rl.coherence import calibrate_coherence
from sipg_rl.verifier import FidelityVerifier

BUILTIN = [
    "The hypertensive patient, who takes lisinopril 10 mg BID, reports no chest pain. Discontinue warfarin 5 mg if dyspnea occurs, because the drug is contraindicated in patients with renal impairment.",
    "Patients with hypertensive urgency may experience headache, epistaxis and dyspnea. Administer 25 mg of metoprolol twice daily unless bradycardia develops.",
    "The patient denies chest pain. Myocardial infarction was excluded by serial troponin measurements, and the electrocardiogram showed no ischemic changes.",
]


def load_texts(a):
    if not a.jsonl:
        return BUILTIN
    rows = [json.loads(l) for l in open(a.jsonl, encoding="utf8") if l.strip()]
    texts = [r[a.src_field] for r in rows if r.get(a.src_field)]
    return texts[: a.limit] if a.limit else texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl"); ap.add_argument("--src-field", default="Expert"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--ckpt", default="runs/v3/policy_seed0.pt")
    ap.add_argument("--llm-model", default=None); ap.add_argument("--llm-url", default="http://localhost:11434/v1")
    ap.add_argument("--revise", action="store_true")
    ap.add_argument("--out", default="eval_profile")
    a = ap.parse_args()

    parts = None
    if a.llm_model:
        from sipg_rl.llm import OpenAICompatLLM
        from sipg_rl.realize import HybridRealizer, RuleRealizer
        parts = default_parts(realizer=HybridRealizer(OpenAICompatLLM(a.llm_url, a.llm_model), RuleRealizer()))
    m = SIPGRL(parts, checkpoint=a.ckpt if os.path.exists(a.ckpt) else None)
    texts = load_texts(a)

    # verifier blindness: the verifier's public API has no profile parameter and its verdict depends only on (src, out)
    import inspect
    sig = inspect.signature(FidelityVerifier.check)
    assert not any("profile" in n for n in sig.parameters), "verifier must stay profile-blind"

    rows = []
    for name in [x for x in a.presets.split(",") if x]:
        prof = PRESETS[name]
        recs = []
        for t in texts:
            res = m.simplify(t, apply_all=True, profile=prof, revise=a.revise or bool(a.llm_model))
            rep = res.profile_report
            ok = FidelityVerifier().check(t, res.text, use_nli=False).ok     # same call regardless of profile
            recs.append(dict(fkgl=rep["fkgl"], in_band=rep["in_band"], rvs=rep["rvs"], rts=rep["rts"],
                             needs_revision=rep["needs_revision"], verified=ok,
                             revised=bool(res.profile_revision and res.profile_revision.get("accepted"))))
        rows.append(dict(profile=name, archetype=prof.archetype, n=len(recs),
                         fkgl=round(mean(r["fkgl"] for r in recs), 2), in_band=round(mean(r["in_band"] for r in recs), 3),
                         rvs=round(mean(r["rvs"] for r in recs), 2), rts=round(mean(r["rts"] for r in recs), 2),
                         needs_revision=round(mean(r["needs_revision"] for r in recs), 3),
                         verified=round(mean(r["verified"] for r in recs), 3),
                         llm_revised=round(mean(r["revised"] for r in recs), 3)))
        print(rows[-1], flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "profile_eval.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(rows=rows, n_texts=len(texts), llm=a.llm_model, revise=a.revise), open(os.path.join(a.out, "profile_eval.json"), "w"), indent=1)
    print(f"wrote {a.out}/profile_eval.csv")


if __name__ == "__main__":
    main()
