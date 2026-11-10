"""Simplify free text with a trained policy and print the audit trail (the traversal path)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sipg_rl.pipeline import SIPGRL, default_parts
from sipg_rl.patient_profile import load_profile, PRESETS, build_system_prompt

DEFAULT = ("The hypertensive patient, who takes lisinopril 10 mg BID, reports no chest pain. "
           "Discontinue warfarin 5 mg if dyspnea occurs, because the drug is contraindicated in patients with renal impairment.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default=DEFAULT)
    ap.add_argument("--ckpt", default="sipg_policy.pt")
    ap.add_argument("--profile", default=None, help=f"preset ({', '.join(PRESETS)}), a .json file, or inline JSON")
    ap.add_argument("--revise", action="store_true", help="try one document-level LLM revision under the profile (needs --llm-model)")
    ap.add_argument("--llm-model", default=None, help="e.g. qwen2.5:7b-instruct (Ollama); enables the hybrid realizer")
    ap.add_argument("--llm-url", default="http://localhost:11434/v1")
    ap.add_argument("--selective", action="store_true", help="let the policy skip operations (default: apply ALL operations, merged)")
    a = ap.parse_args()
    parts = None
    if a.llm_model:
        from sipg_rl.llm import OpenAICompatLLM
        from sipg_rl.realize import HybridRealizer, RuleRealizer
        parts = default_parts(realizer=HybridRealizer(OpenAICompatLLM(a.llm_url, a.llm_model), RuleRealizer()))
    m = SIPGRL(parts, checkpoint=a.ckpt if os.path.exists(a.ckpt) else None)
    prof = load_profile(a.profile)
    if prof:
        print(f"PROFILE: {prof.name or 'custom'} -> archetype={prof.archetype}, level={prof.proficiency_level}, "
              f"steady_flow={prof.steady_flow}, targets={prof.targets()}")
        print("GENERATION PROMPT:", build_system_prompt(prof), "\n")
    res = m.simplify(a.text, apply_all=not a.selective, profile=prof, revise=a.revise or bool(a.llm_model))
    if res.profile_report:
        r = res.profile_report
        print(f"PROFILE REPORT: FKGL {r['fkgl']:.1f} (target {r['target_band']}), RVS {r['rvs']:.2f}, RTS {r['rts']:.2f}, "
              f"needs_revision={r['needs_revision']} {r['reasons']}")
        if res.profile_revision and res.profile_revision.get("attempted"):
            print("LLM REVISION:", res.profile_revision["reason"], res.profile_revision.get("violations"))
    print("OUTPUT:", res.text, "\n")
    for s in res.sentences:
        print(f"SOURCE : {s['source']}\nOUTPUT : {s['output']}")
        print(f"S: {s['S_before']*100:.0f}% -> {s['S_after']*100:.0f}% | verified={s['verified']} {s['violations']}")
        for t in s["trail"]:
            print(f"   step {t['step']:2d} INVOKE node {t['node']} [{t['kind']}] -> {t.get('event')}  (r={t['reward']:+.3f})")
        print()
