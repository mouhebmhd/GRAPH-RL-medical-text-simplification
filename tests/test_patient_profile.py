"""Health Literacy / Patient Profile module. Pure-python tests run anywhere; pipeline tests need spaCy + torch."""
import inspect, os, re
import pytest
from sipg_rl import patient_profile as pp
from sipg_rl.patient_profile import (PatientProfile, PRESETS, build_system_prompt, span_style_suffix, band_score,
                                     ProfileSuitabilityScorer, profile_report, load_profile, SPIRITUAL_SEQUENCE)
from sipg_rl.coherence import rvs, rts, fkgl, calibrate_coherence
from sipg_rl.verifier import FidelityVerifier

STEADY = "Take one pill each day. Drink water with it. Do not skip a day. Call us if you feel sick."
ERRATIC = ("Take it daily. Notwithstanding gastrointestinal intolerance, antihypertensive pharmacotherapeutic "
           "adherence necessitates uninterrupted administration throughout hospitalization. Stop. "
           "Electrocardiographic abnormalities warrant immediate cardiological reevaluation.")


def test_schema_clamps_and_validates():
    p = PatientProfile(core_proficiency=3, digital_preference=-1)
    assert p.core_proficiency == 1.0 and p.digital_preference == 0.0
    with pytest.raises(ValueError):
        PatientProfile(coherence_need="chaotic")
    assert PatientProfile.from_dict({"core_proficiency": .2, "unknown_key": 1}).core_proficiency == .2


@pytest.mark.parametrize("name,arch", [("balanced", "balanced"), ("specialized_digital", "specialized"),
                                       ("specialized_functional", "specialized"), ("transitional", "transitional")])
def test_archetypes(name, arch):
    assert PRESETS[name].archetype == arch


def test_targets_follow_proficiency_and_coherence():
    low, high = PRESETS["low_literacy"], PatientProfile(core_proficiency=0.9)
    assert (low.targets().grade_lo, low.targets().grade_hi) == (4.0, 6.0)
    assert high.targets().grade_lo > low.targets().grade_hi - 1e-9
    assert low.steady_flow and low.targets().rvs_max is not None
    assert not high.steady_flow and high.targets().rvs_max is None
    assert PatientProfile(core_proficiency=0.9, coherence_need="steady").steady_flow       # explicit beats auto
    assert PRESETS["transitional"].steady_flow


def test_prompts_encode_each_dimension():
    assert "4th-6th grade" in build_system_prompt(PRESETS["low_literacy"])
    assert "uniform" in build_system_prompt(PRESETS["low_literacy"])
    t = build_system_prompt(PRESETS["transitional"])
    assert "analogies" in t and "predictable" in t
    s = build_system_prompt(PRESETS["spiritual_first"])
    assert SPIRITUAL_SEQUENCE in s and "Never contradict the patient's spiritual worldview" in s
    assert "step-by-step" in build_system_prompt(PRESETS["specialized_functional"])
    d = build_system_prompt(PRESETS["specialized_digital"])
    assert "Do NOT invent links" in d
    for p in PRESETS.values():                       # every prompt carries the safety footer
        assert "negations" in build_system_prompt(p)
    assert "grade" in span_style_suffix(PRESETS["low_literacy"]).lower()


def test_rvs_rts_separate_steady_from_erratic():
    assert rvs("One sentence only.") == 0.0 and rts("One sentence only.") == 0.0
    assert rvs(STEADY) < rvs(ERRATIC) and rts(STEADY) < rts(ERRATIC)
    q = calibrate_coherence([STEADY, STEADY + " Rest well."], 0.75)
    assert q["rvs_max"] >= 0 and q["rts_max"] >= 0


def test_band_score_and_scorer_shift():
    assert band_score(5, 4, 6) == 1.0 and band_score(10, 4, 6) == 0.0 and 0 < band_score(7, 4, 6) < 1
    base = lambda t: 0.5
    low, high = ProfileSuitabilityScorer(base, PRESETS["low_literacy"]), ProfileSuitabilityScorer(base, PatientProfile(core_proficiency=0.95))
    easy = "The patient has high blood pressure and takes one pill every morning."      # FKGL ~5.8: inside 4-6, outside 8-10
    assert 4 <= fkgl(easy) <= 6
    assert low(easy) > high(easy)                      # same text, same base score: only the target band differs
    hard = ERRATIC
    assert high(hard) >= low(hard)


def test_scorer_wrapper_passes_through_base_attributes():
    class Base:
        tg = 6.0
        def __call__(self, t): return 0.5
        def features(self, t): return {"x": 1}
    w = ProfileSuitabilityScorer(Base(), PRESETS["balanced"])
    assert w.tg == 6.0 and w.features("a") == {"x": 1}


def test_report_flags_revision_but_is_not_a_safety_verdict():
    r = profile_report(ERRATIC, PRESETS["low_literacy"])
    assert r.needs_revision and r.reasons
    assert not hasattr(r, "ok") and not hasattr(r, "violations")      # no safety semantics on the profile report
    assert not profile_report(STEADY, PRESETS["low_literacy"]).needs_revision


def test_load_profile(tmp_path):
    assert load_profile(None) is None and load_profile("low_literacy").name == "low_literacy"
    f = tmp_path / "p.json"; f.write_text('{"core_proficiency": 0.1}')
    assert load_profile(str(f)).core_proficiency == 0.1
    assert load_profile('{"spiritual_first": true}').spiritual_first
    with pytest.raises(ValueError):
        load_profile("nonsense-preset")


# ---- THE INVARIANT: the verifier is profile-blind ---------------------------------------------------------------
def test_verifier_is_profile_blind():
    assert not any("profile" in n for n in inspect.signature(FidelityVerifier.check).parameters)
    assert not any("profile" in n for n in inspect.signature(FidelityVerifier.__init__).parameters)
    import sipg_rl.verifier as v
    src = open(v.__file__, encoding="utf8").read()
    assert "patient_profile" not in src and not re.search(r"\bprofile\b", src, re.I)


def test_verdict_depends_only_on_source_and_output():
    ver = FidelityVerifier(nli=None)
    src, bad = "Do not take warfarin 5 mg if you feel dizzy.", "Take warfarin 50 mg."
    first = ver.check(src, bad, use_nli=False)
    for _ in PRESETS:                                  # same call, any number of profiles in play -> identical verdict
        again = ver.check(src, bad, use_nli=False)
        assert (again.ok, again.violations) == (first.ok, first.violations)
    assert not first.ok


# ---- LLM adapter ------------------------------------------------------------------------------------------------
def test_llm_style_suffix_and_document_rewrite(monkeypatch):
    from sipg_rl.llm import OpenAICompatLLM
    seen = []
    def fake(self, path, payload=None, timeout=None):
        seen.append(payload["messages"][0]["content"])
        return {"choices": [{"message": {"content": "plain words"}}]}
    monkeypatch.setattr(OpenAICompatLLM, "_request", fake)
    llm = OpenAICompatLLM()
    llm("p", "dyspnea", "ctx", mode="lexical")
    assert "Patient style" not in seen[-1]
    llm.style_suffix = "Use a 4th-6th grade vocabulary."
    llm("p", "dyspnea", "ctx", mode="lexical")                 # different cache key: the suffix is part of it
    assert "Patient style: Use a 4th-6th grade vocabulary." in seen[-1] and len(seen) == 2
    assert llm.rewrite_document("SYS", "Some text.") == "plain words"
    assert seen[-1] == "SYS"


# ---- pipeline (needs spaCy + torch + en_core_web_sm) --------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    pytest.importorskip("spacy"); pytest.importorskip("torch")
    from sipg_rl.pipeline import SIPGRL
    return SIPGRL()


TEXT = "The hypertensive patient, who takes lisinopril 10 mg BID, reports no chest pain."


def test_pipeline_returns_report_and_verifier_never_sees_profile(model):
    calls = []
    orig = model.parts["verifier"].check
    model.parts["verifier"].check = lambda *a, **k: (calls.append(k), orig(*a, **k))[1]
    try:
        res = model.simplify(TEXT, profile=PRESETS["low_literacy"])
    finally:
        model.parts["verifier"].check = orig
    assert res.profile["name"] == "low_literacy" and res.profile_report["target_band"] == (4.0, 6.0)
    assert calls and all("profile" not in k for k in calls)
    assert "10" in res.text and "no" in res.text.lower()          # critical spans survive


def test_pipeline_without_profile_is_unchanged(model):
    a, b = model.simplify(TEXT), model.simplify(TEXT, profile=None)
    assert a.text == b.text and a.profile is None and a.profile_report is None


def test_revision_is_discarded_when_verifier_rejects(model):
    from sipg_rl.pipeline import SIPGRL, default_parts
    from sipg_rl.realize import HybridRealizer, RuleRealizer

    class Stub:
        style_suffix = ""
        def __init__(self, out): self.out = out
        def __call__(self, prompt, text, context="", mode=None): return ""
        def rewrite_document(self, system, text): return self.out

    def run(out):
        stub = Stub(out)
        parts = default_parts(nlp=model.parts["nlp"], realizer=HybridRealizer(stub, RuleRealizer()))
        return SIPGRL(parts).simplify(TEXT, profile=PRESETS["spiritual_first"], revise=True)

    bad = run("Take lisinopril 100 mg. The patient reports chest pain.")          # altered dose, dropped negation
    assert bad.profile_revision["attempted"] and not bad.profile_revision["accepted"]
    assert bad.profile_revision["reason"] == "rejected by verifier" and "100" not in bad.text
