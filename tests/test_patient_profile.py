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


def test_scores_differ_between_profiles():
    from sipg_rl.patient_profile import score_all_profiles
    text = ("The patient has high blood pressure and takes one pill every morning. "
            "Electrocardiographic abnormalities warrant immediate cardiological reevaluation. Stop.")
    rows = score_all_profiles(text, text, lambda t: 0.5)
    assert len({r["S_output"] for r in rows}) >= 4, [(r["key"], r["S_output"]) for r in rows]
    by = {r["key"]: r for r in rows}
    assert by["balanced"]["ideal_grade"] > by["low_literacy"]["ideal_grade"]
    assert by["low_literacy"]["parts"]["coherence"] < by["balanced"]["parts"]["coherence"]   # steady-flow cap bites


def test_score_all_profiles_for_dashboard():
    from sipg_rl.patient_profile import score_all_profiles
    base = lambda t: 0.5
    easy = "The patient has high blood pressure and takes one pill every morning."
    rows = score_all_profiles(easy, ERRATIC, base)
    assert [r["key"] for r in rows] == list(PRESETS) and not any(r["in_use"] for r in rows)
    assert all(0 <= r["S_output"] <= 100 and 0 <= r["S_source"] <= 100 for r in rows)
    by = {r["key"]: r for r in rows}
    assert by["low_literacy"]["S_output"] > by["balanced"]["S_output"] or by["low_literacy"]["in_band"]
    rows = score_all_profiles(easy, easy, base, selected=PRESETS["low_literacy"])
    assert [r["key"] for r in rows if r["in_use"]] == ["low_literacy"] and len(rows) == len(PRESETS)
    rows = score_all_profiles(easy, easy, base, selected=PatientProfile(core_proficiency=0.9))
    assert rows[-1]["key"] == "selected" and rows[-1]["in_use"] and len(rows) == len(PRESETS) + 1


class StubChat:
    def __init__(self, replies): self.replies, self.calls = list(replies), []
    def chat(self, system, user, max_tokens=500):
        self.calls.append((system, user)); return self.replies.pop(0) if self.replies else ""


def test_llm_rating_prompt_carries_every_profile_description():
    from sipg_rl.patient_profile import build_rating_prompt, describe_profile, RATING_SYSTEM
    u = build_rating_prompt(PRESETS, "SRC TEXT", "OUT TEXT")
    for k, p in PRESETS.items():
        assert f"- {k}: {describe_profile(p)}" in u
    assert "SRC TEXT" in u and "OUT TEXT" in u
    assert "Do NOT judge medical correctness" in RATING_SYSTEM and "do not give every profile the same score" in RATING_SYSTEM
    assert "SPIRITUAL-FIRST" in describe_profile(PRESETS["spiritual_first"]) and "STEADY" in describe_profile(PRESETS["low_literacy"])


def test_llm_rating_parse_clamp_and_fallback():
    from sipg_rl.patient_profile import llm_rate_profiles, parse_ratings
    keys = list(PRESETS)
    good = '```json\n{"scores":[' + ",".join(f'{{"profile":"{k}","score":{i*15+10},"reason":"r{i}"}}' for i, k in enumerate(keys[:-1])) + ']}\n```'
    stub = StubChat([good, '{"scores":[{"profile":"spiritual_first","score":140,"reason":"x"}]}'])
    r = llm_rate_profiles(stub, "s", "o", PRESETS)
    assert [r[k]["score"] for k in keys[:-1]] == [10, 25, 40, 55, 70] and r["spiritual_first"]["score"] == 100   # clamped
    assert len(stub.calls) == 2 and "- spiritual_first:" in stub.calls[1][1] and "- balanced:" not in stub.calls[1][1]
    assert parse_ratings("not json", keys)["balanced"]["score"] is None
    assert all(v["score"] is None for v in llm_rate_profiles(None, "s", "o", PRESETS).values())


def test_score_all_profiles_attaches_llm_scores():
    from sipg_rl.patient_profile import score_all_profiles
    reply = '{"scores":[' + ",".join(f'{{"profile":"{k}","score":{50+i},"reason":"ok"}}' for i, k in enumerate(PRESETS)) + ']}'
    rows = score_all_profiles("A short text.", "A short text.", lambda t: .5, llm=StubChat([reply]))
    assert [r["llm_score"] for r in rows] == [50 + i for i in range(len(PRESETS))] and all(r["description"] for r in rows)
    assert all(r["llm_score"] is None for r in score_all_profiles("a", "a", lambda t: .5))


def test_notes_flow_into_prompts_and_descriptions():
    from sipg_rl.patient_profile import describe_profile
    p = PatientProfile(core_proficiency=0.2, notes="Retired farmer, prefers short spoken-style sentences.")
    assert "Retired farmer" in build_system_prompt(p) and "Retired farmer" in describe_profile(p)
    assert "context only" in build_system_prompt(p)
    assert len(PatientProfile(notes="x" * 5000).notes) == 1000
    assert PatientProfile.from_dict(p.to_dict()).notes == p.notes


def test_extract_profile_from_text():
    from sipg_rl.patient_profile import extract_profile_from_text
    reply = 'Sure: {"core_proficiency":0.2,"coherence_need":"steady","spiritual_first":true,"digital_preference":0.1,"adaptability":0.3,"topic_variance":0.0}'
    q = extract_profile_from_text(StubChat([reply]), "Elderly, devout, reads printed leaflets.")
    assert q.core_proficiency == 0.2 and q.spiritual_first and q.coherence_need == "steady" and q.notes.startswith("Elderly")
    assert extract_profile_from_text(StubChat(["no json"]), "x") is None
    assert extract_profile_from_text(None, "x") is None and extract_profile_from_text(StubChat([]), "  ") is None
    bad = extract_profile_from_text(StubChat(['{"core_proficiency":0.5,"coherence_need":"weird"}']), "x")
    assert bad.coherence_need == "auto"


def test_explain_suitability_llm_and_fallback():
    from sipg_rl.patient_profile import explain_suitability
    easy = "The patient has high blood pressure and takes one pill every morning."
    e = explain_suitability(StubChat(["Short plain words. Remaining gaps: none noted."]), PRESETS["low_literacy"], ERRATIC, easy, "Low literacy")
    assert e["source"] == "llm" and e["text"].startswith("Short plain") and e["profile_label"] == "Low literacy" and e["checks"]
    e2 = explain_suitability(None, PRESETS["low_literacy"], easy, ERRATIC, "Low literacy")       # hard text: honest "not met"
    assert e2["source"] == "rules" and "Remaining gaps" in e2["text"] and not all(c["ok"] for c in e2["checks"])
    e3 = explain_suitability(StubChat([""]), PRESETS["balanced"], easy, easy)                       # empty LLM reply -> rules
    assert e3["source"] == "rules"


def test_preset_card_keeps_user_notes_when_matching():
    from sipg_rl.patient_profile import score_all_profiles
    mine = PatientProfile(**{**PRESETS["low_literacy"].to_dict(), "name": "", "notes": "Prefers large print."})
    rows = score_all_profiles("a b.", "a b.", lambda t: .5, selected=mine)
    row = next(r for r in rows if r["in_use"])
    assert row["key"] == "low_literacy" and "large print" in row["description"] and len(rows) == len(PRESETS)


def test_viz_summary_shapes():
    from sipg_rl.coherence import viz_summary, split_sentences
    v = viz_summary(ERRATIC, STEADY)
    assert len(v["flow"]["source"]) == len(split_sentences(ERRATIC)) and len(v["flow"]["output"]) == len(split_sentences(STEADY))
    assert v["flow"]["rvs"]["output"] < v["flow"]["rvs"]["source"] and v["fkgl"]["output"] < v["fkgl"]["source"]
    assert set(v["stats"]["source"]) == {"words", "sentences", "avg_len", "long_words"}
    assert v["stats"]["source"]["long_words"] > v["stats"]["output"]["long_words"]


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
