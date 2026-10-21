"""Machinery tests for the corrected reward (A1 Sim, A2/B2 S, B1 gamma). Synthetic data only: these validate that the
metrics behave as specified, they are not evidence about any real corpus."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from sipg_rl.scoring import ContentEncoder, ConceptEncoder, BandSuitabilityScorer, FeatureSuitabilityScorer
from sipg_rl import adversarial as A

SRC = "The hypertensive patient reports no chest pain and takes lisinopril 10 mg BID. Possible pneumonia was seen if symptoms persist."
FAITHFUL = ("The patient who has high blood pressure reports no chest pain. The patient takes lisinopril 10 mg twice a day. "
            "Possible pneumonia was seen if symptoms persist.")
EXPLAINED = FAITHFUL.replace("Possible pneumonia", "Possible pneumonia (a lung infection)")
E = ContentEncoder()
sim = lambda out, src=SRC: E.cos(E.encode(out), E.encode(src))


# ----------------------------------------------------------------------------------------------------- A1
def test_sim_is_one_for_copy_and_for_faithful_rewrites_regardless_of_length():
    assert sim(SRC) == 1.0
    assert sim(FAITHFUL) > 0.98            # sentence split + jargon -> plain wording: NOT penalised (old encoder: 0.84)
    assert sim(EXPLAINED) > 0.95           # appended clarification is cheap (length-invariance)
    old = ConceptEncoder()
    assert old.cos(old.encode(FAITHFUL), old.encode(SRC)) < sim(FAITHFUL) - 0.1


@pytest.mark.parametrize("mode", ["drop_number", "drop_negation", "drop_hedge", "swap_drug"])
def test_sim_ranks_every_unsafe_edit_below_the_faithful_rewrite(mode):
    bad = A.corrupt(mode, FAITHFUL)
    assert bad != FAITHFUL
    assert sim(bad) < sim(FAITHFUL) - 0.15, (mode, sim(bad), sim(FAITHFUL))


def test_sim_penalises_invented_numbers_and_unrelated_text():
    bad = A.corrupt("hallucinate_number", FAITHFUL.replace("lisinopril", "warfarin")).replace("warfarin", "lisinopril")
    assert sim(bad) < sim(FAITHFUL) - 0.1
    assert sim("The weather is nice today.") < 0.1


def test_the_gold_vs_noop_failure_is_gone_for_a_faithful_reference():
    """The audit failure: reference scored 0.514 while doing nothing scored 1.000. A faithful paraphrase must now sit
    next to the no-op and clearly above an unsafe rewrite (the full-reward ordering also needs S, tested below)."""
    assert sim(SRC) - sim(FAITHFUL) < 0.02 and sim(FAITHFUL) > sim(A.corrupt("drop_number", FAITHFUL)) + 0.3


# ----------------------------------------------------------------------------------------------------- A2 / B2
HARD = "Administration of anticoagulant pharmacotherapy necessitates comprehensive hematological surveillance throughout hospitalization."
TARGET = ("Your doctor will check your blood pressure at every visit because high blood pressure can slowly cause "
          "problems with the heart and the kidneys. Please take your pills at the same time each day.")
FRAGMENTS = "Take it. Be good. Do it."
B, OLD = BandSuitabilityScorer(), FeatureSuitabilityScorer(6.0)


def test_band_scorer_peaks_near_target_and_penalises_both_directions():
    assert B(TARGET) > B(HARD) + 0.25            # over-complex
    assert B(TARGET) > B(FRAGMENTS) + 0.1        # over-simplified fragments (old scorer gave both ~1.0 / ~0.15)
    assert OLD(FRAGMENTS) > OLD(TARGET)          # documents the old monotone failure
    assert 0.0 <= B(HARD) <= B(TARGET) <= 1.0


class _AtGrade(BandSuitabilityScorer):
    """Scorer whose only varying input is the readability grade, to probe the shape of the band directly."""
    def __init__(self, grade):
        super().__init__()
        self.g = grade

    def features(self, text):
        return dict(grade=self.g, asl=13.0, poly=0.0, jargon=0.0, asw=1.3)


def test_band_scorer_keeps_a_gradient_near_the_target_b2():
    """Away from the target S must fall smoothly on BOTH sides (no clipped plateau, no saturation at the top)."""
    c = B.bands["grade"][0]
    s = lambda d: _AtGrade(c + d)("x")
    assert s(0) == pytest.approx(1.0) and s(0) > s(1) > s(2) > s(4) > s(8)
    assert s(0) > s(-1) > s(-2) > s(-4) > s(-8)
    assert s(-4) > s(4)                           # the lower tail is deliberately more tolerant than the upper tail


def test_band_scorer_does_not_penalise_clause_splitting():
    """The failure that the first (symmetric) band caused: splitting one 13-word sentence must not look 'over-simplified'."""
    one = "The hypertensive patient reports no chest pain and takes lisinopril 10 mg BID."
    two = "The patient who has high blood pressure reports no chest pain. The patient takes lisinopril 10 mg twice a day."
    assert B(two) > B(one)


def test_fit_anchors_the_band_to_a_reference_distribution():
    refs = [TARGET, TARGET.replace("blood pressure", "blood sugar"),
            "Your nurse will call you next week. She will explain the results and answer any questions you have.",
            "Take one pill in the morning with food. If you feel dizzy, sit down and call us. We can change the medicine if needed.",
            "This test looks at how well your kidneys clean your blood. You do not need to fast before it. Results are ready in two days.",
            "Your blood sugar was higher than we want. Eat less sugary food and walk every day. We will check it again in three months."]
    fitted = BandSuitabilityScorer().fit(refs)
    ref_mean = np.mean([fitted(r) for r in refs])
    assert ref_mean > 0.7                                        # the reference distribution lands near the ceiling
    assert fitted(HARD) < ref_mean - 0.3 and fitted(FRAGMENTS) < ref_mean - 0.2
    again = BandSuitabilityScorer.from_dict(fitted.to_dict())
    assert abs(again(TARGET) - fitted(TARGET)) < 1e-9


def test_saturation_is_reduced_on_invoke_all_outputs_b2():
    import spacy
    from sipg_rl.pipeline import SIPGRL, default_parts
    from sipg_rl.data import synth
    m = SIPGRL(default_parts(nlp=spacy.load("en_core_web_sm")))
    outs = [m.simplify(t, apply_all=True).text for t in synth(80, 3)]
    sat_new = np.mean([B(o) >= 0.95 for o in outs])
    sat_old = np.mean([OLD(o) >= 0.95 for o in outs])
    assert sat_new < 0.35 and sat_new < 0.6 * sat_old     # old scorer: 67% saturated on real outputs, ~45% here


# ----------------------------------------------------------------------------------------------------- B1
def test_gamma_is_material_relative_to_a_typical_edit_gain_b1():
    import spacy
    from sipg_rl.pipeline import default_parts
    from sipg_rl.env import SimplifyEnv
    from sipg_rl.baselines import InvokeAllPolicy, run_episode
    from sipg_rl.data import synth
    P = default_parts(nlp=spacy.load("en_core_web_sm"))
    env = SimplifyEnv(P["scorer"], P["encoder"], P["verifier"], P["realizer"], P["cfg"])
    gains = []
    for t in synth(60, 5):
        run_episode(env, P["builder"].build_text(t), InvokeAllPolicy())
        gains += [tr["reward"] + P["cfg"].gamma for tr in env.trail if tr["action"] == "INVOKE" and tr.get("event") == "edited"]
    med = float(np.median(gains))
    assert P["cfg"].gamma >= 0.02 and P["cfg"].coverage_bonus == 0.0
    assert 0.1 <= P["cfg"].gamma / med <= 0.6, (P["cfg"].gamma, med)   # was ~0.09 (gamma 0.0025-0.01 vs gain 0.115)
