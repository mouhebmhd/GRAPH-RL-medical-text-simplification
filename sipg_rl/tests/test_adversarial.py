import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from sipg_rl import adversarial as A
from sipg_rl.pipeline import SIPGRL, default_parts

SRC = {
    "drop_number": "Take warfarin 5 mg daily after the procedure.",
    "drop_negation": "The patient reports no chest pain after exercise.",
    "swap_drug": "Start warfarin after the operation.",
    "hallucinate_number": "The patient started warfarin after the operation.",
    "antonym_drift": "The infection was excluded and symptoms improved.",
    "drop_hedge": "Possible pneumonia was seen on the scan.",
}


class KeywordNLI:
    """Stub NLI: contradiction when the output contains an antonym that the source does not."""
    def __call__(self, premise, hypothesis):
        bad = A.leaked("antonym_drift", premise, hypothesis)
        return {"entailment": 0.05 if bad else 0.95}


@pytest.fixture(scope="module")
def nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def mk(nlp, nli=None):
    return lambda rz: SIPGRL(default_parts(nlp=nlp, nli=nli, realizer=rz))


def test_corruptions_apply_and_are_detected_by_independent_detectors():
    for mode, s in SRC.items():
        assert A.applies(mode, s), mode
        assert A.leaked(mode, s, A.corrupt(mode, s)), mode
        assert not A.leaked(mode, s, s), mode


def test_lexical_gate_stops_lexical_corruptions_at_the_edit_gate(nlp):
    rows = {r.mode: r for r in A.selftest(mk(nlp), list(SRC.values()), modes=[m for m in A.MODES if m != "antonym_drift"])}
    for m, r in rows.items():
        assert r.injected >= 1 and r.leaked == 0 and r.edit_gate == r.injected and r.final_gate == 0, (m, r)


def test_semantic_drift_leaks_without_nli_and_final_fallback_fires_with_nli(nlp):
    off = A.selftest(mk(nlp), list(SRC.values()), modes=["antonym_drift"])[0]
    assert off.injected >= 1 and off.leaked == off.injected and off.recall == 0.0       # the E1 gap, measured
    on = A.selftest(mk(nlp, KeywordNLI()), list(SRC.values()), modes=["antonym_drift"])[0]
    assert on.leaked == 0 and on.final_gate == on.injected and on.recall == 1.0         # the fallback path executes
