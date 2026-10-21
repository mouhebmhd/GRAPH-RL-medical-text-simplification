import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from sipg_rl.pipeline import SIPGRL, default_parts
from sipg_rl.realize import RoutedRealizer, RuleRealizer, lexical_guard


@pytest.fixture(scope="module")
def nlp():
    import spacy
    return spacy.load("en_core_web_sm")


class Recorder:
    """Fake LLM: records (span) it was asked about; answers from a table, else echoes the span."""
    def __init__(self, table=None):
        self.calls, self.table = [], table or {}

    def __call__(self, prompt, text, context=""):
        self.calls.append(text.strip())
        return self.table.get(text.strip().lower(), text)


def run(nlp, llm, text, **kw):
    rz = RoutedRealizer(llm, RuleRealizer(), **kw)
    m = SIPGRL(default_parts(nlp=nlp, realizer=rz))
    return m.simplify(text, apply_all=True), rz


def test_lexicon_terms_never_reach_the_llm_c4(nlp):
    llm = Recorder()                                    # an LLM that would have returned the term unchanged
    res, rz = run(nlp, llm, "He has hypertensive urgency and takes lisinopril 10 mg.")
    assert "very high blood pressure" in res.text and "lisinopril 10 mg" in res.text
    assert not any("hypertensive" in c.lower() for c in llm.calls) and rz.stats["rules"] >= 1
    ops = [o for s in res.sentences for o in s["operations"]]
    assert ops and all(o["via"] == "rules" for o in ops)


def test_llm_gets_unknown_complex_words_only_and_edit_is_attributed(nlp):
    llm = Recorder({"pharmacokinetic": "how drugs move"})
    res, rz = run(nlp, llm, "The pharmacokinetic profile was reviewed.")
    assert llm.calls and all(len(c.split()) <= 3 for c in llm.calls)
    assert "how drugs move" in res.text
    assert any(o["via"] == "llm" for s in res.sentences for o in s["operations"])


def test_structural_rewrites_from_the_llm_are_blocked_by_the_guard(nlp):
    bad = "It was a profile. The doctors looked at it carefully and then they wrote a long note."
    llm = Recorder({"pharmacokinetic": bad})
    res, rz = run(nlp, llm, "The pharmacokinetic profile was reviewed.")
    assert res.text == "The pharmacokinetic profile was reviewed." and rz.stats["llm_guard_rejected"] >= 1


def test_clause_nodes_are_not_sent_to_the_llm_by_default_but_can_be_opted_in(nlp):
    t = "She reported that the intermittent episodes worsened considerably."
    off, on = Recorder(), Recorder()
    run(nlp, off, t)
    run(nlp, on, t, allow_llm_structure=True)
    clause = "that the intermittent episodes worsened considerably"
    assert clause not in off.calls                       # default: only single words / short spans reach the LLM
    assert clause in on.calls                            # explicit opt-in sends the embedded clause too
    assert all(len(c.split()) <= 3 for c in off.calls)


def test_guard_rules():
    assert lexical_guard("dyspnea", "trouble breathing") and not lexical_guard("dyspnea", "dyspnea")
    assert not lexical_guard("dyspnea", "trouble. breathing") and not lexical_guard("dyspnea", "trouble breathing 5 times")
    assert not lexical_guard("dyspnea", "a very long explanation of what dyspnea means for patients") and not lexical_guard("x", "")
