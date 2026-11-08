import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from sipg_rl.pipeline import default_parts
from sipg_rl.env import make_env, INVOKE
from sipg_rl.realize import HybridRealizer, RuleRealizer


class Recorder:
    """LLM stand-in with the 4-argument signature; records every call (never a real model)."""
    def __init__(self, answers): self.answers, self.calls = answers, []
    def __call__(self, prompt, text, context="", mode=None):
        self.calls.append((mode, text)); return self.answers.get(text.lower(), text)


def run_node(parts, sent, kind, key=None):
    g = parts["builder"].build_text(sent)
    node = next(n for n in g.nodes if n.kind == kind and (key is None or n.meta.get("key") == key))
    env = make_env(parts); env.reset(g); env.cur = node.id
    env.step(INVOKE)
    return env, parts["realizer"]


def parts_with(llm, **kw):
    return default_parts(realizer=HybridRealizer(llm, RuleRealizer(), **kw))


def test_rules_serve_covered_terms_and_the_llm_is_not_called():
    llm = Recorder({"orally": "THE LLM SHOULD NOT BE ASKED"})
    env, rz = run_node(parts_with(llm), "Take warfarin 5 mg orally.", "LEX", "orally")
    assert "by mouth" in env.text and rz.last_via == "rules" and llm.calls == []
    assert env.trail[-1]["realizer"] == "rules" and env.trail[-1]["candidate"]["candidate_used"] == "by mouth"


def test_llm_is_a_constrained_lexical_fallback_for_uncovered_words():
    llm = Recorder({"cardiomyopathy": "heart muscle disease"})
    env, rz = run_node(parts_with(llm), "The patient has cardiomyopathy.", "CPLX")
    assert "heart muscle disease" in env.text and rz.last_via == "llm" and llm.calls == [("lexical", "cardiomyopathy")]


@pytest.mark.parametrize("bad", ["a long explanation of the word with many words", "heart disease.", "disease of 3 heart muscles"])
def test_lexical_constraints_reject_restructuring_digits_and_sentences(bad):
    llm = Recorder({"cardiomyopathy": bad})
    env, rz = run_node(parts_with(llm), "The patient has cardiomyopathy.", "CPLX")
    assert env.text == env.src and rz.last_via is None


def test_conditions_are_never_split_off_by_the_llm():
    sent = "Call your doctor if the patient develops a high fever and chills after taking the tablets."
    llm = Recorder({})
    g = default_parts(realizer=HybridRealizer(llm, RuleRealizer())) ["builder"].build_text(sent)
    node = next(n for n in g.nodes if n.kind == "ADVCL")
    assert node.relation == "condition"
    env = make_env(parts_with(llm)); env.reset(g); env.cur = node.id; env.step(INVOKE)
    assert env.text == env.src and [c for c in llm.calls if c[0] == "clause"] == []


def test_llm_split_must_keep_the_original_words():
    sent = "The patient was admitted to the hospital, but the doctors could not find the cause of the problem for many days."
    g = default_parts()["builder"].build_text(sent)
    cj = next((n for n in g.nodes if n.kind == "CONJ_CL"), None)
    if cj is None:
        pytest.skip("parser produced no CONJ_CL for this sentence")
    # rules split CONJ_CL themselves -> LLM not needed; force the structural path through a rules stub that declines
    class Decline(RuleRealizer):
        def apply(self, *a): self.last_via = self.last_info = None; return False
    paraphrase = Recorder({})
    paraphrase.__call__ = lambda *a, **k: "Totally different words appear in this sentence. Nothing else remains."
    hr = HybridRealizer(lambda p, t, c="", m=None: "Totally different words appear here. Nothing remains of the original.", Decline())
    P = default_parts(realizer=hr); env = make_env(P); env.reset(g); env.cur = cj.id; env.step(INVOKE)
    assert env.text == env.src                                    # keep-ratio constraint rejected the paraphrase
