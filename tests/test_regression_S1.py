"""C4 regression: the 'hypertensive urgency' case. Pins WHAT HAPPENS to that node, so 'skipped vs reverted vs no-op' is never a guess.
Finding encoded here: the rule lexicon HAS the phrase, so it is applied. If the phrase were missing from the lexicon the AMOD prompt
would be a silent NO-OP for rules (not a traversal failure); the hybrid realizer's constrained LLM fallback rescues it."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sipg_rl.pipeline import SIPGRL, default_parts
from sipg_rl.realize import HybridRealizer, RuleRealizer
from sipg_rl.trail import explain_term_outcome
from sipg_rl import lexicon

S1 = "Discontinue lisinopril 10 mg if the patient develops hypertensive urgency, because the drug is contraindicated in renal impairment."
CK = os.path.join(os.path.dirname(__file__), "..", "sipg_policy_v3.pt")


def run(parts, apply_all=True):
    return SIPGRL(parts, checkpoint=CK if os.path.exists(CK) else None).simplify(S1, apply_all=apply_all)


def test_lexicon_has_the_phrase_and_rules_apply_it():
    assert "hypertensive urgency" in lexicon.PLAIN_TERMS
    r = run(default_parts())
    assert explain_term_outcome(r, "hypertensive urgency") == "applied"
    assert "hypertensive" not in r.text.lower() and "if" in r.text.lower() and "10 mg" in r.text


def test_policy_mode_outcome_is_applied_or_skipped_never_reverted():
    r = run(default_parts(), apply_all=False)
    assert explain_term_outcome(r, "hypertensive urgency") in ("applied", "skipped")


def test_without_a_rule_entry_rules_noop_and_hybrid_llm_rescues(monkeypatch):
    saved = dict(lexicon.PLAIN_TERMS)
    for k in ("hypertensive urgency",):
        monkeypatch.delitem(lexicon.PLAIN_TERMS, k)
    try:
        assert explain_term_outcome(run(default_parts()), "hypertensive") == "noop"         # the C4 hypothesis, reproduced
        llm = lambda prompt, text, ctx="", mode=None: "very high blood pressure" if "hypertensive" in text.lower() else text
        r = run(default_parts(realizer=HybridRealizer(llm, RuleRealizer())))
        assert explain_term_outcome(r, "hypertensive") == "applied" and "very high blood pressure" in r.text
    finally:
        lexicon.PLAIN_TERMS.update(saved)
