"""No output leaves the assembler without a final-gate pass: a failing sentence is replaced by its source, never emitted."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sipg_rl.pipeline import SIPGRL, default_parts
from sipg_rl.verifier import FidelityVerifier
from sipg_rl.adversarial import AdversarialRealizer
from sipg_rl.data import synth

CK = os.path.join(os.path.dirname(__file__), "..", "sipg_policy_v3.pt")


class RejectAll:
    def __call__(self, premise, hypothesis): return {"entailment": 0.0}


def model(parts):
    return SIPGRL(parts, checkpoint=CK if os.path.exists(CK) else None)


def test_nli_scope_critical_skips_sentences_without_critical_spans():
    v_crit = FidelityVerifier(nli=RejectAll(), nli_scope="critical")
    v_all = FidelityVerifier(nli=RejectAll())                                  # default scope = all
    drift = ("She has mild chest pain.", "She has severe chest pain.")          # no number / drug / negation: invisible to scope=critical
    assert v_crit.check(*drift).ok and not v_all.check(*drift).ok
    assert not v_crit.check("Take warfarin 5 mg daily.", "Take warfarin 5 mg every day.").ok   # critical spans present: NLI runs


def test_bidirectional_mode_uses_both_directions():
    asym = lambda p, h: {"entailment": 0.9 if len(p) < len(h) else 0.1}
    assert FidelityVerifier(nli=asym).check("a short premise", "a considerably longer hypothesis text").ok
    assert not FidelityVerifier(nli=asym, nli_mode="bidirectional").check("a short premise", "a considerably longer hypothesis text").ok


def test_final_gate_failure_reverts_to_source():
    m = model(default_parts(nli=RejectAll()))
    for s in synth(15, 3):
        r = m.simplify(s).sentences[0]
        assert r["final_verdict"] == "fallback" and r["fallback"] is True and r["output"] == r["source"]


def test_every_emitted_sentence_passes_an_independent_check_even_under_injection():
    m = model(default_parts(realizer=AdversarialRealizer(rate=0.7, seed=1)))
    ver, n_fallback_or_edit = FidelityVerifier(), 0
    for s in synth(80, 4):
        r = m.simplify(s).sentences[0]
        assert r["final_verdict"] in ("pass", "fallback")
        assert ver.check(r["source"], r["output"], use_nli=False).ok, (r["source"], r["output"])
        n_fallback_or_edit += r["output"] != r["source"]
    assert n_fallback_or_edit > 0                       # the pipeline did edit; safety was not achieved by doing nothing


def test_edit_gate_rollbacks_are_recorded():
    m = model(default_parts(realizer=AdversarialRealizer(rate=1.0, seed=2)))
    rej = sum(len(m.simplify(s).sentences[0]["rejected"]) for s in synth(40, 5))
    assert rej > 0
