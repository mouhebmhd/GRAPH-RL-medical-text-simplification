import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from sipg_rl.inject import INJECTORS, inject_swap_drug, gate_report
from sipg_rl.verifier import FidelityVerifier
from sipg_rl.data import synth

EX = [dict(src=s, out=s, orig_out=s) for s in synth(200, 0)]
V = FidelityVerifier()


@pytest.mark.parametrize("name", list(INJECTORS))
def test_every_injector_fires(name):
    pert, n = INJECTORS[name]([dict(e) for e in EX])
    assert n > 0 and sum(p["injected"] for p in pert) == n


def test_swap_drug_regression():
    """The G4 symptom was `injected = 0`: must inject >= 9, the gate must catch every one, nothing leaks."""
    pert, n = inject_swap_drug([dict(e) for e in EX])
    assert n >= 9
    for p, e in zip(pert, EX):
        p["orig_out"] = e["orig_out"]
    r = gate_report(pert, V)
    assert r["caught"] == r["evaluated"] >= 9 and r["leaked"] == 0


def test_empty_vocabulary_raises():
    with pytest.raises(ValueError, match="empty vocabulary"):
        inject_swap_drug([dict(src="a", out="Take warfarin.")], drug_lexicon=[])


def test_zero_injections_raise_instead_of_passing_silently():
    with pytest.raises(RuntimeError, match="zero injections"):
        inject_swap_drug([dict(src="Rest well.", out="Rest well.")])


def test_antonym_is_the_documented_lexical_blind_spot():
    pert, n = INJECTORS["antonym"]([dict(e) for e in EX])
    for p, e in zip(pert, EX):
        p["orig_out"] = e["orig_out"]
    r = gate_report(pert, V)
    assert r["evaluated"] > 0 and r["leaked"] == r["evaluated"]       # the lexical gate cannot see meaning flips


def test_dose_x10_really_changes_the_value():
    from sipg_rl.normalize import number_multiset
    pert, _ = INJECTORS["dose_x10"]([dict(src="Take 0.25 mg daily.", out="Take 0.25 mg daily.")])
    assert number_multiset(pert[0]["out"]) != number_multiset("Take 0.25 mg daily.")   # 0.25 -> 2.5, not 0.250
    r = gate_report([dict(pert[0], orig_out="Take 0.25 mg daily.")], V)
    assert r["caught"] == 1
