import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sipg_rl.twobytwo import run_cell
from sipg_rl.verifier import FidelityVerifier
from sipg_rl.data import synth

CK = os.path.join(os.path.dirname(__file__), "..", "sipg_policy_v3.pt")
CK = CK if os.path.exists(CK) else None


def test_verifier_off_switch_always_passes():
    assert FidelityVerifier(enabled=False).check("Take 5 mg. Do not smoke.", "Take 500 mg. Smoke.").ok
    assert not FidelityVerifier().check("Take 5 mg. Do not smoke.", "Take 500 mg. Smoke.").ok


def test_disabling_the_verifier_lets_at_least_one_injection_leak():
    S = synth(60, 7)
    on = run_cell(S, "strong", True, ckpt=CK, rate=0.6)
    off = run_cell(S, "strong", False, ckpt=CK, rate=0.6)
    assert sum(r["unsafe"] for r in on) == 0
    assert sum(r["unsafe"] for r in off) >= 1                      # without the gate, preservation breaks
