import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sipg_rl.twobytwo import run_2x2
from sipg_rl.data import synth

CK = os.path.join(os.path.dirname(__file__), "..", "sipg_policy_v3.pt")


def test_orthogonality():
    c = run_2x2(synth(80, 11), ckpt=CK if os.path.exists(CK) else None, rate=0.6)
    on_s, off_s, on_w, off_w = c["strong|verifier_on"], c["strong|verifier_off"], c["weak|verifier_on"], c["weak|verifier_off"]
    # ON cells are equal in preservation: nothing unsafe gets through, whichever realizer is behind the gate
    assert on_s["unsafe_rate"] == on_w["unsafe_rate"] == 0.0
    for cat in on_s["preservation"]:
        assert on_s["preservation"][cat]["rate"] == on_w["preservation"][cat]["rate"]
    # OFF cells differ from ON for the strong realizer; the weak one is safe by construction so the gate has nothing to catch
    assert off_s["unsafe_rate"] > on_s["unsafe_rate"] and off_s["unsafe_rate"] > off_w["unsafe_rate"]
    assert off_w["unsafe_rate"] == 0.0 and off_w["edits"] <= off_s["edits"]
