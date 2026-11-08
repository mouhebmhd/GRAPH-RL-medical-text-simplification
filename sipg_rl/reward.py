"""P0.1 - pluggable reward potential.

    phi(state) = sum_c weights[c] * SCORERS[c](state)          reward_t = phi_{t+1} - phi_t - costs        (telescopes)

Default components = ["S"] (suitability only). `Sim` is still COMPUTED and LOGGED every step (post-hoc filter / reported
column) but is not in phi unless you list it. Other components exist for ablations:
  Sim        legacy hashed-cosine fidelity (known broken: gold 0.514 vs no-op 1.000)  -> the 'reward' ablation
  fid        length-invariant fidelity F1 (fidelity.py)            fid_hinge  max(0, tau-F)/tau (use with a NEGATIVE weight)
  band       1 inside the reference band of S, decaying on both sides (penalises over-simplification)"""
from __future__ import annotations
import json, math, os
from dataclasses import dataclass, asdict

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reward_calibration.json")


@dataclass
class RewardCalibration:
    band_lo: float = 0.55
    band_hi: float = 0.85
    band_width: float = 0.15
    tau: float = 0.70
    source: str = "uncalibrated-default"

    @classmethod
    def from_references(cls, S_ref, F_ref, band_q=(10, 90), tau_q=10, source="references"):
        import numpy as np
        lo, hi = (float(np.percentile(S_ref, q)) for q in band_q)
        return cls(lo, hi, max(0.05, (hi - lo) / 2), float(np.percentile(F_ref, tau_q)), source)

    def band(self, S):
        d = max(self.band_lo - S, S - self.band_hi, 0.0)
        return math.exp(-((d / self.band_width) ** 2))

    def hinge(self, F):
        return max(0.0, self.tau - F) / self.tau

    def save(self, path=DEFAULT_PATH):
        json.dump(asdict(self), open(path, "w"), indent=1)

    @classmethod
    def load(cls, path=None):
        path = path or os.environ.get("SIPG_CALIBRATION") or DEFAULT_PATH
        return cls(**json.load(open(path))) if os.path.exists(path) else cls()


SCORERS = {
    "S": lambda c: c["S"],
    "Sim": lambda c: c["sim"],
    "fid": lambda c: c["fid"],
    "band": lambda c: c["cal"].band(c["S"]),
    "fid_hinge": lambda c: c["cal"].hinge(c["fid"]),
}

PRESETS = {  # name -> (components, weights)
    "s_only": (("S",), {"S": 1.0}),                                    # P0.1 default
    "v1": (("S", "Sim"), {"S": 1.0, "Sim": 0.5}),                      # legacy composite: the 'reward' ablation
    "v2": (("band", "fid_hinge"), {"band": 1.0, "fid_hinge": -0.5}),   # band + fidelity constraint
}


def potential(ctx: dict, cfg) -> float:
    """ctx: {S, sim, fid, cal}. Only the components listed in cfg.reward_components contribute."""
    return float(sum(cfg.reward_weights[n] * SCORERS[n](ctx) for n in cfg.reward_components))
