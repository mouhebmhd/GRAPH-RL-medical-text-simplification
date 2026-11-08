"""P0.1 gate: the training objective must rank the human reference above a no-op. If this fails, STOP: every RL result
measures a broken objective. Runs on the toy fixture by default; set SIPG_MEDEASI_JSONL to run it on Med-EASi."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from medeasi_data import load_pairs, IS_REAL
from sipg_rl.scoring import FeatureSuitabilityScorer
from sipg_rl.stats import paired_bootstrap
from sipg_rl.env import EnvConfig


def test_default_potential_is_S_only():
    c = EnvConfig()
    assert tuple(c.reward_components) == ("S",) and c.reward_weights == {"S": 1.0}
    assert c.coverage_bonus == 0.0 and c.reject_penalty == 0.10


def test_reference_scores_above_copy_with_ci_excluding_zero():
    sc, pairs = FeatureSuitabilityScorer(), load_pairs()
    ref = [sc(y) for _, y in pairs]; copy = [sc(x) for x, _ in pairs]
    d, lo, hi, p = paired_bootstrap(ref, copy)
    print(f"[{'MED-EASI' if IS_REAL else 'TOY FIXTURE'}] n={len(pairs)} S(ref)={np.mean(ref):.3f} S(copy)={np.mean(copy):.3f} diff={d:+.3f} CI[{lo:+.3f},{hi:+.3f}]")
    assert lo > 0, "S(reference) must exceed S(copy) with a CI excluding 0 - STOP, objective is broken"
