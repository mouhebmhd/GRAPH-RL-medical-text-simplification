"""Pin: sum of step rewards == R(final) - R(source) to 1e-12 on clean episodes (no rejection / illegal / forced stop)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from sipg_rl.pipeline import default_parts
from sipg_rl.env import make_env, EnvConfig
from sipg_rl.baselines import RandomPolicy
from sipg_rl.data import synth


@pytest.mark.parametrize("version", ["s_only", "v1", "v2"])
def test_telescopes(version):
    parts = default_parts(cfg=EnvConfig(reward_version=version))
    rng, n_clean = RandomPolicy(0), 0
    for s in synth(60, 1):
        g = parts["builder"].build_text(s)
        env = make_env(parts); env.reset(g); tot = 0.0
        while not env.done:
            tot += env.step(rng.act(env))[0]
        if env.n_reject == 0 and env.steps < env.max_steps and env.trail[-1].get("event") != "final_reject":
            assert abs(tot - (env.final_reward - env.phi0)) < 1e-12
            n_clean += 1
    assert n_clean >= 30


def test_sim_is_logged_but_not_in_potential():
    parts = default_parts()
    g = parts["builder"].build_text("The hypertensive patient, who takes lisinopril 10 mg BID, reports no chest pain.")
    env = make_env(parts); env.reset(g)
    from sipg_rl.baselines import InvokeAllPolicy
    pol = InvokeAllPolicy(); pol.reset()
    while not env.done:
        env.step(pol.act(env))
    assert env.sim < 1.0 and 0.0 <= env.fid <= 1.0          # computed & logged
    assert abs(env.phi - env.S) < 1e-12                       # potential == S only
