"""Item 3: traversal instrumentation + H3 ablation. For several reward configs: train PPO, then report
coverage of non-trivial prompts, visit rate of condition nodes, and a full step log for sentence S1."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from sipg_rl.pipeline import default_parts
from sipg_rl.env import SimplifyEnv, EnvConfig, format_trail, INVOKE
from sipg_rl.model import SIPGPolicy
from sipg_rl.ppo import PPOTrainer
from sipg_rl.scoring import ConceptEncoder
from sipg_rl.baselines import GreedyReadabilityPolicy, LearnedPolicy
from sipg_rl.data import synth

S1 = "Discontinue lisinopril 10 mg if the patient develops hypertensive urgency, because the drug is contraindicated in renal impairment."
BASE = EnvConfig()
EVAL = default_parts(cfg=BASE)                      # common yardstick for every config
sents = synth(450, 0)
graphs = [EVAL["builder"].build_text(s) for s in sents]
train, test = graphs[:301], graphs[301:]
g_s1 = EVAL["builder"].build_text(S1)


def diag(policy, graphs):
    env = SimplifyEnv(EVAL["scorer"], EVAL["encoder"], EVAL["verifier"], EVAL["realizer"], BASE)
    cov, cond, relcl, R, S, sim = [], [], [], [], [], []
    for g in graphs:
        env.reset(g)
        nt = [j for j in range(g.n) if env.peek(j)[0] > 1e-6]          # prompts with a real effect at the start
        policy.reset()
        while not env.done:
            env.step(policy.act(env))
        if nt: cov.append(np.mean([env.invoked[j] for j in nt]))
        c = [n.id for n in g.nodes if n.relation == "condition"]
        if c: cond.append(np.mean([env.visited[j] for j in c]))
        rl = [n.id for n in g.nodes if n.kind == "RELCL"]
        if rl: relcl.append(np.mean([env.invoked[j] for j in rl]))
        R.append(env.final_reward - (BASE.alpha * env.S0 + BASE.beta)); S.append(env.S); sim.append(env.sim)
    return dict(dR=np.mean(R), S=np.mean(S), sim=np.mean(sim), coverage=np.mean(cov), cond_visit=np.mean(cond),
                relcl_fired=np.mean(relcl), n_cond=len(cond))


configs = {
    "baseline (g=.01)": (EnvConfig(), ConceptEncoder()),
    "low gamma (.002)": (EnvConfig(gamma=0.002, move_cost=0.0005), ConceptEncoder()),
    "coverage bonus .05": (EnvConfig(coverage_bonus=0.05), ConceptEncoder()),
    "binary Sim + reject .25": (EnvConfig(reject_penalty=0.25), ConceptEncoder(binary=True, bigrams=False)),
}
ONLY = sys.argv[1:] or list(configs)
rows = {"greedy-readability": diag(GreedyReadabilityPolicy(), test)}
trails = {}
for name in ONLY:
    cfg, enc = configs[name]
    t0 = time.time()
    parts = default_parts(cfg=cfg, encoder=enc, nlp=EVAL["nlp"])
    parts["builder"] = EVAL["builder"]
    torch.manual_seed(0)
    pi = SIPGPolicy()
    PPOTrainer(pi, parts, seed=0).train(train, iters=120, episodes=32, log=lambda *_: None)
    pi.eval()
    rows[name] = diag(LearnedPolicy(pi), test)
    env = SimplifyEnv(EVAL["scorer"], EVAL["encoder"], EVAL["verifier"], EVAL["realizer"], BASE)
    env.reset(g_s1)
    pol = LearnedPolicy(pi)
    while not env.done:
        env.step(pol.act(env))
    trails[name] = env
    print(f"[{name}] trained+evaluated in {time.time()-t0:.0f}s", flush=True)
    r = rows[name]
    print(f"RESULT {name:26s} dR {r['dR']:+.3f} S {r['S']:.3f} sim {r['sim']:.3f} coverage {r['coverage']*100:.0f}% "
          f"cond-visit {r['cond_visit']*100:.0f}% RELCL-fired {r['relcl_fired']*100:.0f}%", flush=True)
    print(f"--- S1 trail [{name}] ---\nOUT: {env.text}\n{format_trail(env, with_prompts=False)}\n", flush=True)

print(f"\n{'config':26s} {'dR':>7s} {'S':>6s} {'sim':>6s} {'coverage':>9s} {'cond-visit':>10s} {'RELCL fired':>11s}")
for k, r in rows.items():
    print(f"{k:26s} {r['dR']:+7.3f} {r['S']:6.3f} {r['sim']:6.3f} {r['coverage']*100:8.0f}% {r['cond_visit']*100:9.0f}% {r['relcl_fired']*100:10.0f}%")
print(f"(coverage = share of prompts with a real effect at t=0 that the policy fired; cond-visit over {rows['greedy-readability']['n_cond']} test sentences with a condition node)")

print("\n=== S1 graph ===")
for n in g_s1.nodes:
    print(f"  node {n.id} {n.kind:8s} rel={n.relation or '-':11s} parent={n.parent:2d} | {n.text[:70]}")
for name in [n for n in ("baseline (g=.01)", "binary Sim + reject .25") if n in trails]:
    e = trails[name]
    print(f"\n=== S1 traversal, config: {name} ===\nSRC: {e.src}\nOUT: {e.text}\n{format_trail(e, with_prompts=True)}")
