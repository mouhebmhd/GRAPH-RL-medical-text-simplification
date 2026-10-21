"""Phase-1 pilot + kill criterion: do learned traversals beat random and greedy traversals on held-out
graphs by more than noise? Trains PPO on a train split and compares all policies on the test split."""
import argparse, json, sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from sipg_rl.pipeline import default_parts
from sipg_rl.env import SimplifyEnv, EnvConfig
from sipg_rl.model import SIPGPolicy
from sipg_rl.ppo import PPOTrainer
from sipg_rl.baselines import (NoOpPolicy, RandomPolicy, InvokeAllPolicy, GreedyReadabilityPolicy,
                               LearnedPolicy, run_episode)
from sipg_rl.data import synth, load_parallel_jsonl


def evaluate(parts, graphs, policy):
    env = SimplifyEnv(parts["scorer"], parts["encoder"], parts["verifier"], parts["realizer"], parts["cfg"])
    rows = []
    for g in graphs:
        run_episode(env, g, policy)
        ok = parts["verifier"].check(env.src, env.text, use_nli=False).ok
        rows.append(dict(R=env.final_reward, dR=env.final_reward - (env.cfg.alpha * env.S0 + env.cfg.beta),
                         S=env.S, S0=env.S0, sim=env.sim, inv=env.n_inv, mv=env.n_move, rej=env.n_reject,
                         changed=env.n_changed, preserved=ok, out=env.text, src=env.src))
    return rows


def paired_bootstrap(a, b, n=5000, seed=0):
    d = np.array(a) - np.array(b)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), float((means <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--train_frac", type=float, default=0.67)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default=None, help="optional JSONL with a 'source' field (Med-EASi etc.)")
    ap.add_argument("--coverage_bonus", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.03)
    ap.add_argument("--out", default="pilot_report.json")
    ap.add_argument("--ckpt", default="sipg_policy.pt")
    a = ap.parse_args()

    parts = default_parts(cfg=EnvConfig(gamma=a.gamma, coverage_bonus=a.coverage_bonus))
    sents = [p[0] for p in load_parallel_jsonl(a.corpus)][:a.n] if a.corpus else synth(a.n, a.seed)
    t0 = time.time()
    graphs = [parts["builder"].build_text(s) for s in sents]
    k = int(len(graphs) * a.train_frac)
    train, test = graphs[:k], graphs[k:]
    print(f"graphs built: {len(graphs)} in {time.time()-t0:.1f}s | train {len(train)} test {len(test)} | "
          f"mean nodes {np.mean([g.n for g in graphs]):.1f}")

    torch.manual_seed(a.seed)
    policy = SIPGPolicy()
    PPOTrainer(policy, parts, seed=a.seed).train(train, iters=a.iters, episodes=a.episodes)
    policy.eval()
    torch.save(policy.state_dict(), a.ckpt)

    pols = [NoOpPolicy(), RandomPolicy(a.seed), InvokeAllPolicy(), GreedyReadabilityPolicy(),
            LearnedPolicy(policy, greedy=True)]
    res = {p.name: evaluate(parts, test, p) for p in pols}
    print(f"\n{'policy':22s} {'dR':>7s} {'S':>6s} {'sim':>6s} {'inv':>5s} {'mv':>5s} {'rej':>5s} {'preserved':>9s}")
    for name, rows in res.items():
        m = lambda key: np.mean([r[key] for r in rows])
        print(f"{name:22s} {m('dR'):+7.3f} {m('S'):6.3f} {m('sim'):6.3f} {m('inv'):5.1f} {m('mv'):5.1f} "
              f"{m('rej'):5.2f} {m('preserved')*100:8.1f}%")

    learned = res["SIPG-RL (PPO)"]
    print("\nKill-criterion check (paired bootstrap on per-sentence R, learned minus baseline):")
    verdict = True
    for base in ("random traversal", "greedy-readability"):
        d, lo, hi, p = paired_bootstrap([r["R"] for r in learned], [r["R"] for r in res[base]])
        beats = lo > 0
        verdict &= beats
        print(f"  vs {base:20s}: diff {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p~{p:.3f}  -> "
              f"{'learned policy wins' if beats else 'NOT distinguishable from / worse than baseline'}")
    print("\nVERDICT:", "PASS - structured action space adds value over random/greedy on this pilot."
          if verdict else "KILL-CRITERION TRIGGERED on this pilot - learned traversal does not beat both baselines.")
    json.dump({n: [{k: v for k, v in r.items()} for r in rows] for n, rows in res.items()},
              open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
