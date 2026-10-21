"""Train SIPG-RL policies with the corrected reward, several seeds, optional LLM realizer in the loop.

  python scripts/train_policy.py --seeds 0,1,2,3,4 --iters 60 --out runs/v2                    # rule realizer (fast)
  python scripts/train_policy.py --seeds 0,1,2 --realizer llm --llm-model qwen2.5:7b-instruct \
         --llm-cache runs/llm_cache.jsonl --iters 30 --out runs/v2_llm                          # LLM in the loop (slow, cached)

What changed vs the first pilot (see README, "Model fixes"):
  * reward  : ContentEncoder Sim (length-invariant, safety-weighted) + BandSuitabilityScorer S (target band, no saturation)
  * gamma   : 0.03 (edit cost now matters)   * coverage bonus: 0 (removed; fix the base reward instead)
  * realizer: rules-first RoutedRealizer; with --realizer llm the policy is trained against that SAME realizer (B4a)
  * seeds   : one checkpoint per seed, mean +- std and a t-interval across seeds (B5)
Every checkpoint gets a <ckpt>.meta.json recording what it was trained against, so a policy trained with rules but
deployed with an LLM is visibly an inference-time augmentation (B4b), not a claim that the policy optimised for it.
Synthetic template data is for MACHINERY checks only (templates favour greedy); do not draw scientific conclusions from it.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from sipg_rl.pipeline import default_parts
from sipg_rl.env import EnvConfig
from sipg_rl.model import SIPGPolicy
from sipg_rl.ppo import PPOTrainer
from sipg_rl.baselines import NoOpPolicy, RandomPolicy, InvokeAllPolicy, GreedyReadabilityPolicy, LearnedPolicy
from sipg_rl.data import synth, load_parallel_jsonl
from sipg_rl.realize import RoutedRealizer, RuleRealizer
from pilot import evaluate

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def across_seeds(x):
    """mean, std, 95% t-interval half-width across seeds."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan"), float("nan")
    sd = x.std(ddof=1)
    return float(x.mean()), float(sd), float(T975.get(len(x) - 1, 1.96) * sd / np.sqrt(len(x)))


def build_parts(a, nlp=None):
    cfg = EnvConfig(gamma=a.gamma, coverage_bonus=a.coverage_bonus,
                    reject_penalty=a.reject_penalty if a.reject_penalty is not None else (0.25 if a.realizer == "llm" else 0.10))
    realizer, llm = RuleRealizer(), None
    if a.realizer == "llm":
        from sipg_rl.llm import OpenAICompatLLM
        llm = OpenAICompatLLM(a.llm_url, a.llm_model, a.llm_key, timeout=a.llm_timeout, cache_path=a.llm_cache)
        h = llm.health()
        if not h["reachable"] or h["model_available"] is False:
            sys.exit(f"LLM not usable: {h}")
        realizer = RoutedRealizer(llm, RuleRealizer(), allow_llm_structure=a.allow_llm_structure)
    return default_parts(cfg=cfg, realizer=realizer, nlp=nlp), llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--train_frac", type=float, default=0.67)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--corpus", default=None, help="optional JSONL with a 'source' field")
    ap.add_argument("--gamma", type=float, default=0.03)
    ap.add_argument("--coverage_bonus", type=float, default=0.0)
    ap.add_argument("--reject_penalty", type=float, default=None)
    ap.add_argument("--realizer", choices=["rules", "llm"], default="rules")
    ap.add_argument("--allow-llm-structure", action="store_true", help="let the LLM touch CCOMP/ACL clauses (off by default)")
    ap.add_argument("--llm-url", default="http://localhost:11434/v1")
    ap.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--llm-key", default=None)
    ap.add_argument("--llm-timeout", type=int, default=120)
    ap.add_argument("--llm-cache", default=None, help="persistent JSONL cache: strongly recommended for --realizer llm")
    ap.add_argument("--out", default="runs/v2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    parts, llm = build_parts(a)
    sents = [p[0] for p in load_parallel_jsonl(a.corpus)][:a.n] if a.corpus else synth(a.n, 0)
    graphs = [parts["builder"].build_text(s) for s in sents]
    k = int(len(graphs) * a.train_frac)
    train, test = graphs[:k], graphs[k:]
    print(f"{len(graphs)} graphs | train {len(train)} test {len(test)} | mean nodes {np.mean([g.n for g in graphs]):.1f} | "
          f"dropped-by-24-cap: {sum(g.n_dropped > 0 for g in graphs)} sentences | realizer={a.realizer} gamma={a.gamma} bonus={a.coverage_bonus}")

    meta = dict(realizer_in_loop=a.realizer, llm_model=a.llm_model if llm else None, allow_llm_structure=a.allow_llm_structure,
                gamma=a.gamma, coverage_bonus=a.coverage_bonus, reject_penalty=parts["cfg"].reject_penalty,
                scorer=type(parts["scorer"]).__name__, encoder=type(parts["encoder"]).__name__, iters=a.iters,
                episodes=a.episodes, data="synthetic" if not a.corpus else a.corpus)
    base = [GreedyReadabilityPolicy(), InvokeAllPolicy(), NoOpPolicy()]
    base_res = {p.name: evaluate(parts, test, p) for p in base}
    rows = []
    for sd in seeds:
        t0 = time.time()
        torch.manual_seed(sd)
        policy = SIPGPolicy()
        PPOTrainer(policy, parts, seed=sd).train(train, iters=a.iters, episodes=a.episodes)
        policy.eval()
        ck = os.path.join(a.out, f"policy_seed{sd}.pt")
        torch.save(policy.state_dict(), ck)
        json.dump(dict(meta, seed=sd), open(ck + ".meta.json", "w"), indent=1)
        res = evaluate(parts, test, LearnedPolicy(policy, greedy=True))
        rnd = evaluate(parts, test, RandomPolicy(sd))
        m = lambda rr, key: float(np.mean([r[key] for r in rr]))
        g = base_res["greedy-readability"]
        row = dict(seed=sd, dR=m(res, "dR"), S=m(res, "S"), sim=m(res, "sim"), inv=m(res, "inv"), rej=m(res, "rej"),
                   preserved=m(res, "preserved"), dR_random=m(rnd, "dR"),
                   d_vs_greedy=m(res, "dR") - m(g, "dR"), inv_vs_greedy=m(res, "inv") - m(g, "inv"),
                   S_vs_greedy=m(res, "S") - m(g, "S"))
        rows.append(row)
        print(f"seed {sd}: dR {row['dR']:+.3f} (greedy {m(g,'dR'):+.3f}, random {row['dR_random']:+.3f}) S {row['S']:.3f} sim {row['sim']:.3f} "
              f"edits {row['inv']:.2f} (greedy {m(g,'inv'):.2f}) preserved {row['preserved']*100:.0f}%  [{time.time()-t0:.0f}s]", flush=True)

    print(f"\nacross {len(seeds)} seeds (mean ± std, 95% t-interval half-width):")
    summ = {}
    for key, label in [("dR", "dR (PPO)"), ("d_vs_greedy", "dR PPO - greedy"), ("inv_vs_greedy", "edits PPO - greedy"),
                       ("S_vs_greedy", "S PPO - greedy"), ("sim", "Sim"), ("preserved", "preserved")]:
        mu, sd_, hw = across_seeds([r[key] for r in rows])
        summ[key] = dict(mean=mu, std=sd_, ci_half=hw)
        print(f"  {label:20s} {mu:+.4f} ± {sd_:.4f}   [{mu-hw:+.4f}, {mu+hw:+.4f}]")
    if len(seeds) >= 2:
        lo, hi = summ["d_vs_greedy"]["mean"] - summ["d_vs_greedy"]["ci_half"], summ["d_vs_greedy"]["mean"] + summ["d_vs_greedy"]["ci_half"]
        print("  -> PPO vs greedy:", "distinguishable from greedy (CI excludes 0)" if lo > 0 or hi < 0 else "NOT distinguishable from greedy across seeds")
    if llm:
        print(f"LLM calls={llm.stats['calls']} cache_hits={llm.stats['cache_hits']} errors={llm.stats['errors']}")
    json.dump(dict(meta=meta, rows=rows, summary=summ, baselines={k: float(np.mean([r['dR'] for r in v])) for k, v in base_res.items()}),
              open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
