"""Train one PPO policy per seed under the CURRENT reward (default: S only) and sanity-check each on held-out sentences.

  python scripts/train_seeds.py --seeds 0,1,2,3,4 --iters 120 --out-pattern sipg_policy_v3_seed{seed}.pt            # synthetic (machinery only)
  python scripts/train_seeds.py --jsonl medeasi_train.jsonl --seeds 0,1,2,3,4 --out-pattern ...                      # real sentences
  python scripts/train_seeds.py ... --realizer hybrid --llm-model qwen2.5:7b-instruct --max-llm-calls 20000          # B4(a): LLM in the loop
With --realizer rules (default) the policy is trained against the rule realizer: any LLM mode at inference is then an
inference-time AUGMENTATION, not something the policy optimised for (state that in the paper)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from sipg_rl.pipeline import default_parts
from sipg_rl.env import EnvConfig
from sipg_rl.model import SIPGPolicy
from sipg_rl.ppo import PPOTrainer
from sipg_rl.baselines import LearnedPolicy, GreedyReadabilityPolicy, RandomPolicy
from sipg_rl.data import synth
from pilot import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0"); ap.add_argument("--iters", type=int, default=120); ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--n", type=int, default=450); ap.add_argument("--jsonl", default=None)
    ap.add_argument("--out-pattern", default="sipg_policy_v3_seed{seed}.pt")
    ap.add_argument("--reward", default="s_only"); ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--realizer", default="rules", choices=["rules", "hybrid"])
    ap.add_argument("--llm-url", default="http://localhost:11434/v1"); ap.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--llm-key", default=None); ap.add_argument("--max-llm-calls", type=int, default=20000)
    ap.add_argument("--cache", default=None, help="(unused placeholder) LLM answers are cached in memory per run")
    a = ap.parse_args()
    cfg = EnvConfig(reward_version=a.reward) if a.gamma is None else EnvConfig(reward_version=a.reward, gamma=a.gamma)
    realizer, llm = None, None
    if a.realizer == "hybrid":
        from sipg_rl.llm import OpenAICompatLLM
        from sipg_rl.realize import HybridRealizer, RuleRealizer
        llm = OpenAICompatLLM(a.llm_url, a.llm_model, a.llm_key)
        h = llm.health()
        if not h["reachable"] or h["model_available"] is False:
            sys.exit(f"LLM not usable: {h}")
        realizer = HybridRealizer(llm, RuleRealizer())
    parts = default_parts(cfg=cfg, realizer=realizer)
    if a.jsonl:
        rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
        k = lambda r, *n: next((r[x] for x in n if x in r and r[x]), None)
        sents = [s.text for r in rows for s in parts["nlp"](k(r, "Expert", "source") or "").sents if s.text.strip()][:a.n]
    else:
        sents = synth(a.n, 0)
    graphs = [parts["builder"].build_text(s) for s in sents]
    k = int(len(graphs) * 0.67)
    train, test = graphs[:k], graphs[k:]
    print(f"reward={a.reward} components={cfg.reward_components} gamma={cfg.gamma} realizer={a.realizer} | train {len(train)} test {len(test)} | mean nodes {np.mean([g.n for g in graphs]):.1f}")
    for sd in [int(x) for x in a.seeds.split(",")]:
        t0 = time.time(); torch.manual_seed(sd)
        pi = SIPGPolicy()
        PPOTrainer(pi, parts, seed=sd).train(train, iters=a.iters, episodes=a.episodes, log=lambda *_: None)
        pi.eval()
        path = a.out_pattern.format(seed=sd); torch.save(pi.state_dict(), path)
        ev = {nm: evaluate(parts, test, pol) for nm, pol in (("random", RandomPolicy(sd)), ("greedy", GreedyReadabilityPolicy()), ("sipg", LearnedPolicy(pi)))}
        m = lambda nm, key: np.mean([r[key] for r in ev[nm]])
        print(f"seed {sd}: saved {path} ({time.time()-t0:.0f}s) | " + " | ".join(
            f"{nm}: dR {m(nm,'dR'):+.3f} S {m(nm,'S'):.3f} edits {m(nm,'inv'):.1f}" for nm in ev), flush=True)
        if llm:
            print(f"   LLM calls {llm.stats['calls']} cache_hits {llm.stats['cache_hits']} errors {llm.stats['errors']}")
            if llm.stats["calls"] > a.max_llm_calls:
                sys.exit("LLM call budget exceeded")


if __name__ == "__main__":
    main()
