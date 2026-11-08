"""P2.3 - ablations, one per safety/architecture claim.   python scripts/run_ablations.py --ablate verifier|nli|graph|reward|all

  verifier : replace the verifier by an always-pass stub; inject unsafe edits mid-pipeline; compare leaked unsafe edits and
             critical-fact preservation of the FINAL outputs (full vs ablated).
  nli      : drift-suite recall with the lexical gate only vs lexical + NLI (needs --nli <hf model>; `--nli stub` is a PLUMBING
             check with a token-overlap stand-in and is not a measurement).
  graph    : train a FLAT policy (BiGRU over nodes in text order, no syntactic/discourse edges) and compare with the GAT policy.
  reward   : train with the legacy composite S + 0.5*Sim and evaluate under the S-only objective, to show the failure.
Corpus: synthetic template sentences by default (machinery check only - D2); --jsonl for real sentences.
`graph`/`reward` train per seed (~1 min each on CPU); the full-system reference is sipg_policy_v3_seed{seed}.pt."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from sipg_rl import metrics as M
from sipg_rl.pipeline import default_parts
from sipg_rl.env import EnvConfig, make_env
from sipg_rl.model import SIPGPolicy, FlatSequencePolicy
from sipg_rl.ppo import PPOTrainer
from sipg_rl.baselines import LearnedPolicy
from sipg_rl.graph import flatten_graph
from sipg_rl.data import synth
from sipg_rl.verifier import FidelityVerifier, VerifierResult
from sipg_rl.adversarial import AdversarialRealizer, DRIFT, FAITHFUL
from sipg_rl.stats import mean_std_across_seeds, mean_ci


class AlwaysPass:
    """Ablated verifier: never rejects (an edit gate and a final gate that do nothing)."""
    nli = None
    def check(self, src, out, use_nli=True): return VerifierResult(True, [])


def corpus(a, parts):
    if a.jsonl:
        rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
        k = lambda r, *n: next((r[x] for x in n if x in r and r[x]), None)
        sents = [s.text for r in rows for s in parts["nlp"](k(r, "Expert", "source") or "").sents if s.text.strip()][:a.n]
    else:
        sents = synth(a.n, 0)
    gs = [parts["builder"].build_text(s) for s in sents]
    k = int(len(gs) * 0.67)
    return gs[:k], gs[k:]


def run_policy(parts, graphs, policy, flat=False):
    rows = []
    for g in graphs:
        g = flatten_graph(g) if flat else g
        env = make_env(parts); env.reset(g); pol = LearnedPolicy(policy)
        while not env.done:
            env.step(pol.act(env))
        crit = M.critical_preservation_by_category(env.src, env.text)
        rows.append(dict(S=env.S, S0=env.S0, dS=env.S - env.S0, edits=env.n_changed, rejected=env.n_reject, sim=env.sim,
                         R=env.final_reward - env.phi0, verifier_ok=FidelityVerifier().check(env.src, env.text, use_nli=False).ok, **{f"crit_{k}": v for k, v in crit.items()}))
    return rows


def summ(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
    return out


def abl_verifier(a):
    print("\n=== ABLATION: verifier (always-pass) under injected unsafe edits ===")
    pol = SIPGPolicy(); pol.load_state_dict(torch.load(a.ckpt)); pol.eval()
    res = {}
    for name, ver in (("full verifier", None), ("verifier ablated", AlwaysPass())):
        adv = AdversarialRealizer(rate=a.inject_rate, seed=0)
        parts = default_parts(realizer=adv)
        if ver: parts["verifier"] = ver
        _, test = corpus(a, parts)
        inj = leaked = 0; rows = []
        for g in test:
            env = make_env(parts); env.reset(g); p = LearnedPolicy(pol)
            while not env.done:
                act = p.act(env); before = env.n_reject; env.step(act)
                if act == 0 and adv.last_injected:
                    inj += 1; leaked += int(env.n_reject == before)
            crit = M.critical_preservation_by_category(env.src, env.text)
            rows.append(dict(unsafe=int(not FidelityVerifier().check(env.src, env.text, use_nli=False).ok), **{f"crit_{k}": v for k, v in crit.items()}))
        res[name] = (inj, leaked, rows)
    print(f"{'condition':18}{'injected':>9}{'leaked':>8}{'unsafe outputs':>16}" + "".join(f"{c:>13}" for c in M.CRITICAL_CATEGORIES[:5]))
    for name, (inj, leaked, rows) in res.items():
        s = summ(rows, ["unsafe"] + [f"crit_{c}" for c in M.CRITICAL_CATEGORIES[:5]])
        print(f"{name:18}{inj:9d}{leaked:8d}{100*s['unsafe']:15.1f}%" + "".join(f"{('n/a' if s['crit_'+c] is None else '%.1f%%' % (100*s['crit_'+c])):>13}" for c in M.CRITICAL_CATEGORIES[:5]))
    return res


def abl_nli(a):
    print("\n=== ABLATION: NLI adapter (drift-suite recall, lexical gate vs lexical + NLI) ===")
    lex = FidelityVerifier()
    Ld = np.array([not lex.check(s, o, use_nli=False).ok for _, s, o in DRIFT]); Lf = np.array([not lex.check(s, o, use_nli=False).ok for s, o in FAITHFUL])
    print(f"NLI OFF: drift recall {100*Ld.mean():.0f}% ({Ld.sum()}/{len(Ld)}), false-positive {100*Lf.mean():.0f}% ({Lf.sum()}/{len(Lf)})")
    if not a.nli:
        print("NLI ON : not run - pass --nli microsoft/deberta-large-mnli (needs transformers + network) for the real row."); return
    if a.nli == "stub":
        import re
        W = lambda t: set(re.findall(r"[a-z]+", t.lower()))
        nli = lambda p, h: {"entailment": len(W(p) & W(h)) / max(1, len(W(h)))}
        print("!! STUB NLI (token overlap): plumbing check only, NOT a measurement of NLI recall !!")
    else:
        from sipg_rl.verifier import HFNLI
        nli = HFNLI(a.nli)
    print(f"{'gate':34}{'thr':>5}{'drift recall':>14}{'faithful FPR':>14}")
    print(f"{'lexical only (NLI off)':34}{'':>5}{100*Ld.mean():13.0f}%{100*Lf.mean():13.0f}%")
    for mode in ("forward", "bidirectional"):
        for t in (0.3, 0.5, 0.7, 0.9):
            v = FidelityVerifier(nli=nli, nli_threshold=t, nli_mode=mode)          # the SAME gate the pipeline uses
            rd = np.array([not v.check(s, o).ok for _, s, o in DRIFT]); rf = np.array([not v.check(s, o).ok for s, o in FAITHFUL])
            print(f"{'lexical + NLI ' + mode:34}{t:5.2f}{100*rd.mean():13.0f}%{100*rf.mean():13.0f}%")


def abl_graph(a):
    print("\n=== ABLATION: graph (flat BiGRU over nodes in text order vs GAT over the induced graph) ===")
    parts = default_parts(); train, test = corpus(a, parts)
    flat_train = [flatten_graph(g) for g in train]
    rows = {"GAT (full)": [], "flat sequence": []}
    for sd in a.seeds:
        full = SIPGPolicy(); full.load_state_dict(torch.load(a.ckpt_pattern.format(seed=sd))); full.eval()
        rows["GAT (full)"].append(summ(run_policy(parts, test, full), ["S", "edits", "R"]))
        t0 = time.time(); torch.manual_seed(sd)
        flat = FlatSequencePolicy(); PPOTrainer(flat, parts, seed=sd).train(flat_train, iters=a.iters, episodes=32, log=lambda *_: None); flat.eval()
        rows["flat sequence"].append(summ(run_policy(parts, test, flat, flat=True), ["S", "edits", "R"]))
        print(f"  seed {sd}: flat trained in {time.time()-t0:.0f}s")
    table(rows)
    return rows


def abl_reward(a):
    print("\n=== ABLATION: reward (legacy S + 0.5*Sim, evaluated under the S-only objective) ===")
    parts_s = default_parts(); parts_v1 = default_parts(cfg=EnvConfig(reward_version="v1"))
    train, test = corpus(a, parts_s)
    rows = {"S only (full)": [], "S + 0.5 Sim (legacy)": []}
    for sd in a.seeds:
        full = SIPGPolicy(); full.load_state_dict(torch.load(a.ckpt_pattern.format(seed=sd))); full.eval()
        rows["S only (full)"].append(summ(run_policy(parts_s, test, full), ["S", "edits", "R", "sim"]))
        t0 = time.time(); torch.manual_seed(sd)
        leg = SIPGPolicy(); PPOTrainer(leg, parts_v1, seed=sd).train(train, iters=a.iters, episodes=32, log=lambda *_: None); leg.eval()
        rows["S + 0.5 Sim (legacy)"].append(summ(run_policy(parts_s, test, leg), ["S", "edits", "R", "sim"]))
        print(f"  seed {sd}: legacy-reward policy trained in {time.time()-t0:.0f}s")
    table(rows)
    return rows


def table(rows):
    keys = [k for k in next(iter(rows.values()))[0] if next(iter(rows.values()))[0][k] is not None]
    print(f"{'condition':26}" + "".join(f"{k:>22}" for k in keys))
    for name, per_seed in rows.items():
        cells = []
        for k in keys:
            m, sd, _ = mean_std_across_seeds([r[k] for r in per_seed])
            cells.append(f"{m:.3f}" + (f" +- {sd:.3f}" if sd == sd else ""))
        print(f"{name:26}" + "".join(f"{c:>22}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", default="all", choices=["verifier", "nli", "graph", "reward", "all"])
    ap.add_argument("--seeds", default="0"); ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--ckpt", default="sipg_policy_v3.pt"); ap.add_argument("--ckpt-pattern", default="sipg_policy_v3_seed{seed}.pt")
    ap.add_argument("--jsonl", default=None); ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--inject-rate", type=float, default=0.5); ap.add_argument("--nli", default=None)
    a = ap.parse_args(); a.seeds = [int(s) for s in a.seeds.split(",")]
    todo = ["verifier", "nli", "graph", "reward"] if a.ablate == "all" else [a.ablate]
    for t in todo:
        {"verifier": abl_verifier, "nli": abl_nli, "graph": abl_graph, "reward": abl_reward}[t](a)


if __name__ == "__main__":
    main()
