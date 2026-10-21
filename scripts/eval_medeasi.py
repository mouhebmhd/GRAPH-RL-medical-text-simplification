"""Evaluate SIPG-RL on Med-EASi (expert -> layman sentence simplification).

    pip install datasets
    python scripts/eval_medeasi.py --split test --limit 200 --systems copy,rules,llm \
           --llm-model qwen2.5:7b-instruct --out eval_medeasi

Fields used: Expert (source), Simple (reference); optional: expert_terms / layman_terms (term retention),
expert_fk_grade / layman_fk_grade (cross-check of our readability formula). Offline: --jsonl file with the same keys.

Systems
  copy       source unchanged (floor for SARI/readability; ceiling for fidelity)
  rules      trained policy + rule realizer
  rules_all  ... + apply every node's prompt after the policy stops
  llm        trained policy + Ollama realizer (+ sweep of term nodes)
  llm_all    ... + apply every node's prompt
  reference  the human Simple text (reported for readability / verifier strictness only; not scored by SARI)
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sipg_rl import metrics as M

SYSTEMS = ["copy", "rules", "rules_all", "llm", "llm_all", "reference"]


# ----------------------------------------------------------------------------------------------- data
def load_rows(a, split=None):
    split = split or a.split
    if a.jsonl and split == a.split:
        rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset(a.dataset)
        if split not in ds:
            sys.exit(f"split '{split}' not in {list(ds.keys())}")
        rows = [dict(r) for r in ds[split]]
    key = lambda r, *names: next((r[n] for n in names if n in r and r[n]), None)
    out = []
    for i, r in enumerate(rows):
        x, y = key(r, "Expert", "source"), key(r, "Simple", "target")
        if not x or not y or len(x.split()) < a.min_words:
            continue
        out.append(dict(id=int(r.get("idx", i)), src=x.strip(), ref=y.strip(), r=r))
    if a.shuffle:
        np.random.default_rng(a.seed).shuffle(out)
    return out[:a.limit] if a.limit else out


# ----------------------------------------------------------------------------------------------- systems
def build(a, names):
    import torch
    from sipg_rl.pipeline import SIPGRL, default_parts
    from sipg_rl.realize import RoutedRealizer, RuleRealizer
    from sipg_rl.llm import OpenAICompatLLM
    ck = a.ckpt if a.ckpt and os.path.exists(a.ckpt) else None
    if a.ckpt and not ck:
        print(f"WARNING: checkpoint {a.ckpt} not found; policy is UNTRAINED", file=sys.stderr)
    base = SIPGRL(checkpoint=ck)
    if a.train_ppo_iters:
        base = train_on_medeasi(a, base)
    nlp = base.parts["nlp"]
    systems, llm = {}, None
    if "rules" in names or "rules_all" in names:
        systems["rules"] = (base, False); systems["rules_all"] = (base, True)
    if "llm" in names or "llm_all" in names:
        llm = OpenAICompatLLM(a.llm_url, a.llm_model, a.llm_key, timeout=a.llm_timeout)
        h = llm.health()
        if not h["reachable"] or h["model_available"] is False:
            sys.exit(f"LLM not usable: {h}")
        parts = default_parts(nlp=nlp, realizer=RoutedRealizer(llm, RuleRealizer()))
        m = SIPGRL(parts, policy=base.policy, sweep=True)
        systems["llm"] = (m, False); systems["llm_all"] = (m, True)
    return systems, llm, base


def train_on_medeasi(a, base):
    """OPTIONAL: PPO on Expert sentences of the TRAIN split (rule realizer), then evaluate on the held-out split."""
    from sipg_rl.ppo import PPOTrainer
    rows = load_rows(a, "train")[:a.train_limit]
    p, graphs = base.parts, []
    for r in rows:
        for s in p["nlp"](r["src"]).sents:
            if s.text.strip():
                graphs.append(p["builder"].build(s))
    print(f"PPO on {len(graphs)} train sentences (mean nodes {np.mean([g.n for g in graphs]):.1f}), {a.train_ppo_iters} iters")
    base.policy.train()
    PPOTrainer(base.policy, p, seed=a.seed).train(graphs, iters=a.train_ppo_iters)
    base.policy.eval()
    path = os.path.join(a.out, "policy_medeasi.pt"); os.makedirs(a.out, exist_ok=True); base.save(path)
    print("saved", path)
    return base


def run_system(name, systems, ex):
    if name == "copy":
        return ex["src"], dict(n_edits=0, n_rejected=0, n_invokes=0, final_fallback=0)
    if name == "reference":
        return ex["ref"], dict(n_edits=0, n_rejected=0, n_invokes=0, final_fallback=0)
    m, all_ = systems[name]
    res = m.simplify(ex["src"], apply_all=all_)
    return res.text, dict(n_edits=sum(len(s["operations"]) for s in res.sentences),
                          n_rejected=sum(len(s["rejected"]) for s in res.sentences),
                          n_invokes=sum(s["n_invokes"] for s in res.sentences),
                          final_fallback=int(any(s["final_rejected"] for s in res.sentences)))


# ----------------------------------------------------------------------------------------------- metrics per example
def score(ex, out, info, parts):
    s, r = ex["src"], ex["ref"]
    sc = M.sari(s, out, r)
    row = dict(id=ex["id"], out=out, **info, **sc, rougeL=M.rouge_l(out, r),
               fkgl=M.fkgl(out), ari=M.ari(out), d_fkgl=M.fkgl(out) - M.fkgl(s),
               S=parts["scorer"](out), sim=parts["encoder"].cos(parts["encoder"].encode(out), parts["encoder"].encode(s)),
               changed=int(out.strip() != s.strip()), len_ratio=len(out.split()) / max(1, len(s.split())),
               nums_kept=int(M.numbers_kept(s, out)), negs_kept=int(M.negations_kept(s, out)),
               verifier_ok=int(parts["verifier"].check(s, out, use_nli=False).ok))
    et, lt = M.parse_terms(ex["r"].get("expert_terms")), M.parse_terms(ex["r"].get("layman_terms"))
    row["expert_term_kept"] = M.term_retention(et, out)      # lower = more jargon removed
    row["layman_term_hit"] = M.term_retention(lt, out)       # higher = closer to human wording
    return row


COLS = [("SARI", "sari", 1), ("keep", "keep", 1), ("del", "delete", 1), ("add", "add", 1), ("ROUGE-L", "rougeL", 1),
        ("FKGL", "fkgl", 2), ("dFKGL", "d_fkgl", 2), ("ARI", "ari", 2), ("S", "S", 3), ("Sim", "sim", 3),
        ("changed%", "changed", 100), ("len×", "len_ratio", 2), ("edits", "n_edits", 2), ("rejected", "n_rejected", 2),
        ("fallback%", "final_fallback", 100), ("nums%", "nums_kept", 100), ("negs%", "negs_kept", 100),
        ("verif%", "verifier_ok", 100), ("jargon kept%", "expert_term_kept", 100), ("layman hit%", "layman_term_hit", 100)]


def fmt(v, d):
    return "  n/a" if v is None or v != v else (f"{v*100:5.1f}" if d == 100 else f"{v:6.{d}f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cbasu/Med-EASi")
    ap.add_argument("--jsonl", default=None, help="offline file with Expert/Simple (or source/target) keys")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200, help="0 = all")
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--systems", default="copy,rules,llm,reference")
    ap.add_argument("--ckpt", default="sipg_policy_v2.pt")
    ap.add_argument("--llm-url", default="http://localhost:11434/v1")
    ap.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--llm-key", default=None)
    ap.add_argument("--llm-timeout", type=int, default=120)
    ap.add_argument("--train-ppo-iters", type=int, default=0, help="optional PPO on the train split first")
    ap.add_argument("--train-limit", type=int, default=400)
    ap.add_argument("--out", default="eval_medeasi")
    ap.add_argument("--resume", action="store_true", help="skip examples already in <out>/<system>.jsonl")
    a = ap.parse_args()
    names = [n.strip() for n in a.systems.split(",") if n.strip()]
    bad = [n for n in names if n not in SYSTEMS]
    if bad:
        sys.exit(f"unknown systems {bad}; choose from {SYSTEMS}")
    os.makedirs(a.out, exist_ok=True)

    data = load_rows(a)
    print(f"{len(data)} pairs from {a.jsonl or a.dataset}:{a.split}")
    systems, llm, base = build(a, names)
    parts = base.parts

    # cross-check our FKGL against the dataset's precomputed grade columns
    chk = [(M.fkgl(e["src"]), e["r"]["expert_fk_grade"]) for e in data if isinstance(e["r"].get("expert_fk_grade"), (int, float))]
    if len(chk) > 10:
        print(f"FKGL cross-check vs dataset column expert_fk_grade: r = {np.corrcoef(*zip(*chk))[0,1]:.3f}, "
              f"mean abs diff = {np.mean([abs(x-y) for x,y in chk]):.2f} grades (n={len(chk)})")

    rows = {}
    for name in names:
        path, done = os.path.join(a.out, f"{name}.jsonl"), {}
        if a.resume and os.path.exists(path):
            done = {r["id"]: r for r in map(json.loads, open(path))}
        f = open(path, "a" if a.resume else "w")
        rows[name], t0 = [], time.time()
        for k, ex in enumerate(data):
            if ex["id"] in done:
                rows[name].append(done[ex["id"]]); continue
            t1 = time.time()
            out, info = run_system(name, systems, ex)
            r = score(ex, out, info, parts); r["sec"] = time.time() - t1; r["src"] = ex["src"]; r["ref"] = ex["ref"]
            rows[name].append(r); f.write(json.dumps(r) + "\n"); f.flush()
            if (k + 1) % 10 == 0:
                print(f"  [{name}] {k+1}/{len(data)}  {(time.time()-t0)/(k+1):.1f}s/ex", flush=True)
        f.close()

    # ---- table
    print("\n" + f"{'system':10}" + "".join(f"{c[0]:>13}" for c in COLS))
    summary = {}
    for name in names:
        summary[name] = {}
        line = f"{name:10}"
        for label, key, d in COLS:
            vals = [r[key] for r in rows[name] if r.get(key) is not None]
            if name == "reference" and key in ("sari", "keep", "delete", "add", "rougeL"):
                m = lo = hi = float("nan")
            else:
                m, lo, hi = M.mean_ci(vals)
            summary[name][key] = dict(mean=m, lo=lo, hi=hi, n=len(vals))
            line += f"{fmt(m, d):>13}"
        print(line)
    if llm:
        print(f"\nLLM: {llm.model}  calls={llm.stats['calls']}  cache_hits={llm.stats['cache_hits']}  errors={llm.stats['errors']}"
              + (f"  last_error={llm.stats['last_error']}" if llm.stats["errors"] else ""))

    # ---- paired comparisons on SARI (same examples)
    comps = {}
    print("\nPaired bootstrap, SARI difference (95% CI):")
    for x in names:
        for y in names:
            if x < y and x not in ("reference",) and y not in ("reference",):
                ids = [r["id"] for r in rows[x]]
                d, lo, hi = M.paired_diff_ci([r["sari"] for r in rows[x]], [r["sari"] for r in rows[y]])
                comps[f"{x}-{y}"] = dict(diff=d, lo=lo, hi=hi)
                print(f"  {x:9} - {y:9}: {d:+6.2f}  [{lo:+.2f}, {hi:+.2f}]" + ("   (CI excludes 0)" if lo > 0 or hi < 0 else ""))
    json.dump(dict(args=vars(a), n=len(data), summary=summary, paired_sari=comps), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"\nper-example rows: {a.out}/<system>.jsonl   summary: {a.out}/summary.json")


if __name__ == "__main__":
    main()
