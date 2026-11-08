"""P1/P2 evaluation orchestrator: 8 systems, diagnostics, CIs, seeds, audit trail.

  python scripts/run_eval.py --jsonl medeasi_test.jsonl --limit 200 --seeds 0,1,2,3,4 \
        --ckpt-pattern "sipg_policy_v3_seed{seed}.pt" --llm-model qwen2.5:7b-instruct --out eval_out

Systems:  copy | random | invoke_all | greedy (graph traversals + rule realizer, no learning)
          rules  = learned policy + rule realizer        llm    = learned policy + free-form LLM realizer (+ sweep)
          hybrid = invoke_all + HybridRealizer (rules lexical, constrained LLM fallback)
          sipg   = learned policy + HybridRealizer       reference = the human Simple text (diagnostics only)
Without a reachable LLM, hybrid/sipg run RULES-ONLY (flagged) and llm is skipped. Every headline number gets a 95% CI;
comparisons vs --compare-to use a paired bootstrap with Holm-Bonferroni across the metric family.
Learned/stochastic systems (random, rules, llm, sipg) are run once per seed; deterministic ones once."""
import argparse, csv, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sipg_rl import metrics as M
from sipg_rl.stats import mean_ci, paired_bootstrap, holm, mean_std_across_seeds
from sipg_rl.trail import build_trail_record
import eval_medeasi as E

ALL = ["copy", "random", "invoke_all", "greedy", "rules", "llm", "hybrid", "sipg", "reference"]
STOCHASTIC = {"random", "rules", "llm", "sipg"}
HEAD = [("SARI", "sari"), ("ROUGE-L", "rougeL"), ("FKGL", "fkgl"), ("S", "S"), ("fid F1", "fid"), ("layman hit", "layman_term_hit"),
        ("jargon kept", "expert_term_kept"), ("nums", "nums_kept"), ("negs", "negs_kept"), ("verif", "verifier_ok"),
        ("changed", "changed"), ("edits", "n_edits"), ("rejected", "n_rejected"), ("fallback", "final_fallback"),
        ("in FKGL band", "in_band"), ("len x", "len_ratio")]
PCT = {"layman_term_hit", "expert_term_kept", "nums_kept", "negs_kept", "verifier_ok", "changed", "final_fallback", "in_band"}


def build_systems(a, base_parts, policy, names, seed, llm):
    """-> {name: (SIPGRL, apply_all)} for one seed."""
    from sipg_rl.pipeline import SIPGRL, default_parts
    from sipg_rl.realize import CallableRealizer, RuleRealizer, HybridRealizer
    nlp = base_parts["nlp"]
    mk = lambda realizer: default_parts(nlp=nlp, realizer=realizer, nli=base_parts["nli"], fidelity=base_parts["fidelity"])
    rules = mk(RuleRealizer())
    hyb = HybridRealizer(llm if llm else (lambda *x, **k: ""), RuleRealizer(), lexical_llm=bool(llm), structural_llm=bool(llm))
    S = {}
    if "random" in names: S["random"] = SIPGRL(rules, policy=policy, traversal="random", seed=seed)
    if "invoke_all" in names: S["invoke_all"] = SIPGRL(rules, policy=policy, traversal="invoke_all")
    if "greedy" in names: S["greedy"] = SIPGRL(rules, policy=policy, traversal="greedy")
    if "rules" in names: S["rules"] = SIPGRL(rules, policy=policy)
    if "llm" in names and llm: S["llm"] = SIPGRL(mk(CallableRealizer(llm, RuleRealizer())), policy=policy, sweep=True)
    if "hybrid" in names: S["hybrid"] = SIPGRL(mk(hyb), policy=policy, traversal="invoke_all")
    if "sipg" in names: S["sipg"] = SIPGRL(mk(hyb), policy=policy)
    return S


def evaluate(a, rows, parts, systems, names, seed, graphs):
    out = {}
    for name in names:
        res_rows, trails = [], []
        for ex in rows:
            if name in ("copy", "reference"):
                text, res = (ex["src"] if name == "copy" else ex["ref"]), None
                info = dict(n_edits=0, n_rejected=0, n_invokes=0, final_fallback=0)
            elif name in systems:
                res = systems[name].simplify(ex["src"])
                text = res.text
                info = dict(n_edits=sum(len(s["operations"]) for s in res.sentences), n_rejected=sum(len(s["rejected"]) for s in res.sentences),
                            n_invokes=sum(s["n_invokes"] for s in res.sentences), final_fallback=int(any(s["final_rejected"] for s in res.sentences)))
            else:
                continue
            row = E.score(ex, text, info, parts)
            s_ = ex["src"]
            row["fid"] = parts["fidelity"].score(s_, text)["f1"]
            row["in_band"] = int(a.band[0] <= row["fkgl"] <= a.band[1])
            nl = M.nli_entailment(s_, text, parts["nli"])
            row["nli_min"], row["nli_drift"] = (nl["min"], int(M.nli_disagreement(s_, text, parts["nli"]))) if nl else (None, None)
            row["layman_by_node"] = M.layman_hit_by_node_type(s_, text, graphs[ex["id"]])
            row["critical"] = M.critical_preservation_by_category(s_, text)
            row["jargon_cat"] = M.jargon_kept_by_category(s_, text)
            row["edit_types"] = M.edit_type_distribution(s_, text)
            row["src"], row["ref"] = s_, ex["ref"]
            res_rows.append(row)
            trails.append(build_trail_record(ex["id"], name, s_, text, res))
        out[name] = (res_rows, trails)
    return out


def cell(dicts, k):
    """Per-category rate with a bootstrap 95% CI (Task 21)."""
    v = [float(d[k]) for d in dicts if d.get(k) is not None]
    if not v:
        return "n/a"
    m, lo, hi = mean_ci(v)
    return f"{100*m:.1f} [{100*lo:.1f},{100*hi:.1f}] (n={len(v)})"


def col(rows, key):
    return [r.get(key) for r in rows]


def fmtci(v, lo, hi, pct):
    if v != v: return "n/a"
    k = 100 if pct else 1
    return f"{v*k:.1f} [{lo*k:.1f},{hi*k:.1f}]" if pct else f"{v:.3f} [{lo:.3f},{hi:.3f}]" if abs(v) < 5 else f"{v:.2f} [{lo:.2f},{hi:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cbasu/Med-EASi"); ap.add_argument("--jsonl"); ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200); ap.add_argument("--min-words", type=int, default=5); ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seeds", default="0")
    ap.add_argument("--systems", default=",".join(ALL)); ap.add_argument("--compare-to", default="greedy")
    ap.add_argument("--ckpt", default="sipg_policy_v3.pt"); ap.add_argument("--ckpt-pattern", default=None)
    ap.add_argument("--llm-url", default="http://localhost:11434/v1"); ap.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--llm-key", default=None); ap.add_argument("--llm-timeout", type=int, default=120); ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--llm-cache", default=None); ap.add_argument("--nli", default=None); ap.add_argument("--fidelity", default="soft", choices=["soft", "bertscore"])
    ap.add_argument("--band", type=float, nargs=2, default=[6, 8]); ap.add_argument("--out", default="eval_out")
    a = ap.parse_args()
    a.train_ppo_iters = 0
    os.makedirs(a.out, exist_ok=True)
    names = [n for n in a.systems.split(",") if n in ALL]
    seeds = [int(s) for s in a.seeds.split(",")]
    rows = E.load_rows(a)
    print(f"{len(rows)} examples | systems {names} | seeds {seeds}")

    from sipg_rl.pipeline import SIPGRL, default_parts
    from sipg_rl.llm import OpenAICompatLLM
    from sipg_rl.verifier import HFNLI
    from sipg_rl.fidelity import make_fidelity
    nli = HFNLI(a.nli) if a.nli else None
    base_parts = default_parts(nli=nli, fidelity=make_fidelity(a.fidelity))
    base_parts["nli"], base_parts["fidelity"] = nli, base_parts["fidelity"]
    llm = None
    if not a.no_llm and any(n in names for n in ("llm", "hybrid", "sipg")):
        cand = OpenAICompatLLM(a.llm_url, a.llm_model, a.llm_key, timeout=a.llm_timeout, cache_path=a.llm_cache)
        h = cand.health()
        if h["reachable"] and h["model_available"] is not False:
            llm = cand
        else:
            print(f"WARNING: LLM not usable ({h['error'] or 'model missing'}): 'llm' skipped, 'hybrid'/'sipg' run RULES-ONLY", file=sys.stderr)
    graphs = {ex["id"]: [g for s in base_parts["nlp"](ex["src"]).sents if s.text.strip() for g in [base_parts["builder"].build(s)]] for ex in rows}

    per_seed = {}
    for sd in seeds:
        ck = a.ckpt_pattern.format(seed=sd) if a.ckpt_pattern else a.ckpt
        pol = SIPGRL(checkpoint=ck if os.path.exists(ck) else None)
        if not os.path.exists(ck):
            print(f"WARNING: checkpoint {ck} not found: policy is UNTRAINED for seed {sd}", file=sys.stderr)
        run_names = [n for n in names if (sd == seeds[0] or n in STOCHASTIC)]
        systems = build_systems(a, base_parts, pol.policy, run_names, sd, llm)
        t0 = time.time()
        per_seed[sd] = evaluate(a, rows, base_parts, systems, run_names, sd, graphs)
        print(f"seed {sd}: {len(run_names)} systems in {time.time()-t0:.0f}s")
    # example-level rows: average stochastic systems over seeds (per example), keep deterministic as-is
    final = {}
    for n in names:
        sds = [sd for sd in seeds if n in per_seed[sd] and per_seed[sd][n][0]]
        if not sds:
            print(f"(system '{n}' produced no rows: skipped)")
            continue
        base_rows = per_seed[sds[0]][n][0]
        if not base_rows:
            continue
        merged = []
        for i, r in enumerate(base_rows):
            m = dict(r)
            for k in [k for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None]:
                m[k] = float(np.mean([per_seed[sd][n][0][i][k] for sd in sds if per_seed[sd][n][0][i].get(k) is not None]))
            merged.append(m)
        final[n] = merged
    with open(os.path.join(a.out, "trail.jsonl"), "w") as f:
        for sd in seeds:
            for n, (_, trails) in per_seed[sd].items():
                for t in trails:
                    f.write(json.dumps(dict(t, seed=sd), default=float) + "\n")

    # ---------------- main table (mean [95% CI over examples]) ----------------
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    P(f"## Main table (n={len(rows)} examples, mean [95% bootstrap CI]; % columns in percent)\n")
    P("| system | " + " | ".join(h for h, _ in HEAD) + " |"); P("|---|" + "---|" * len(HEAD))
    for n, rr in final.items():
        cells = [fmtci(*mean_ci([x for x in col(rr, k) if x is not None]), k in PCT) for _, k in HEAD]
        P(f"| {n}{'*' if n in ('hybrid','sipg') and llm is None else ''} | " + " | ".join(cells) + " |")
    if llm is None and any(n in final for n in ("hybrid", "sipg")):
        P("\n*no LLM was reachable: hybrid/sipg ran RULES-ONLY.")
    # ---------------- comparisons with Holm ----------------
    ref_sys = a.compare_to if a.compare_to in final else next(iter(final))
    P(f"\n## Paired differences vs `{ref_sys}` (mean diff [95% CI], Holm-adjusted p across the {len(HEAD)} metrics)\n")
    P("| system | " + " | ".join(h for h, _ in HEAD) + " |"); P("|---|" + "---|" * len(HEAD))
    for n, rr in final.items():
        if n == ref_sys: continue
        cells, ps = {}, {}
        for h, k in HEAD:
            d, lo, hi, p = paired_bootstrap([x if x is not None else np.nan for x in col(rr, k)], [x if x is not None else np.nan for x in col(final[ref_sys], k)])
            cells[k], ps[k] = (d, lo, hi), p
        adj = holm(ps)
        out = []
        for h, k in HEAD:
            d, lo, hi = cells[k]; kk = 100 if k in PCT else 1
            out.append(f"{d*kk:+.3f} [{lo*kk:+.3f},{hi*kk:+.3f}]{'*' if adj[k] < 0.05 else ''}")
        P(f"| {n} | " + " | ".join(out) + " |")
    P("\n`*` = significant after Holm-Bonferroni (adj. p < 0.05).")
    # ---------------- seed variance ----------------
    if len(seeds) > 1:
        P(f"\n## Across-seed variability ({len(seeds)} seeds; mean +- std [t 95% CI half-width] of per-seed system means)\n")
        P("| system | " + " | ".join(h for h, _ in HEAD[:8]) + " |"); P("|---|" + "---|" * 8)
        for n in names:
            if n not in STOCHASTIC or n not in final: continue
            cells = []
            for h, k in HEAD[:8]:
                vals = [np.mean([x for x in col(per_seed[sd][n][0], k) if x is not None]) for sd in seeds if n in per_seed[sd]]
                m, sdv, hw = mean_std_across_seeds(vals); kk = 100 if k in PCT else 1
                cells.append(f"{m*kk:.3f} +- {sdv*kk:.3f} [{hw*kk:.3f}]")
            P(f"| {n} | " + " | ".join(cells) + " |")
    # ---------------- sub-tables ----------------
    P("\n## Layman-hit by source node type (share of nodes whose plain wording / split appears)\n")
    kinds = ["LEX", "NOM", "AMOD", "CPLX", "clause"]
    P("| system | " + " | ".join(kinds) + " |"); P("|---|" + "---|" * len(kinds))
    for n, rr in final.items():
        P(f"| {n} | " + " | ".join(cell([r["layman_by_node"] for r in rr], k) for k in kinds) + " |")
    P("\n## Critical-fact preservation by category (% preserved, among examples containing the category)\n")
    P("| system | " + " | ".join(M.CRITICAL_CATEGORIES) + " |"); P("|---|" + "---|" * len(M.CRITICAL_CATEGORIES))
    for n, rr in final.items():
        P(f"| {n} | " + " | ".join(cell([r["critical"] for r in rr], k) for k in M.CRITICAL_CATEGORIES) + " |")
    P("\n## Jargon kept by category (% of source jargon still present; drug must stay 100)\n")
    P("| system | " + " | ".join(M.JARGON_CATEGORIES) + " |"); P("|---|" + "---|" * len(M.JARGON_CATEGORIES))
    for n, rr in final.items():
        P(f"| {n} | " + " | ".join(cell([r["jargon_cat"] for r in rr], k) for k in M.JARGON_CATEGORIES) + " |")
    P("\n## Edit-type distribution (fraction of edited tokens)\n")
    et = ["replacement", "deletion", "insertion", "elaboration"]
    P("| system | " + " | ".join(et) + " |"); P("|---|" + "---|" * len(et))
    dist = {}
    for n, rr in final.items():
        dist[n] = [float(np.mean([r["edit_types"][k] for r in rr])) for k in et]
        P(f"| {n} | " + " | ".join(f"{100*v:.1f}" for v in dist[n]) + " |")
    P("\n## Document-level SARI (whole eval set as one document)\n")
    for n, rr in final.items():
        if n in ("copy", "reference"): continue
        ds = M.d_sari([r["src"] for r in rr], [r["out"] for r in rr], [r["ref"] for r in rr])
        P(f"- {n}: D-SARI {ds['sari']:.1f}")
    if nli:
        P("\n## NLI semantic-drift proxy (src entails out but out does not entail src)\n")
        for n, rr in final.items():
            v = [r["nli_drift"] for r in rr if r.get("nli_drift") is not None]
            P(f"- {n}: {100*np.mean(v):.1f}% of outputs" if v else f"- {n}: n/a")
    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(lines))
    # stacked bar of edit types
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4)); bottom = np.zeros(len(dist))
        for j, k in enumerate(et):
            vals = np.array([dist[n][j] for n in dist]); ax.bar(list(dist), vals, bottom=bottom, label=k); bottom += vals
        ax.set_ylabel("fraction of edited tokens"); ax.legend(); plt.xticks(rotation=30); plt.tight_layout()
        plt.savefig(os.path.join(a.out, "edit_types.png"), dpi=130)
    except Exception as e:
        print("plot skipped:", e)
    # case series for the human-eval substitute (rejections / fallbacks / drift)
    with open(os.path.join(a.out, "case_series.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["system", "id", "reason", "source", "output"])
        for n, rr in final.items():
            for r in rr:
                why = [x for x, c in (("final_fallback", r.get("final_fallback", 0) > 0), ("edit_rejected", r.get("n_rejected", 0) > 0),
                                      ("nli_drift", (r.get("nli_drift") or 0) > 0), ("verifier_fail", r.get("verifier_ok", 1) < 1)) if c]
                if why and n not in ("reference",): w.writerow([n, r["id"], "+".join(why), r["src"], r["out"]])
    json.dump({n: [{k: v for k, v in r.items() if k not in ("src", "ref")} for r in rr] for n, rr in final.items()}, open(os.path.join(a.out, "rows.json"), "w"), default=float)
    print(f"\nwrote {a.out}/tables.md, trail.jsonl, case_series.csv, rows.json, edit_types.png")


if __name__ == "__main__":
    main()
