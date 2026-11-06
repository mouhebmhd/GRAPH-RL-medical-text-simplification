"""SIPG-RL evaluation suite (gates -> core table -> diagnostics -> safety -> risk-coverage -> pre-registered verdict).

  python scripts/eval_suite.py prereg --out eval                         # 1. freeze the primary metric + pass/fail criteria
  python scripts/eval_suite.py d3     --nli hf                           # 2. verifier recall with NLI on/off (no dataset needed)
  python scripts/eval_suite.py gates  --split test --limit 150 --out eval --nli hf --llm-model qwen2.5:7b-instruct
  python scripts/eval_suite.py run    --split test --limit 0   --out eval --nli hf --ckpts runs/v2/policy_seed0.pt,...,policy_seed4.pt
  python scripts/eval_suite.py audit  --out eval --audit-labels eval/audit_gold_labeled.csv   # 3. summarise manual G3 labels

Second corpus: same commands with --dataset <name> or --jsonl <file> and --src-field/--ref-field.
Human evaluation (clinician hallucination rate, kappa) is deliberately NOT part of this script. The hallucination column here is an
AUTOMATIC PROXY (new number or drug name not in the source); the pre-registered "hallucination <= 2%" criterion is reported as
NOT EVALUATED until a clinician sample exists.  Not implemented (use EASSE / lens): D-SARI, LENS.
"""
import argparse, csv, datetime, hashlib, json, os, random, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)        # nanmean of all-NaN columns (e.g. no layman terms)
from sipg_rl import metrics as M, stress as ST, drift as DR, adversarial as ADV

SYSTEMS = ["copy", "rules", "llm", "hybrid", "greedy", "sipg", "sipg_noverif", "reference"]
POLICY_SYSTEMS = {"rules", "llm", "hybrid", "sipg", "sipg_noverif"}          # these depend on the checkpoint (seeds)
DEFAULT_PREREG = dict(
    primary_metric="safe_layman_hit",
    primary_definition="mean over examples of (layman-term hit rate) x (1 if every critical-span category is preserved else 0)",
    criteria=dict(
        beats_noop="hybrid or sipg: SARI > copy SARI, paired-bootstrap 95% CI excludes 0",
        does_task="sipg layman hit% >= 0.65",
        safe="critical-span preservation = 100% on the stress set AND clinician-sample hallucination <= 2% (the latter needs human evaluation)",
        beats_rules="sipg layman hit% > rules, Holm-adjusted p < 0.05, AND SARI not significantly worse",
        rl_earns_keep="sipg > greedy on SARI or layman hit%, Holm-adjusted p < 0.05, on real data",
        fallback_works="fallback% > 0 (real or adversarial inputs) AND fallback outputs 100% critical-span preserved"),
    thresholds=dict(layman_hit_min=0.65, stress_preservation=1.0, hallucination_max=0.02, alpha=0.05, g1_margin=0.15, g2_overshoot=0.03, g3_fp_max=0.10),
    seeds_min=5, bootstrap=5000)


# ================================================================================================ data
def load_rows(a, split=None, need_ref=True):
    split = split or a.split
    if a.jsonl and split == a.split:
        raw = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset(a.dataset)
        if split not in ds:
            sys.exit(f"split '{split}' not in {list(ds.keys())}")
        raw = [dict(r) for r in ds[split]]
    pick = lambda r, *ks: next((r[k] for k in ks if k in r and r[k]), None)
    out = []
    for i, r in enumerate(raw):
        x, y = pick(r, a.src_field, "source"), pick(r, a.ref_field, "target")
        if not x or (need_ref and not y) or len(x.split()) < a.min_words:
            continue
        out.append(dict(id=int(r.get("idx", i)), src=x.strip(), ref=(y or "").strip(), raw=r))
    if a.shuffle:
        np.random.default_rng(a.seed).shuffle(out)
    return out[:a.limit] if a.limit else out


# ================================================================================================ instruments (the things we MEASURE with)
class Instruments:
    """Scorer/encoder used for the reported S and Sim. Kept separate from the policy's training reward on purpose."""
    def __init__(self, a):
        from sipg_rl.scoring import BandSuitabilityScorer, ContentEncoder
        self.encoder, self.fitted, self.note = ContentEncoder(), False, ""
        self.scorer = BandSuitabilityScorer()
        texts = []
        try:
            if a.fit_jsonl:
                texts = [json.loads(l).get(a.ref_field) or json.loads(l).get("target") for l in open(a.fit_jsonl) if l.strip()]
            elif not a.no_fit:
                texts = [r["ref"] for r in load_rows(a, a.fit_split)]
        except Exception as e:
            self.note = f"could not load fit split ({type(e).__name__}: {e})"
        texts = [t for t in texts if t]
        if len(texts) >= 30:
            self.scorer.fit(texts); self.fitted = True
            self.note = f"bands fitted on {len(texts)} reference texts from {a.fit_jsonl or a.fit_split}"
            os.makedirs(a.out, exist_ok=True)
            json.dump(self.scorer.to_dict(), open(os.path.join(a.out, "scorer_bands.json"), "w"), indent=1)
        else:
            self.note = self.note or "NOT FITTED: default bands (G2 result is not valid)"

        self.sim_kind = getattr(a, "sim", "content")
        self._simfn = None
        if self.sim_kind == "nli":
            from sipg_rl.fidelity import NLISim
            nli = make_nli(a)
            if nli is None:
                sys.exit("--sim nli needs --nli hf")
            self._simfn = NLISim(nli, getattr(a, "sim_agg", "mean"))
        elif self.sim_kind == "bertscore":
            from sipg_rl.fidelity import BERTScoreSim
            self._simfn = BERTScoreSim(a.bertscore_model, getattr(a, "bertscore_layers", None))
        self.note += f" | Sim = {self.sim_kind}"

    def S(self, t):
        return self.scorer(t)


    def sim(self, src, out):
        if self._simfn is not None:
            return self._simfn(src, out)
        return self.encoder.cos(self.encoder.encode(out), self.encoder.encode(src))


# ================================================================================================ systems
_NLI = {}


def make_nli(a):
    if a.nli == "hf":
        if a.nli_model not in _NLI:
            from sipg_rl.verifier import HFNLI
            try:
                _NLI[a.nli_model] = HFNLI(a.nli_model)
            except Exception as e:
                sys.exit(f"--nli hf failed ({type(e).__name__}: {str(e)[:150]}).\nFix the install (e.g. `pip uninstall torchvision`, see README) or pass --nli none explicitly; "
                         "without NLI the final gate cannot catch semantic drift.")
        return _NLI[a.nli_model]
    if a.nli == "stub":          # SMOKE TESTS ONLY: flags antonym drift; not a real NLI model
        return lambda p, h: {"entailment": 0.05 if ADV.leaked("antonym_drift", p, h) else 0.95,
                             "contradiction": 0.9 if ADV.leaked("antonym_drift", p, h) else 0.02}
    return None


def make_verifier(a, nli):
    from sipg_rl.verifier import FidelityVerifier
    return FidelityVerifier(nli=nli, nli_mode=a.nli_mode, nli_threshold=a.nli_threshold, nli_contra_threshold=a.nli_contra_threshold)


class Systems:
    def __init__(self, a, names, ckpts):
        import spacy
        from sipg_rl.llm import OpenAICompatLLM
        self.a, self.names, self.ckpts = a, names, ckpts
        self.nlp = spacy.load("en_core_web_sm")
        self.nli = make_nli(a)
        self.llm = None
        if set(names) & {"llm", "hybrid", "sipg", "sipg_noverif", "greedy"} and a.llm_model != "none":
            self.llm = OpenAICompatLLM(a.llm_url, a.llm_model, a.llm_key, timeout=a.llm_timeout, cache_path=a.llm_cache)
            h = self.llm.health()
            if not h["reachable"] or h["model_available"] is False:
                sys.exit(f"LLM not usable: {h}. Start Ollama / pull the model, or use --llm-model none to skip LLM systems.")
        self._cache = {}

    def _parts(self, kind):
        from sipg_rl.pipeline import default_parts
        from sipg_rl.realize import RuleRealizer, RoutedRealizer, CallableRealizer
        from sipg_rl.verifier import NullVerifier
        rz = {"rules": lambda: RuleRealizer(),
              "llm": lambda: CallableRealizer(self.llm, RuleRealizer()),
              "hybrid": lambda: RoutedRealizer(self.llm, RuleRealizer(), allow_llm_structure=getattr(self.a, "llm_structure", False)),
              "sipg": lambda: RoutedRealizer(self.llm, RuleRealizer()) if self.llm else RuleRealizer(),
              "sipg_noverif": lambda: RoutedRealizer(self.llm, RuleRealizer()) if self.llm else RuleRealizer()}[kind]()
        ver = NullVerifier() if kind == "sipg_noverif" else make_verifier(self.a, self.nli)
        return default_parts(nlp=self.nlp, realizer=rz, verifier=ver)

    def runner(self, name, ck=None):
        """-> callable(text) -> SimplifyResult"""
        from sipg_rl.pipeline import SIPGRL
        from sipg_rl.baselines import GreedyReadabilityPolicy
        key = (name, ck)
        if ck and not os.path.exists(ck):
            sys.exit(f"checkpoint not found: {ck}")
        if key in self._cache:
            return self._cache[key]
        if name == "greedy":
            from sipg_rl.pipeline import default_parts
            from sipg_rl.realize import RoutedRealizer, RuleRealizer
            m = SIPGRL(default_parts(nlp=self.nlp, realizer=RoutedRealizer(self.llm, RuleRealizer()) if self.llm else RuleRealizer(),
                                     verifier=make_verifier(self.a, self.nli)))
            fn = lambda t: m.simplify(t, driver=GreedyReadabilityPolicy())
        else:
            m = SIPGRL(self._parts(name), checkpoint=ck, sweep=name != "rules")
            fn = lambda t: m.simplify(t)
        self._cache[key] = fn
        return fn


def run_one(name, fn, ex):
    if name == "copy":
        return ex["src"], dict(n_edits=0, n_rejected=0, n_invokes=0, final_fallback=0, n_dropped=0)
    if name == "reference":
        return ex["ref"], dict(n_edits=0, n_rejected=0, n_invokes=0, final_fallback=0, n_dropped=0)
    res = fn(ex["src"])
    S = res.sentences
    return res.text, dict(n_edits=sum(len(s["operations"]) for s in S), n_rejected=sum(len(s["rejected"]) for s in S),
                          n_invokes=sum(s["n_invokes"] for s in S), final_fallback=int(any(s["final_rejected"] for s in S)),
                          n_dropped=sum(s.get("n_dropped", 0) for s in S))


def row_metrics(ex, out, info, inst, ver):
    s, r = ex["src"], ex["ref"]
    crit = M.critical_report(s, out)
    flags = [v for v in crit.values() if v is not None]
    crit_all = all(flags) if flags else True
    et, lt = M.parse_terms(ex["raw"].get("expert_terms")), M.parse_terms(ex["raw"].get("layman_terms"))
    lh = M.term_retention(lt, out)
    row = dict(id=ex["id"], out=out, **info, **M.sari(s, out, r), rougeL=M.rouge_l(out, r), fkgl=M.fkgl(out), ari=M.ari(out),
               d_fkgl=M.fkgl(out) - M.fkgl(s), S=inst.S(out), S_src=inst.S(s), sim=inst.sim(s, out),
               changed=int(out.strip() != s.strip()), len_ratio=len(out.split()) / max(1, len(s.split())),
               crit_all=int(crit_all), halluc=int(M.hallucination_proxy(s, out)),
               verifier_ok=int(ver.check(s, out, use_nli=False).ok), layman_hit=lh, expert_term_kept=M.term_retention(et, out),
               safe_layman=None if lh is None else lh * int(crit_all))
    row.update({f"crit_{k}": (None if v is None else int(v)) for k, v in crit.items()})
    row.update({f"edit_{k}": v for k, v in M.edit_type_counts(s, out).items()})
    return row


def run_systems(a, names, data, inst, S, ckpts):
    """rows[name] -> list over seeds of list of row dicts (policy systems only have >1 seed)."""
    ver = make_verifier(a, None)
    rows = {}
    os.makedirs(a.out, exist_ok=True)
    for name in names:
        seeds = ckpts if name in POLICY_SYSTEMS else [None]
        rows[name] = []
        for k, ck in enumerate(seeds):
            tag = name + (f"#{k}" if len(seeds) > 1 else "")
            path = os.path.join(a.out, f"rows_{tag}.jsonl")
            done = {}
            if a.resume and os.path.exists(path):
                done = {r["id"]: r for r in map(json.loads, open(path))}
            fn = None if name in ("copy", "reference") else S.runner(name, ck)
            f, rr, t0 = open(path, "a" if a.resume else "w"), [], time.time()
            for i, ex in enumerate(data):
                if ex["id"] in done:
                    rr.append(done[ex["id"]]); continue
                t1 = time.time()
                out, info = run_one(name, fn, ex)
                r = row_metrics(ex, out, info, inst, ver); r["sec"] = time.time() - t1
                rr.append(r); f.write(json.dumps(r) + "\n"); f.flush()
                if (i + 1) % 25 == 0:
                    print(f"  [{tag}] {i+1}/{len(data)} {(time.time()-t0)/(i+1):.1f}s/ex", flush=True)
            f.close(); rows[name].append(rr)
    return rows


# ================================================================================================ aggregation helpers
def col(rows, name, key):
    """seed-averaged per-example column (NaN where undefined)."""
    mats = np.array([[np.nan if r.get(key) is None else float(r[key]) for r in rr] for rr in rows[name]], float)
    with np.errstate(all="ignore"):
        return np.nanmean(mats, axis=0) if mats.shape[0] > 1 else mats[0]


def seed_means(rows, name, key):
    return [float(np.nanmean([np.nan if r.get(key) is None else float(r[key]) for r in rr])) for rr in rows[name]]


def mean(rows, name, key):
    v = col(rows, name, key)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def pct(x):
    return "n/a" if x is None or x != x else f"{100*x:.1f}"


def paired(rows, a, b, key, seed=0):
    x, y = col(rows, a, key), col(rows, b, key)
    m = np.isfinite(x) & np.isfinite(y)
    return M.paired_bootstrap(x[m], y[m], seed=seed) if m.sum() > 2 else None


def md_table(header, body):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in body])


# ================================================================================================ G3: verifier vs gold
def triage(src, ref, violations):
    low, tags = ref.lower(), []
    spelled = re.search(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|hundred|half|once|twice)\b", low)
    for v in violations:
        if "number" in v:
            fmt = set(re.findall(r"\d+(?:\.\d+)?", src.replace(",", ""))) <= set(re.findall(r"\d+(?:\.\d+)?", ref.replace(",", "")))
            tags.append("likely_FP:number_format" if fmt else "likely_FP:spelled_number" if spelled else "review:number_dropped")
        elif "negation" in v:
            tags.append("likely_FP:negation_paraphrase" if re.search(r"\b(?:cannot|can't|unable|fail\w*|lack\w*|free of|stop|avoid|unlikely|none)\b", low) else "review:negation_dropped")
        elif "hedge" in v:
            tags.append("likely_FP:hedge_paraphrase" if re.search(r"\b(?:might|could|maybe|perhaps|chance|probabl\w+|seems?|looks like)\b", low) else "review:hedge_dropped")
        elif "conditional" in v:
            tags.append("likely_FP:conditional_paraphrase" if re.search(r"\b(?:when|whenever|in case|provided|as long as|should)\b", low) else "review:conditional_dropped")
        else:
            tags.append("review:critical_term")
    return tags


def audit_gold(a, data, out_dir, thresholds):
    """Gold rejections by the CURRENT verifier (numbers normalised), compared with the legacy number check on the same rows.
    audit_gold.csv = rows still rejected (to label a/b/c); audit_gold_format_artefacts.csv = rows the legacy check rejected and
    normalisation now accepts (sample them to confirm they really are formatting)."""
    from sipg_rl import numnorm
    from sipg_rl.verifier import FidelityVerifier
    ver = make_verifier(a, None)
    old = FidelityVerifier(nli=None, legacy_numbers=True)
    rej, fixed, kinds, causes = [], [], {}, {}
    n_old = 0
    for ex in data:
        r = ver.check(ex["src"], ex["ref"], use_nli=False)
        ro = old.check(ex["src"], ex["ref"], use_nli=False)
        n_old += int(not ro.ok)
        if r.ok and not ro.ok:
            dg = numnorm.diagnose(ex["src"], ex["ref"])
            for d in dg:
                causes[d["cause"]] = causes.get(d["cause"], 0) + 1
            fixed.append(dict(id=ex["id"], legacy_violations=" | ".join(ro.violations), causes=" | ".join(d["cause"] for d in dg), src=ex["src"], ref=ex["ref"]))
        if not r.ok:
            tg = triage(ex["src"], ex["ref"], r.violations)
            likely = bool(tg) and all(t.startswith("likely_FP") for t in tg)
            rej.append(dict(id=ex["id"], violations=" | ".join(r.violations), auto_triage=" | ".join(tg),
                            suggested="a? (likely false positive)" if likely else "review", src=ex["src"], ref=ex["ref"], label="", note=""))
            for v in r.violations:
                k = v.split(":")[0].split("(")[0].strip()
                kinds[k] = kinds.get(k, 0) + 1
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "audit_gold.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "violations", "auto_triage", "suggested", "src", "ref", "label", "note"])
        w.writeheader(); w.writerows(rej)
    with open(os.path.join(out_dir, "audit_gold_format_artefacts.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "legacy_violations", "causes", "src", "ref"])
        w.writeheader(); w.writerows(fixed)
    lo, hi = wilson(len(rej), len(data))
    likely_fp = sum(1 for r in rej if r["suggested"].startswith("a?"))
    res = dict(n=len(data), rejected=len(rej), rate=len(rej) / max(1, len(data)), ci=(lo, hi), by_kind=kinds, auto_likely_fp=likely_fp,
               legacy_rejected=n_old, legacy_rate=n_old / max(1, len(data)), format_artefacts=len(fixed), artefact_causes=causes,
               csv=os.path.join(out_dir, "audit_gold.csv"))
    json.dump(res, open(os.path.join(out_dir, "gold_audit.json"), "w"), indent=1, default=float)
    return res


def summarise_labels(path, thresholds):
    rows = [r for r in csv.DictReader(open(path, encoding="utf8")) if r.get("label", "").strip()]
    c = {k: sum(1 for r in rows if r["label"].strip().lower() == k) for k in "abc"}
    n = sum(c.values())
    return dict(n=n, **c, fp_share=c["a"] / n if n else float("nan"))


# ================================================================================================ D3 / G4 / D1 helpers
def d3_recall(a, nli):
    """Verifier recall on semantic drift + false-positive rate on faithful rewrites, lexical-only vs NLI modes."""
    from sipg_rl.verifier import FidelityVerifier
    configs = [("lexical only", None, "entailment", 0.5)]
    if nli is not None:
        configs += [("lexical + NLI entailment>=%.2f" % t, nli, "entailment", t) for t in (0.3, 0.5, 0.7)]
        configs += [("lexical + NLI contradiction<%.2f" % t, nli, "contradiction", t) for t in (0.3, 0.5, 0.7)]
    out = []
    for label, n_, mode, thr in configs:
        v = FidelityVerifier(nli=n_, nli_mode=mode, nli_threshold=thr if mode == "entailment" else 0.5,
                             nli_contra_threshold=thr if mode == "contradiction" else 0.5, nli_scope="all")
        caught = {}
        for kind, s, d in DR.DRIFT:
            caught.setdefault(kind, []).append(not v.check(s, d, use_nli=n_ is not None).ok)
        allc = [x for l in caught.values() for x in l]
        fp = [not v.check(s, f, use_nli=n_ is not None).ok for s, f in DR.FAITHFUL]
        out.append(dict(config=label, recall=float(np.mean(allc)), recall_ci=wilson(sum(allc), len(allc)), fp=float(np.mean(fp)),
                        fp_ci=wilson(sum(fp), len(fp)), by_kind={k: float(np.mean(v_)) for k, v_ in caught.items()}))
    return out


def adversarial_check(S, sources, nli_on):
    """Where each injected corruption is stopped: edit gate / final gate / leaked (see sipg_rl.adversarial)."""
    from sipg_rl.pipeline import SIPGRL, default_parts
    mk = lambda rz: SIPGRL(default_parts(nlp=S.nlp, realizer=rz, verifier=make_verifier(S.a, S.nli if nli_on else None)))
    return ADV.selftest(mk, sources, max_per_mode=25, min_injected=9, nli=S.nli if nli_on else None)


def stress_eval(S, names, ckpts):
    items = ST.build_stress_set()
    res = {}
    for name in names:
        if name in ("copy", "reference"):
            continue
        seeds = ckpts if name in POLICY_SYSTEMS else [None]
        per_cat = {}
        for ck in seeds:
            fn = S.runner(name, ck)
            for it in items:
                ok = ST.check_item(it, fn(it.text).text)
                per_cat.setdefault(it.category, []).append(ok)
        res[name] = {c: (sum(v), len(v)) for c, v in per_cat.items()}
    return res


# ================================================================================================ C2: by span type
def span_type_breakdown(S, data, rows, systems):
    import difflib
    from sipg_rl.pipeline import default_parts
    P = default_parts(nlp=S.nlp)
    norm = lambda t: re.sub(r"\s+([.,;:!?])", r"\1", " ".join(t.lower().split()))
    stats = {sy: {} for sy in systems}
    for i, ex in enumerate(data):
        st, rt = ex["src"].lower().split(), ex["ref"].lower().split()
        sm = difflib.SequenceMatcher(a=st, b=rt, autojunk=False)
        repl = [(set(w.strip(".,;:()") for w in st[i1:i2]), " ".join(rt[j1:j2]).strip(".,;:()")) for tag, i1, i2, j1, j2 in sm.get_opcodes()
                if tag == "replace"]
        if not repl:
            continue
        nodes = []
        for sent in P["nlp"](ex["src"]).sents:
            if sent.text.strip():
                g = P["builder"].build(sent)
                nodes += [(n.kind, norm(n.text)) for n in g.nodes if n.kind != "ROOT"]
        for kind, text in nodes:
            toks = [w.strip(".,;:()") for w in text.split()]
            hit_ref = [rw for ts, rw in repl if toks and sum(w in ts for w in toks) / len(toks) >= 0.5]
            if not hit_ref:
                continue
            for sy in systems:
                out = norm(rows[sy][0][i]["out"])
                d = stats[sy].setdefault(kind, dict(n=0, changed=0, ref_wording=0))
                d["n"] += 1
                changed = text not in out
                d["changed"] += int(changed)
                d["ref_wording"] += int(changed and any(rw and rw in out for rw in hit_ref))
    return stats


# ================================================================================================ E: risk-coverage
def risk_coverage_eval(a, data, rows, inst, nli, pool=("sipg_noverif", "llm")):
    sig_rows = []
    for name in pool:
        if name not in rows:
            continue
        for rr in rows[name][:1]:
            for ex, r in zip(data, rr):
                feat = dict(sim=r["sim"], n_edits=r["n_edits"], len_ratio=r["len_ratio"], dS=r["S"] - r["S_src"],
                            nli=(nli(ex["src"], r["out"]).get("entailment", 1.0) if nli else None), logprob=r.get("logprob"),
                            err=int((not r["crit_all"]) or r["halluc"]), system=name)
                sig_rows.append(feat)
    if len(sig_rows) < 40:
        return dict(skipped="fewer than 40 outputs")
    rng = np.random.default_rng(a.seed); idx = rng.permutation(len(sig_rows)); h = len(idx) // 2
    tr, te = [sig_rows[i] for i in idx[:h]], [sig_rows[i] for i in idx[h:]]
    err_te = np.array([r["err"] for r in te], float)
    if err_te.sum() < 5 or (1 - err_te).sum() < 5:
        return dict(skipped=f"only {int(err_te.sum())} critical-span errors among {len(te)} held-out outputs; need >= 5 of each class "
                            "(a verified system produces ~none: evaluate on the unverified ablation / legacy LLM with a real model)")
    feats = ["sim", "n_edits", "len_ratio", "dS"] + (["nli"] if nli else [])
    X = lambda rs: np.array([[r[f] for f in feats] for r in rs], float)
    lr = M.LogReg().fit(X(tr), [r["err"] for r in tr])
    p_err = lr.predict(X(te))
    curves = {"calibrated (logreg)": (1 - p_err), "Sim only (ContentEncoder)": np.array([r["sim"] for r in te])}
    if nli:
        curves["NLI entailment only"] = np.array([r["nli"] for r in te])
    if all(r["logprob"] is not None for r in te):
        curves["decoder logprob only"] = np.array([r["logprob"] for r in te])
    res = dict(n_test=len(te), error_rate=float(err_te.mean()), curves={})
    for k, conf in curves.items():
        cov, risk, aurc = M.risk_coverage(conf, err_te)
        res["curves"][k] = dict(aurc=aurc, risk_at_80=M.risk_at_coverage(conf, err_te, 0.8), cov=cov.tolist(), risk=risk.tolist())
    res["ece_calibrated"] = M.ece(p_err, err_te)
    res["missing_signals"] = [s for s in ("decoder logprob only", "BERTScore only") if s not in curves]
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, v in res["curves"].items():
            ax.plot(v["cov"], v["risk"], label=f"{k} (AURC {v['aurc']:.3f})")
        ax.set_xlabel("coverage (share of outputs kept; rest revert to source)"); ax.set_ylabel("critical-span error rate among kept")
        ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(a.out, "risk_coverage.png"), dpi=150); plt.close(fig)
        res["plot"] = os.path.join(a.out, "risk_coverage.png")
    except Exception as e:
        res["plot_error"] = str(e)
    return res


# ================================================================================================ stages
def stage_prereg(a):
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "prereg.json")
    if os.path.exists(p) and not a.force:
        sys.exit(f"{p} already exists: pre-registration is frozen (use --force only to start a NEW study).")
    body = dict(DEFAULT_PREREG)
    body["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    body["sha256"] = hashlib.sha256(json.dumps(DEFAULT_PREREG, sort_keys=True).encode()).hexdigest()
    json.dump(body, open(p, "w"), indent=1)
    print(f"wrote {p}\nprimary metric: {body['primary_metric']} = {body['primary_definition']}\nsha256 {body['sha256'][:16]}...")


def load_prereg(a):
    p = os.path.join(a.out, "prereg.json")
    if os.path.exists(p):
        return json.load(open(p))
    if a.no_prereg:
        return dict(DEFAULT_PREREG, sha256="NOT PRE-REGISTERED")
    sys.exit(f"no {p}: run `eval_suite.py prereg --out {a.out}` first (or pass --no-prereg and the report will say so).")


def stage_d3(a):
    nli = make_nli(a)
    res = d3_recall(a, nli)
    print(md_table(["config", "drift recall (95% CI)", "false rejection of faithful (95% CI)"],
                   [[r["config"], f"{pct(r['recall'])} [{pct(r['recall_ci'][0])}, {pct(r['recall_ci'][1])}]",
                     f"{pct(r['fp'])} [{pct(r['fp_ci'][0])}, {pct(r['fp_ci'][1])}]"] for r in res]))
    print("\nrecall by drift type:", {k: pct(v) for k, v in res[-1]["by_kind"].items()})
    if nli is None:
        print("\n(no --nli given: only the lexical verifier was measured. Add --nli hf to measure what NLI contributes.)")
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "d3_verifier_recall.json"), "w"), indent=1, default=float)
    return res


def stage_audit(a):
    prereg = load_prereg(a)
    s = summarise_labels(a.audit_labels, prereg["thresholds"])
    print(f"labelled {s['n']}: (a) verifier false positive {s['a']}, (b) acceptable change {s['b']}, (c) annotator error {s['c']}")
    gp = os.path.join(a.out, "gold_audit.json")
    if not os.path.exists(gp):
        print("run `gates` first (it writes gold_audit.json) to get rates over all gold rows"); return
    g = json.load(open(gp))
    n, rej = g["n"], g["rejected"]
    fp = s["a"] * (rej / s["n"]) if s["n"] and s["n"] < rej else s["a"]       # extrapolate if only a sample was labelled
    note = "" if s["n"] >= rej else f" (labelled {s['n']} of {rej}; false positives extrapolated)"
    print(f"gold n={n}: legacy rejection {pct(g['legacy_rate'])}% -> after number normalisation {pct(rej / n)}% -> after removing verifier false positives {pct((rej - fp) / n)}%{note}")
    print(f"false-positive rate on gold = {pct(fp / n)}% (G3 limit {pct(prereg['thresholds']['g3_fp_max'])}%): {'PASS' if fp / n <= prereg['thresholds']['g3_fp_max'] else 'FAIL'}")


def stage_gates(a):
    prereg = load_prereg(a); th = prereg["thresholds"]
    inst = Instruments(a)
    data = load_rows(a)
    names = [n for n in ("copy", "reference", "rules", "llm", "sipg") if n != "llm" or a.llm_model != "none"]
    if "llm" not in names:
        print("NOTE: --llm-model none: G1/G2 are computed without the LLM system.")
    S = Systems(a, names, [a.ckpts.split(",")[0]])
    rows = run_systems(a, names, data, inst, S, [a.ckpts.split(",")[0]])
    rep = ["# Instrument validation (G1-G4)", f"\nprereg sha256: `{prereg['sha256'][:16]}`  n={len(data)}  scorer: {inst.note}\n"]
    R = lambda n: mean(rows, n, "S") + 0.5 * mean(rows, n, "sim")
    tab = [[n, f"{mean(rows, n, 'sim'):.3f}", f"{mean(rows, n, 'S'):.3f}", f"{R(n):.3f}"] for n in names]
    rep += [md_table(["system", "Sim(out, source)", "S (fitted band)", "S + 0.5*Sim"], tab)]
    sim = {n: mean(rows, n, "sim") for n in names}
    # ---- G1
    lit = (sim["reference"] > sim["rules"] > sim.get("llm", sim["rules"] - 1e-9) > sim["copy"]) and sim["reference"] - sim["copy"] > th["g1_margin"]
    g1i = sim["copy"] - sim["reference"] <= th["g1_margin"]
    drops = {}
    for mode in ("drop_number", "drop_negation", "drop_hedge", "swap_drug"):
        d = [inst.sim(e["src"], e["ref"]) - inst.sim(e["src"], ADV.corrupt(mode, e["ref"])) for e in data if ADV.applies(mode, e["ref"])]
        drops[mode] = (float(np.mean(d)) if len(d) >= 10 else None, len(d))
    g1ii = all(v[0] is not None and v[0] >= th["g1_margin"] for v in drops.values() if v[0] is not None) and any(v[0] is not None for v in drops.values())
    cmp_ = paired(rows, "reference", "copy", "S")
    comp = np.nan_to_num(col(rows, "reference", "S") + 0.5 * col(rows, "reference", "sim")) - (col(rows, "copy", "S") + 0.5 * col(rows, "copy", "sim"))
    g1iii_stat = M.paired_bootstrap(comp, np.zeros_like(comp))
    g1iii = g1iii_stat["diff"] > th["g1_margin"] and g1iii_stat["lo"] > 0
    g1iv = cmp_ is not None and cmp_["diff"] > 0 and cmp_["lo"] > 0            # item 1: the training reward is S alone
    rep += [f"\n## G1 Sim / reward ordering  -> **{'PASS' if (g1i and g1ii and g1iv) else 'FAIL'}**",
            f"- literal criterion (ordering reference > rules > llm > copy on Sim, and Sim(ref) - Sim(copy) > {th['g1_margin']}): **{'pass' if lit else 'fail'}**. "
            "NOTE: for any source-anchored fidelity score Sim(copy, source) = 1 is the maximum, so `reference > copy` on Sim alone cannot hold by construction; "
            "it is only meaningful for the composite reward below.",
            f"- (i) gold is not penalised for being a rewrite: Sim(copy) - Sim(ref) = {sim['copy'] - sim['reference']:.3f} <= {th['g1_margin']}: **{'pass' if g1i else 'fail'}**",
            "- (ii) Sim discriminates unsafe edits (mean drop on the reference when corrupted): " + ", ".join(f"{k} {('%.3f (n=%d)' % v) if v[0] is not None else 'n<10'}" for k, v in drops.items()) + f" (need >= {th['g1_margin']}): **{'pass' if g1ii else 'fail'}**",
            f"- (iii) composite reward S + 0.5*Sim: reference - copy = {g1iii_stat['diff']:+.3f} [{g1iii_stat['lo']:+.3f}, {g1iii_stat['hi']:+.3f}] (need > {th['g1_margin']}, CI > 0): **{'pass' if g1iii else 'fail'}**; "
            f"ordering ref > rules > llm > copy on composite: **{'pass' if (R('reference') > R('rules') > (R('llm') if 'llm' in names else R('rules') - 1e-9) > R('copy')) else 'fail'}** (informational)"]
    rep += [f"- (iv) S ALONE (the training objective): S(reference) - S(copy) = {cmp_['diff']:+.3f} [{cmp_['lo']:+.3f}, {cmp_['hi']:+.3f}] (need > 0 with the CI excluding 0): **{'pass' if g1iv else 'fail'}**" if cmp_ else "- (iv) S(reference) - S(copy): not enough rows"]
    # ---- G2
    ov = mean(rows, "llm", "S") - mean(rows, "reference", "S") if "llm" in names else float("nan")
    ov2 = mean(rows, "sipg", "S") - mean(rows, "reference", "S")
    valid = inst.fitted
    g2 = valid and (ov != ov or ov <= th["g2_overshoot"]) and ov2 <= th["g2_overshoot"]
    rep += [f"\n## G2 S calibrated to gold  -> **{'PASS' if g2 else 'FAIL' if valid else 'INVALID (scorer not fitted on a reference split)'}**",
            f"- S(llm) - S(reference) = {ov:+.3f}; S(sipg) - S(reference) = {ov2:+.3f}; limit {th['g2_overshoot']:+.2f}. |S(ref) - S(llm)| = {abs(ov):.3f}. {inst.note}"]
    # ---- G3
    gold = audit_gold(a, data, a.out, th)
    labels = os.path.join(a.out, "audit_gold_labeled.csv")
    lab = summarise_labels(labels, th) if os.path.exists(labels) else None
    fp_rate_gold = (lab["a"] / lab["n"] * gold["rate"]) if lab and lab["n"] else None
    g3 = "PENDING (label audit_gold.csv -> audit_gold_labeled.csv with label a/b/c, then re-run)" if not lab or not lab["n"] else \
        ("PASS" if fp_rate_gold <= th["g3_fp_max"] else "FAIL")
    rep += [f"\n## G3 verifier vs gold  -> **{g3}**",
            f"- the verifier rejects {gold['rejected']}/{gold['n']} references = {pct(gold['rate'])}% (95% CI {pct(gold['ci'][0])}-{pct(gold['ci'][1])}); by violation: {gold['by_kind']}",
            f"- number normalisation: legacy check rejected {gold['legacy_rejected']}/{gold['n']} = {pct(gold['legacy_rate'])}%; after normalisation {gold['rejected']} = {pct(gold['rate'])}%. "
            f"{gold['format_artefacts']} rows were formatting artefacts (causes: {gold['artefact_causes']}); see audit_gold_format_artefacts.csv",
            f"- auto-triage: {gold['auto_likely_fp']} rejections are *likely* false positives (format/paraphrase) - a heuristic, not a label. Classify {gold['csv']} manually as (a) false positive / (b) acceptable change / (c) annotator error."]
    if lab and lab["n"]:
        rep += [f"- labelled {lab['n']}: a={lab['a']} b={lab['b']} c={lab['c']}; false positives = {pct(fp_rate_gold)}% of gold (limit {pct(th['g3_fp_max'])}%)"]
    # ---- G4
    srcs = [e["src"] for e in data][:60]
    adv_off = adversarial_check(S, srcs, False)
    adv_on = adversarial_check(S, srcs, True) if S.nli is not None else None
    real_fb = float(np.nanmean(col(rows, "sipg", "final_fallback"))); real_rej = float(np.nanmean(col(rows, "sipg", "n_rejected")))
    fb_exec = (real_fb > 0) or (adv_on is not None and any(r.final_gate > 0 for r in adv_on))
    rep += [f"\n## G4 fallback executes  -> **{'PASS' if fb_exec else 'FAIL'}**",
            f"- real data (sipg): final fallback {pct(real_fb)}% of examples; rejected edits per example {real_rej:.2f}",
            "- adversarial injection (NLI off): " + "; ".join(f"{r.mode}: injected {r.injected}, edit-gate {r.edit_gate}, final-gate {r.final_gate}, leaked {r.leaked}" for r in adv_off)]
    if adv_on:
        rep += ["- adversarial injection (NLI on): " + "; ".join(f"{r.mode}: injected {r.injected}, edit-gate {r.edit_gate}, final-gate {r.final_gate}, leaked {r.leaked}" for r in adv_on)]
    else:
        rep += ["- NLI is off, so the FINAL gate cannot fire (it repeats the edit gate's lexical check): pass `--nli hf` to exercise it."]
    chk = adv_on or adv_off
    sd = next((r for r in chk if r.mode == "swap_drug"), None)
    if sd is not None:
        okk = sd.injected >= 9 and (sd.edit_gate + sd.final_gate) >= 9 and sd.leaked == 0
        rep += [f"- swap_drug requirement (injected >= 9, caught >= 9, leaked = 0): injected {sd.injected} ({sd.n_synthetic} synthetic top-up), caught {sd.edit_gate + sd.final_gate}, leaked {sd.leaked}: **{'PASS' if okk else 'FAIL'}**"]
    leaks = [(tag, r.mode, l) for tag, rows_ in (("NLI off", adv_off), ("NLI on", adv_on or [])) for r in rows_ for l in r.leaks]
    with open(os.path.join(a.out, "adversarial_leaks.md"), "w", encoding="utf8") as f:
        f.write("# Adversarial leaks (corruption reached the output)\n\n" + ("none\n" if not leaks else "\n".join(
            f"- [{t}] **{m}**{' (synthetic)' if l['synthetic'] else ''}\n  - source: {l['source']}\n  - output: {l['output']}\n  - NLI: {l['nli']}" for t, m, l in leaks)))
    rep += [f"- leaked cases are written with their NLI probabilities to {os.path.join(a.out, 'adversarial_leaks.md')} ({len(leaks)} total)"]
    txt = "\n".join(rep)
    open(os.path.join(a.out, "gates.md"), "w").write(txt)
    print(txt)
    json.dump(dict(G1=bool(g1i and g1ii and g1iv), G2=bool(g2) if valid else None, G3=g3, G4=bool(fb_exec)), open(os.path.join(a.out, "gates.json"), "w"))


def stage_run(a):
    prereg = load_prereg(a); th = prereg["thresholds"]
    inst = Instruments(a)
    data = load_rows(a)
    ckpts = [c for c in a.ckpts.split(",") if c]
    names = [n.strip() for n in a.systems.split(",") if n.strip()]
    if a.llm_model == "none":
        names = [n for n in names if n not in ("llm", "hybrid")]      # sipg / greedy then run with the rule realizer only
    for n in ("copy", "reference"):
        if n not in names:
            names.insert(0, n)
    S = Systems(a, names, ckpts)
    rows = run_systems(a, names, data, inst, S, ckpts)
    rep = [f"# SIPG-RL evaluation report", f"\nn={len(data)} examples ({a.jsonl or a.dataset}:{a.split}), {len(ckpts)} checkpoint(s)/seeds, LLM {a.llm_model}, "
           f"NLI {a.nli}/{a.nli_mode}\nprereg: {prereg['sha256'][:16]}  scorer: {inst.note}\n"]
    sysn = names
    # ---------------------------------------------------------------- B: core table
    groups = [("Simplification quality", [("SARI", "sari", 1), ("keep", "keep", 1), ("del", "delete", 1), ("add", "add", 1), ("ROUGE-L", "rougeL", 1)]),
              ("Readability (target grade 6-8)", [("FKGL", "fkgl", 2), ("dFKGL", "d_fkgl", 2), ("ARI", "ari", 2)]),
              ("Faithfulness (instrument Sim)", [("Sim", "sim", 3), ("S", "S", 3)]),
              ("Task performance", [("layman hit %", "layman_hit", 100), ("jargon kept %", "expert_term_kept", 100), ("safe layman hit % (primary)", "safe_layman", 100)]),
              ("Safety", [("critical-span kept %", "crit_all", 100), ("halluc. proxy %", "halluc", 100), ("verifier pass %", "verifier_ok", 100), ("fallback %", "final_fallback", 100)]),
              ("Edit behaviour", [("changed %", "changed", 100), ("len x", "len_ratio", 2), ("edits", "n_edits", 2), ("rejected", "n_rejected", 2)])]
    summary = {}
    rep.append("## B. Core comparison (mean [95% CI]; ± = sd across seeds for policy systems)\n")
    for title, cols in groups:
        body = []
        for n in sysn:
            cells = [n]
            for label, key, sc in cols:
                if n == "reference" and key in ("sari", "keep", "delete", "add", "rougeL", "n_edits", "n_rejected", "final_fallback"):
                    cells.append("-"); continue
                v = col(rows, n, key); v = v[np.isfinite(v)]
                m, lo, hi = M.mean_ci(v, n=1000) if len(v) else (float("nan"),) * 3
                f = (lambda x: f"{100*x:.1f}") if sc == 100 else (lambda x, sc=sc: f"{x:.{sc}f}")
                sd = ""
                if len(rows[n]) > 1:
                    sd = f" ±{np.nanstd(seed_means(rows, n, key), ddof=1) * (100 if sc == 100 else 1):.{1 if sc == 100 else max(sc,2)}f}"
                cells.append(f"{f(m)} [{f(lo)}, {f(hi)}]{sd}" if len(v) else "n/a")
                summary.setdefault(n, {})[key] = m
            body.append(cells)
        rep += [f"**{title}**\n", md_table(["system"] + [c[0] for c in cols], body), ""]
    # corpus-level readability / optional metrics
    corp = []
    for n in sysn:
        text = " ".join(r["out"] for r in rows[n][0])
        corp.append([n, f"{M.smog(text):.2f}", f"{sum(1 for r in rows[n][0] if r['n_dropped'] > 0) / len(data) * 100:.1f}"])
    rep += ["**Corpus-level SMOG and graph truncation (F1)**\n", md_table(["system", "SMOG", "% examples with nodes lost to the 24-node cap"], corp), ""]
    try:
        import sacrebleu
        rep += ["**BLEU (sacrebleu, corpus)**: " + ", ".join(f"{n} {sacrebleu.corpus_bleu([r['out'] for r in rows[n][0]], [[e['ref'] for e in data]]).score:.1f}" for n in sysn if n != 'reference'), ""]
    except ImportError:
        rep += ["BLEU: `pip install sacrebleu` to include it.", ""]
    if a.bertscore:
        try:
            from bert_score import score as bscore
            refs, srcs = [e["ref"] for e in data], [e["src"] for e in data]
            bt = []
            for n in sysn:
                outs = [r["out"] for r in rows[n][0]]
                P_, R_, F_ = bscore(outs, srcs, model_type=a.bertscore_model, lang="en", verbose=False)
                Pr, Rr, Fr = bscore(outs, refs, model_type=a.bertscore_model, lang="en", verbose=False)
                bt.append([n, f"{P_.mean():.3f}/{R_.mean():.3f}/{F_.mean():.3f}", f"{Pr.mean():.3f}/{Rr.mean():.3f}/{Fr.mean():.3f}"])
            rep += ["**BERTScore P/R/F1** (" + a.bertscore_model + ")\n", md_table(["system", "vs source (faithfulness)", "vs reference (quality)"], bt), ""]
        except Exception as e:
            rep += [f"BERTScore skipped: {type(e).__name__}: {e}", ""]
    # ---------------------------------------------------------------- paired comparisons with Holm
    pairs = [(x, y, k) for x, y in (("sipg", "copy"), ("hybrid", "copy"), ("sipg", "rules"), ("sipg", "greedy"), ("sipg", "llm"), ("sipg", "sipg_noverif"))
             for k in ("sari", "layman_hit", "safe_layman")
             if x in rows and y in rows]
    res = {f"{x}-{y}:{k}": paired(rows, x, y, k) for x, y, k in pairs}
    res = {k: v for k, v in res.items() if v}
    adj = M.holm({k: v["p"] for k, v in res.items()})
    rep += ["## Paired bootstrap (5000 resamples; Holm-adjusted p over the whole family of %d tests)\n" % len(res),
            md_table(["comparison", "diff [95% CI]", "d_z", "p", "Holm p"],
                     [[k, f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]", f"{v['dz']:.2f}", f"{v['p']:.4f}", f"{adj[k]:.4f}"] for k, v in res.items()]), ""]
    # ---------------------------------------------------------------- C1
    rep.append("## C1. Edit-type distribution (our alignment taxonomy, applied identically; % of edit operations)\n")
    body = []
    for n in sysn:
        tot = {k: int(np.nansum(col(rows, n, f"edit_{k}"))) for k in ("replacement", "elaboration", "insertion", "deletion")}
        if n == "reference":
            tot = {k: sum(M.edit_type_counts(e["src"], e["ref"])[k] for e in data) for k in tot}
        s_ = sum(tot.values()) or 1
        body.append([n] + [f"{100*tot[k]/s_:.1f}" for k in tot] + [sum(tot.values())])
    rep += [md_table(["system", "replacement", "elaboration", "insertion", "deletion", "total ops"], body)]
    ann = [M.annotation_label_counts(e["raw"].get("Annotation")) for e in data if "Annotation" in e["raw"]]
    if ann:
        t = {k: sum(x[k] for x in ann) for k in ann[0]}
        rep += [f"\nDataset's own `Annotation` field (keyword counts only; format not assumed): {t}"]
    rep.append("")
    # ---------------------------------------------------------------- C2
    focus = [n for n in ("rules", "hybrid", "sipg") if n in rows]
    if focus:
        st = span_type_breakdown(S, data, rows, focus)
        rep.append("## C2. By span type (spans the reference rewrote; hit = changed AND the reference's wording appears)\n")
        kinds = sorted({k for n in focus for k in st[n]})
        rep += [md_table(["node kind", "n targeted"] + [f"{n}: changed% / ref-wording%" for n in focus],
                         [[k, st[focus[0]].get(k, {}).get("n", 0)] + [f"{pct(st[n][k]['changed']/st[n][k]['n']) if k in st[n] and st[n][k]['n'] else 'n/a'} / "
                                                                    f"{pct(st[n][k]['ref_wording']/st[n][k]['n']) if k in st[n] and st[n][k]['n'] else 'n/a'}" for n in focus] for k in kinds]), ""]
    # ---------------------------------------------------------------- C3
    rep.append("## C3. Critical-span preservation by category (% preserved, n with that category in the source)\n")
    body = []
    for n in sysn:
        cells = [n]
        for c in M.CRIT_CATEGORIES:
            v = col(rows, n, f"crit_{c}"); v = v[np.isfinite(v)]
            cells.append(f"{pct(v.mean())} (n={len(v)})" if len(v) else "n/a")
        body.append(cells)
    rep += [md_table(["system"] + list(M.CRIT_CATEGORIES), body), ""]
    # ---------------------------------------------------------------- C4 sample
    rng = random.Random(a.seed)
    with open(os.path.join(a.out, "failure_sample.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["system", "id", "source", "output", "reference", "auto_tags", "label (meaning_changed/jargon_untouched/ungrammatical/over_simplified/other/ok)"])
        for n in [x for x in ("rules", "llm", "hybrid", "sipg") if x in rows]:
            for i in rng.sample(range(len(data)), min(100, len(data))):
                r = rows[n][0][i]
                tags = [t for t, c in (("critical_lost", not r["crit_all"]), ("jargon_untouched", (r.get("expert_term_kept") or 0) >= 0.5),
                                       ("low_sim", r["sim"] < 0.7), ("over_simplified_candidate", r["S"] < r["S_src"] - 0.2), ("no_change", not r["changed"])) if c]
                w.writerow([n, data[i]["id"], data[i]["src"], r["out"], data[i]["ref"], ",".join(tags), ""])
    rep += ["## C4. Failure taxonomy\n", f"Sample of 100 outputs per system written to `{a.out}/failure_sample.csv` with automatic pre-tags; the final labels are manual.\n"]
    # ---------------------------------------------------------------- D1 stress
    st_names = [n for n in sysn if n not in ("copy", "reference")]
    stress = stress_eval(S, st_names, ckpts)
    rep.append("## D1. Critical-span stress set (120 items; preserved / total, 95% Wilson CI)\n")
    body = []
    for n, d in stress.items():
        body.append([n] + [f"{d[c][0]}/{d[c][1]}" for c in M.CRIT_CATEGORIES] + [f"{sum(v[0] for v in d.values())}/{sum(v[1] for v in d.values())}"])
    rep += [md_table(["system"] + list(M.CRIT_CATEGORIES) + ["total"], body), ""]
    stress_ok = {n: all(v[0] == v[1] for v in d.values()) for n, d in stress.items()}
    # ---------------------------------------------------------------- D3
    nli = S.nli
    d3 = d3_recall(a, nli)
    rep += ["## D3. Verifier recall on semantic drift (42 pairs) and false rejection of faithful rewrites (24 pairs)\n",
            md_table(["config", "recall", "false rejection"], [[r["config"], f"{pct(r['recall'])} [{pct(r['recall_ci'][0])}, {pct(r['recall_ci'][1])}]",
                                                              f"{pct(r['fp'])} [{pct(r['fp_ci'][0])}, {pct(r['fp_ci'][1])}]"] for r in d3]), ""]
    if nli is None:
        rep.append("(NLI off: pass `--nli hf` to measure the NLI contribution.)\n")
    # ---------------------------------------------------------------- D4 / G4
    rep.append("## D4. Fallback behaviour\n")
    fb_rows = [(r, ) for r in rows["sipg"][0] if r["final_fallback"]] if "sipg" in rows else []
    rej_rows = [r for r in rows["sipg"][0] if r["n_rejected"] > 0] if "sipg" in rows else []
    fb_pres = float(np.mean([r["crit_all"] for (r,) in fb_rows])) if fb_rows else float("nan")
    rep += [f"- sipg: {len(rej_rows)}/{len(data)} examples had >= 1 rejected (rolled-back) edit; critical-span preservation among them: {pct(np.mean([r['crit_all'] for r in rej_rows])) if rej_rows else 'n/a'}%",
            f"- sipg: {len(fb_rows)}/{len(data)} examples hit the final fallback; output preserved all critical spans in {pct(fb_pres)}%"]
    srcs = [e["src"] for e in data][:60]
    adv = adversarial_check(S, srcs, nli is not None)
    rep += [f"- adversarial injection ({'NLI on' if nli else 'NLI off: final gate unreachable'}): " + "; ".join(
        f"{r.mode} injected {r.injected}/edit-gate {r.edit_gate}/final-gate {r.final_gate}/leaked {r.leaked}" for r in adv), ""]
    adv_fb = any(r.final_gate > 0 for r in adv) and all(r.leaked == 0 for r in adv if r.mode != "antonym_drift" or nli)
    # ---------------------------------------------------------------- E
    rc = risk_coverage_eval(a, data, rows, inst, nli)
    rep.append("## E. Risk-coverage (critical-span error vs coverage)\n")
    if "skipped" in rc:
        rep.append(f"skipped: {rc['skipped']}\n")
    else:
        rep += [f"held-out outputs {rc['n_test']}, error rate {pct(rc['error_rate'])}%, ECE of the calibrated signal {rc['ece_calibrated']:.3f}",
                md_table(["signal", "AURC (lower better)", "error rate at 80% coverage"], [[k, f"{v['aurc']:.4f}", pct(v['risk_at_80'])] for k, v in rc["curves"].items()]),
                f"plot: {rc.get('plot', rc.get('plot_error'))}. Signals not collected here: {rc['missing_signals']} (decoder logprob needs a server that returns logprobs; BERTScore-only needs the model).\n"]
    # ---------------------------------------------------------------- H verdict
    def sig(k):
        return res.get(k) and adj[k] < th["alpha"] and res[k]["diff"] > 0
    verdict = []
    best = [n for n in ("hybrid", "sipg") if n in rows]
    ok1 = any(res.get(f"{n}-copy:sari") and res[f"{n}-copy:sari"]["lo"] > 0 for n in best)
    verdict.append(("Beats no-op", ok1, "SARI vs copy: " + ", ".join(f"{n} {res[f'{n}-copy:sari']['diff']:+.2f} [{res[f'{n}-copy:sari']['lo']:+.2f},{res[f'{n}-copy:sari']['hi']:+.2f}]" for n in best if f"{n}-copy:sari" in res)))
    lh = summary.get("sipg", {}).get("layman_hit", float("nan"))
    verdict.append(("Does the task", lh == lh and lh >= th["layman_hit_min"], f"sipg layman hit {pct(lh)}% (need >= {pct(th['layman_hit_min'])}; copy {pct(summary['copy'].get('layman_hit'))}, reference {pct(summary['reference'].get('layman_hit'))})"))
    sp = stress_ok.get("sipg")
    verdict.append(("Safe", None if sp else False, f"stress set preservation 100%: {sp}; clinician hallucination <= {pct(th['hallucination_max'])}%: NOT EVALUATED (needs a clinician sample); automatic proxy for sipg {pct(summary['sipg'].get('halluc'))}%" if "sipg" in summary else "sipg not run"))
    k1, k2 = "sipg-rules:layman_hit", "sipg-rules:sari"
    verdict.append(("Beats rules", bool(sig(k1) and res.get(k2) and not res[k2]["hi"] < 0), f"layman hit diff {res[k1]['diff']:+.3f} (Holm p {adj[k1]:.3f}); SARI diff {res[k2]['diff']:+.2f} [{res[k2]['lo']:+.2f},{res[k2]['hi']:+.2f}]" if k1 in res and k2 in res else "rules/sipg not both run"))
    g1_, g2_ = "sipg-greedy:sari", "sipg-greedy:layman_hit"
    verdict.append(("RL earns its keep", bool(sig(g1_) or sig(g2_)), "; ".join(f"{k} {res[k]['diff']:+.3f} (Holm p {adj[k]:.3f})" for k in (g1_, g2_) if k in res) or "greedy/sipg not both run"))
    fb_ok = (len(fb_rows) > 0 and fb_pres == 1.0) or adv_fb
    verdict.append(("Fallback works", fb_ok, f"real fallbacks {len(fb_rows)} (preserved {pct(fb_pres)}%); adversarial final-gate fired and nothing leaked: {adv_fb}"))
    rep += ["## H. Pre-registered pass / fail\n", f"primary metric: **{prereg['primary_metric']}** - {prereg['primary_definition']}\n",
            md_table(["claim", "result", "evidence"], [[c, "NOT EVALUATED" if ok is None else "PASS" if ok else "FAIL", ev] for c, ok, ev in verdict]), ""]
    if verdict[4][1] is False:
        rep.append("The RL claim failed. If the other claims pass, report the system as a safe, graph-grounded simplifier with verifier gating, and the RL comparison as a negative result with the diagnostics above. "
                   "(The Holm family here is deliberately conservative: it includes the duplicated primary/secondary metrics and the ablation comparisons.)\n")
    txt = "\n".join(rep)
    open(os.path.join(a.out, "report.md"), "w").write(txt)
    json.dump(dict(summary=summary, paired=res, holm=adj, verdict=[(c, ok) for c, ok, _ in verdict], d3=d3, stress=stress,
                   llm=(self_stats(S)), prereg=prereg["sha256"]), open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(txt)
    print(f"\nreport: {a.out}/report.md  summary: {a.out}/summary.json  rows: {a.out}/rows_*.jsonl")


def self_stats(S):
    return None if S.llm is None else dict(model=S.llm.model, **S.llm.stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prereg", "d3", "gates", "run", "audit"])
    ap.add_argument("--dataset", default="cbasu/Med-EASi")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fit-split", default="train", help="split whose references calibrate the S bands (never the test split)")
    ap.add_argument("--fit-jsonl", default=None)
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--src-field", default="Expert")
    ap.add_argument("--ref-field", default="Simple")
    ap.add_argument("--limit", type=int, default=150, help="0 = all")
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--ckpts", default=",".join(f"runs/v3/policy_seed{i}.pt" for i in range(5)), help="comma-separated; 5 seeds by default (train: scripts/train_policy.py --seeds 0,1,2,3,4 --out runs/v3)")
    ap.add_argument("--llm-url", default="http://localhost:11434/v1")
    ap.add_argument("--llm-model", default="qwen2.5:7b-instruct", help="'none' skips every LLM system")
    ap.add_argument("--llm-key", default=None)
    ap.add_argument("--llm-timeout", type=int, default=120)
    ap.add_argument("--llm-cache", default=None)
    ap.add_argument("--llm-structure", action="store_true", help="opt IN to letting the LLM rewrite embedded clauses (CCOMP/ACL) in the hybrid system; off by default (rules own structure)")
    ap.add_argument("--nli", choices=["none", "hf", "stub"], default="hf", help="default hf: NLI at the final gate (audit item 5); pass none EXPLICITLY to disable")
    ap.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--nli-mode", choices=["entailment", "contradiction", "either"], default="contradiction")
    ap.add_argument("--nli-threshold", type=float, default=0.5)
    ap.add_argument("--nli-contra-threshold", type=float, default=0.5)
    ap.add_argument("--sim", choices=["content", "nli", "bertscore"], default="content", help="instrument Sim (post-hoc only; not in the training reward)")
    ap.add_argument("--sim-agg", choices=["mean", "min"], default="mean")
    ap.add_argument("--bertscore-layers", type=int, default=None)
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--bertscore-model", default="microsoft/deberta-xlarge-mnli")
    ap.add_argument("--out", default="eval_suite")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-prereg", action="store_true")
    ap.add_argument("--audit-labels", default=None)
    a = ap.parse_args()
    {"prereg": stage_prereg, "d3": stage_d3, "gates": stage_gates, "run": stage_run, "audit": stage_audit}[a.stage](a)


if __name__ == "__main__":
    main()
