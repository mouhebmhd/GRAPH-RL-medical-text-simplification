import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from sipg_rl import metrics as M
from sipg_rl.stats import paired_bootstrap, holm, mean_std_across_seeds, mean_ci
from sipg_rl.selective import risk_coverage, ece, aurc, auroc, fit_logistic
from sipg_rl.pipeline import SIPGRL, default_parts
from sipg_rl.trail import build_trail_record
from sipg_rl.graph import flatten_graph
from sipg_rl.env import make_env
from sipg_rl.model import FlatSequencePolicy, collate

SRC = "The hypertensive patient, who takes lisinopril 10 mg BID, reports no chest pain after endoscopy."
OUT = "The patient who has high blood pressure reports no chest pain after endoscopy. The patient takes lisinopril 10 mg twice a day."


def test_stats_paired_bootstrap_and_holm():
    rng = np.random.default_rng(0); a = rng.normal(0.3, 0.1, 80); b = rng.normal(0.0, 0.1, 80)
    d, lo, hi, p = paired_bootstrap(a, b)
    assert lo > 0 and p < 0.01 and abs(d - 0.3) < 0.05
    d0, lo0, hi0, p0 = paired_bootstrap(b, b + rng.normal(0, 0.1, 80))
    assert lo0 < 0 < hi0
    adj = holm({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.04) and adj["c"] == pytest.approx(0.5)
    m, sd, hw = mean_std_across_seeds([1, 2, 3]); assert m == 2 and sd == pytest.approx(1.0) and hw > 0
    assert mean_ci([1, 2, 3, None])[0] == 2


def test_diagnostic_metrics():
    P = default_parts(); g = P["builder"].build_text(SRC)
    lb = M.layman_hit_by_node_type(SRC, OUT, g)
    assert lb["LEX"] == 1.0 and lb["AMOD"] == 1.0 and lb["clause"] == 1.0
    assert M.jargon_kept_by_category(SRC, OUT)["drug"] == 1.0 and M.jargon_kept_by_category(SRC, OUT)["abbrev"] == 0.0
    assert all(v in (True, None) for v in M.critical_preservation_by_category(SRC, OUT).values())
    bad = M.critical_preservation_by_category(SRC, "The patient takes lisinopril 100 mg and has chest pain.")
    assert bad["number"] is False and bad["negation"] is False and bad["drug"] is True
    et = M.edit_type_distribution(SRC, OUT); assert abs(sum(et.values()) - 1) < 1e-9
    assert M.target_attainment([5, 6, 7, 9, 8], 6, 8) == 0.6
    assert M.nli_entailment(SRC, OUT, None) is None and M.nli_disagreement(SRC, OUT, lambda p, h: {"entailment": 0.9 if p == SRC else 0.1}) is True
    assert 0 <= M.bertscore(SRC, OUT)["f1"] <= 1 and M.d_sari([SRC, SRC], [OUT, OUT], [OUT, OUT])["sari"] > 90


def test_selective_prediction():
    y = np.array([1] * 8 + [0] * 2); good = np.array([.9] * 8 + [.1] * 2)
    cv = risk_coverage(good, y, [0.2, 0.8, 1.0]); assert cv[0][1] == 0 and cv[1][1] == 0 and cv[2][1] == pytest.approx(0.2)
    assert auroc(good, y) == 1.0 and aurc(risk_coverage(good, y)) < aurc(risk_coverage(1 - good, y))
    assert ece(np.array([.8] * 5), np.array([1, 1, 1, 1, 0])) == pytest.approx(0.0)
    X = np.r_[np.random.default_rng(0).normal(1, .5, (60, 1)), np.random.default_rng(1).normal(-1, .5, (60, 1))]
    yy = np.r_[np.ones(60), np.zeros(60)]; p = fit_logistic(X, yy)(X); assert auroc(p, yy) > 0.95


def test_trail_record_schema_and_provenance():
    m = SIPGRL(default_parts(), checkpoint=os.path.join(os.path.dirname(__file__), "..", "sipg_policy_v3.pt"))
    r = m.simplify(SRC, apply_all=True)
    rec = build_trail_record("x1", "rules", SRC, r.text, r)
    assert {"id", "system", "source", "output", "traversal", "final_verdict", "fallback"} <= set(rec)
    inv = [t for t in rec["traversal"] if t["action"] == "INVOKE"]
    assert inv and all({"prompt", "candidate_used", "candidate_rank", "candidate_count", "realizer", "verifier", "S_before", "S_after"} <= set(t) for t in inv)
    lex = next(t for t in inv if t["kind"] == "LEX")
    assert lex["candidate_used"] == "twice a day" and lex["realizer"] == "rules" and lex["verifier"] == "pass" and lex["S_after"] >= lex["S_before"]
    assert json.dumps(rec, default=float)


def test_flat_graph_ablation_pieces():
    P = default_parts(); g = P["builder"].build_text(SRC); f = flatten_graph(g)
    assert [n.anchor for n in f.nodes] == sorted(n.anchor for n in f.nodes) and set(f.adj[f.adj >= 0].ravel().tolist()) <= {7, 8, 9}
    env = make_env(P); ob = env.reset(f); assert env.cur == f.start
    lg, v = FlatSequencePolicy()(collate([ob, ob])); assert lg.shape[0] == 2 and v.shape == (2,)


def test_run_eval_has_all_eight_systems():
    src = open(os.path.join(os.path.dirname(__file__), "..", "scripts", "run_eval.py")).read()
    assert all(s in src.split("ALL = ")[1].split("\n")[0] for s in ("copy", "random", "invoke_all", "greedy", "rules", "llm", "hybrid", "sipg"))
