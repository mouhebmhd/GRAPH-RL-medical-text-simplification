import os, sys, json, subprocess
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pytest
import eval_suite as ES


def test_wilson_is_clamped_and_sane():
    lo, hi = ES.wilson(0, 40)
    assert lo == 0.0 and 0 < hi < 0.15
    assert ES.wilson(40, 40)[1] == 1.0 and ES.wilson(0, 0)[0] != ES.wilson(0, 0)[0]


def test_triage_flags_format_paraphrase_and_leaves_real_losses_for_review():
    assert ES.triage("Take 1,000 mg daily.", "Take 1000 mg a day.", ["number lost: 1,000"])[0] == "likely_FP:number_format"
    assert ES.triage("Take 2 tablets.", "Take two tablets.", ["number lost: 2"])[0] == "likely_FP:spelled_number"
    assert ES.triage("Take 2 tablets.", "Take tablets.", ["number lost: 2"])[0] == "review:number_dropped"
    assert ES.triage("He has no pain.", "He cannot feel pain.", ["negation dropped"])[0] == "likely_FP:negation_paraphrase"
    assert ES.triage("He has no pain.", "He has pain.", ["negation dropped"])[0] == "review:negation_dropped"
    assert ES.triage("Possible flu.", "It might be flu.", ["hedge dropped"])[0] == "likely_FP:hedge_paraphrase"


def test_prereg_is_frozen_and_hash_matches(tmp_path):
    import argparse
    a = argparse.Namespace(out=str(tmp_path), force=False)
    ES.stage_prereg(a)
    p = json.load(open(tmp_path / "prereg.json"))
    assert p["primary_metric"] == "safe_layman_hit" and p["thresholds"]["layman_hit_min"] == 0.65 and len(p["sha256"]) == 64
    with pytest.raises(SystemExit):
        ES.stage_prereg(a)                                            # a second call must refuse to overwrite


def test_summarise_labels(tmp_path):
    f = tmp_path / "l.csv"
    f.write_text("id,label\n1,a\n2,b\n3,c\n4,a\n5,\n")
    s = ES.summarise_labels(str(f), {})
    assert (s["n"], s["a"], s["b"], s["c"]) == (4, 2, 1, 1) and s["fp_share"] == 0.5


def test_full_run_stage_smoke_without_llm(tmp_path):
    """Whole pipeline on 12 synthetic pairs, rule realizer only (no LLM server needed)."""
    from sipg_rl.data import synth
    from sipg_rl.pipeline import SIPGRL, default_parts
    m = SIPGRL(default_parts())
    rows = [dict(Expert=t, Simple=m.simplify(t, apply_all=True).text, expert_terms="['hypertension']", layman_terms="['high blood pressure']") for t in synth(30, 21)]
    f = tmp_path / "d.jsonl"; f.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "ev"
    base = [sys.executable, os.path.join(ROOT, "scripts", "eval_suite.py")]
    flags = ["--jsonl", str(f), "--split", "test", "--fit-jsonl", str(f), "--llm-model", "none", "--nli", "stub", "--out", str(out),
             "--ckpts", os.path.join(ROOT, "sipg_policy_v2.pt"), "--limit", "12", "--systems", "copy,rules,greedy,sipg,sipg_noverif,reference"]
    assert subprocess.run(base + ["prereg", "--out", str(out)], capture_output=True, text=True).returncode == 0
    r = subprocess.run(base + ["run"] + flags, capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = (out / "report.md").read_text()
    for h in ("## B. Core comparison", "## C1.", "## C2.", "## C3.", "## D1.", "## D3.", "## D4.", "## E.", "## H. Pre-registered"):
        assert h in rep, h
    summ = json.load(open(out / "summary.json"))
    assert summ["summary"]["copy"]["sari"] < summ["summary"]["rules"]["sari"]
    assert dict(summ["verdict"])["Safe"] is None                        # clinician hallucination criterion must stay NOT EVALUATED
    assert (out / "failure_sample.csv").exists() and (out / "audit_gold.csv").exists() is False   # audit file belongs to `gates`
    g = subprocess.run(base + ["gates"] + flags, capture_output=True, text=True, timeout=280)
    assert g.returncode == 0, g.stderr[-2000:]
    assert (out / "audit_gold.csv").exists() and "G4 fallback executes" in (out / "gates.md").read_text()
