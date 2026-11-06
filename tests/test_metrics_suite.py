import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sipg_rl import metrics as M

SRC = "No chest pain was reported; take warfarin 5 mg daily if dyspnea occurs, possible pneumonia."


def test_critical_report_per_category():
    ok = M.critical_report(SRC, SRC)
    assert all(ok[c] is True for c in M.CRIT_CATEGORIES)
    r = M.critical_report(SRC, SRC.replace("5 mg", "").replace("No ", "").replace("possible ", "").replace(" if ", " when ").replace("warfarin", "aspirin"))
    assert r == dict(numbers=False, units=False, negations=False, hedges=False, conditionals=False, drug_names=False)
    assert M.critical_report("The heart is fine.", "The heart is fine.")["numbers"] is None


def test_hallucination_proxy_flags_new_numbers_and_drugs_only():
    assert not M.hallucination_proxy(SRC, SRC) and M.hallucination_proxy(SRC, SRC + " Also 40 mg.")
    assert M.hallucination_proxy(SRC, SRC + " Add insulin.")


def test_smog_increases_with_polysyllables():
    assert M.smog("Hematological pharmacokinetic considerations necessitate comprehensive surveillance.") > M.smog("You take a pill. We check you.")


def test_edit_type_taxonomy_applies_to_any_pair():
    s = "The patient had epistaxis and was given a medicine."
    assert M.edit_type_counts(s, s) == dict(replacement=0, deletion=0, insertion=0, elaboration=0)
    assert M.edit_type_counts(s, "The patient had a nosebleed and was given a medicine.")["replacement"] == 1
    assert M.edit_type_counts(s, "The patient had epistaxis and was given a medicine")["deletion"] == 0
    assert M.edit_type_counts(s, "The patient had epistaxis and was given.")["deletion"] == 1
    assert M.edit_type_counts(s, s + " This means a nosebleed from the nose.")["elaboration"] == 1
    assert M.annotation_label_counts("Replacement: x; Elaboration: y; replacement")["replacement"] == 2


def test_paired_bootstrap_and_holm():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 300)
    r = M.paired_bootstrap(x + 0.4, x)
    assert r["lo"] > 0.39 and r["p"] < 0.01 and r["dz"] > 1
    null = M.paired_bootstrap(x, x + rng.normal(0, 1e-9, 300))
    assert null["p"] > 0.05 or null["lo"] <= 0 <= null["hi"]
    adj = M.holm({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adj["a"] == 0.03 and adj["b"] == 0.04 and adj["c"] == 0.5 and all(adj[k] >= p for k, p in {"a": .01, "b": .02, "c": .5}.items())


def test_risk_coverage_rewards_a_good_signal_and_ece_detects_miscalibration():
    rng = np.random.default_rng(1)
    err = (rng.random(2000) < 0.2).astype(float)
    good = -err + rng.normal(0, 0.3, 2000)          # lower confidence when wrong
    junk = rng.normal(0, 1, 2000)
    assert M.risk_coverage(good, err)[2] < M.risk_coverage(junk, err)[2] - 0.03
    assert M.risk_at_coverage(good, err, 0.5) < M.risk_at_coverage(junk, err, 0.5)
    assert M.risk_at_coverage(good, err, 1.0) == np.mean(err)
    assert M.ece(np.full(2000, 0.2), err) < 0.04 and M.ece(np.full(2000, 0.8), err) > 0.5


def test_logreg_learns_a_separable_signal():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (600, 3)); y = (X[:, 0] + 0.2 * rng.normal(0, 1, 600) > 0.5).astype(float)
    m = M.LogReg().fit(X[:400], y[:400])
    p = m.predict(X[400:])
    assert ((p > 0.5) == (y[400:] > 0.5)).mean() > 0.9
