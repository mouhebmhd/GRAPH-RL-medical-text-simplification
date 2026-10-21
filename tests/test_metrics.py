import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sipg_rl import metrics as M

SRC = "The patient exhibited tachycardia and syncope after administration of 10 mg."
REF = "The patient had a fast heartbeat and fainted after getting 10 mg."


def test_sari_perfect_and_copy():
    assert M.sari(SRC, REF, REF)["sari"] > 99.9                      # output == reference
    cp = M.sari(SRC, SRC, REF)
    assert cp["keep"] > 0 and cp["add"] < 5 and cp["sari"] < M.sari(SRC, REF, REF)["sari"]
    assert M.sari(SRC, SRC, SRC)["sari"] > 99.9                      # nothing to change, nothing changed


def test_sari_rewards_good_edits_over_bad_edits():
    good = "The patient had a fast heartbeat and fainted after administration of 10 mg."
    bad = "The patient banana tachycardia and syncope after administration of 10 mg."
    # SARI is not monotone in "badness": a junk insertion that also deletes reference-deleted words can beat copying.
    assert M.sari(SRC, good, REF)["sari"] > M.sari(SRC, SRC, REF)["sari"]
    assert M.sari(SRC, good, REF)["sari"] > M.sari(SRC, bad, REF)["sari"] + 10


def test_rouge_l_bounds():
    assert M.rouge_l(REF, REF) == 100.0 and M.rouge_l("", REF) == 0.0 and 0 < M.rouge_l(SRC, REF) < 100


def test_readability_direction():
    hard = "Administration of anticoagulant pharmacotherapy necessitates comprehensive hematological surveillance."
    easy = "You take a blood thinner. We check your blood often."
    assert M.fkgl(hard) > M.fkgl(easy) + 5 and M.ari(hard) > M.ari(easy)


def test_independent_safety_checks():
    assert M.numbers_kept("Take 10 mg and 2.5 mg", "Take 2.5 mg and 10 mg") and not M.numbers_kept("Take 10 mg", "Take 1 mg")
    assert M.negations_kept("no chest pain", "no chest pain") and not M.negations_kept("no chest pain", "chest pain")


def test_parse_terms_is_forgiving():
    assert M.parse_terms("['hypertension', 'edema']") == ["hypertension", "edema"]
    assert M.parse_terms('{"renal impairment": "x"}') == ["renal impairment"]
    assert M.parse_terms("hypertension; edema, ab") == ["hypertension", "edema"]
    assert M.parse_terms(None) == [] and M.term_retention([], "x") is None
    assert M.term_retention(["edema", "renal"], "Edema was seen") == 0.5


def test_bootstrap_ci_contains_mean_and_detects_shift():
    x = np.random.default_rng(0).normal(1.0, 1.0, 200)
    m, lo, hi = M.mean_ci(x)
    assert lo < m < hi
    d, lo, hi = M.paired_diff_ci(x + 0.5, x)
    assert abs(d - 0.5) < 1e-9 and lo > 0.49
