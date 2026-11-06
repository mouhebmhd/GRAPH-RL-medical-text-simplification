"""Regression tests for the round-3 audit fixes (numbers, S-only reward, swap_drug injector, edit-type ownership, mining)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from sipg_rl import numnorm, adversarial as A, mining
from sipg_rl.verifier import FidelityVerifier
from sipg_rl.env import EnvConfig
from sipg_rl.realize import lexical_guard, LLM_KINDS
from sipg_rl.llm import sanitize

V = FidelityVerifier()
OK = [("Take 50mg daily.", "Take 50 mg every day."), ("Give 10 mg.", "Give 10 milligrams."), ("Treat for 7 days.", "Use it for a 7-day period."),
      ("Dose is 1,000 mg.", "Dose is 1000 mg."), ("Take 2 tablets.", "Take two tablets."), ("Take 0.5 mg.", "Take .5 mg."),
      ("Give 5 mL.", "Give 5 milliliters."), ("Take 1/2 tablet.", "Take half a tablet."), ("Take 5 mg twice a day.", "Take 5 mg 2 times a day."),
      ("BP was 20%.", "BP was 20 percent."), ("COVID-19 is present.", "COVID is present.")]
BAD = [("Take 10 mg.", "Take 20 mg."), ("Take 10 mg.", "Take 10 g."), ("Take 10 mg.", "Take the pill."), ("Take 10 mg.", "Take 10 tablets."),
       ("Take 5 mg.", "Take 5 mg and 500 mg."), ("Dose 20%.", "Dose 20."), ("Take 5 mg twice a day.", "Take 5 mg 3 times a day.")]


@pytest.mark.parametrize("s,o", OK)
def test_notation_differences_are_not_violations(s, o):
    assert V.check(s, o, use_nli=False).ok


@pytest.mark.parametrize("s,o", BAD)
def test_real_number_changes_still_rejected(s, o):
    assert not V.check(s, o, use_nli=False).ok


def test_legacy_flag_reproduces_old_false_rejection():
    assert not FidelityVerifier(legacy_numbers=True).check("Give 10 mg.", "Give 10 milligrams.", use_nli=False).ok


def test_diagnose_names_the_cause():
    d = numnorm.diagnose("Dose is 1,000 mg.", "Dose is 1000 mg.")
    assert d and d[0]["status"] == "format_artefact" and d[0]["cause"] == "thousands separator"


def test_training_reward_is_S_alone():
    assert EnvConfig().beta == 0.0


def test_swap_drug_injector_applies_beyond_the_old_eight_drugs():
    assert len(A._SWAP) > 40
    assert A.corrupt("swap_drug", "Give tacrolimus 1 mg.") != "Give tacrolimus 1 mg."
    assert all(A.applies("swap_drug", s) for s in A.synthetic_sources(12))


def test_selftest_never_reports_zero_injected_for_swap_drug():
    from sipg_rl.pipeline import SIPGRL, default_parts
    import spacy
    nlp = spacy.load("en_core_web_sm")
    rows = A.selftest(lambda rz: SIPGRL(default_parts(nlp=nlp, realizer=rz, nli=None)), ["No drug is named here."], modes=("swap_drug",), min_injected=9)
    r = rows[0]
    assert r.injected >= 9 and r.n_synthetic >= 9
    assert r.edit_gate >= 9 and r.leaked == 0


def test_llm_is_lexical_only_and_cannot_compress():
    assert "CCOMP" not in LLM_KINDS and "ACL" not in LLM_KINDS
    assert not lexical_guard("chronic obstructive pulmonary disease", "lung")
    assert lexical_guard("hypertensive urgency", "very high blood pressure")
    assert sanitize("ok", "a long clinical phrase here") == ""


def test_mining_counts_filters_and_is_train_only(tmp_path):
    rows = [dict(id=i, src="The patient had quuxolithiasis and used warfarin 5 mg.", ref="The patient had dry mouth and used warfarin 5 mg.", raw={})
            for i in range(3)]
    rows += [dict(id=9, src="Edema was noted.", ref="Swelling was noted.", raw={})]
    ent, rep = mining.mine(rows, "train", min_count=2)
    assert ent["quuxolithiasis"]["plain"] == "dry mouth" and "edema" not in ent      # edema below min_count
    assert not any("warfarin" in k for k in ent)                                       # drug names never mined
    rec = mining.recall_at_k(rows[:3], ent)
    assert rec["ALL"]["exact@1"] == 1.0
    p = tmp_path / "lex.json"; mining.save(ent, rep, str(p))
    from sipg_rl import lexicon
    info = lexicon.load_mined_lexicon(str(p))
    assert info["added"] >= 1 and info["split"] == "train"
    lexicon.PLAIN_TERMS.pop("quuxolithiasis", None)
