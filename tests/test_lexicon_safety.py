import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from sipg_rl import lexicon as L


def test_shipped_lexicon_is_safe_and_every_entry_is_lowercase():
    L.assert_lexicon_safe()
    assert all(k == k.lower() for k in L.PLAIN_TERMS)


@pytest.mark.parametrize("key,plain,cat", [
    ("warfarin", "blood thinner", "term"),            # drug name as key
    ("anticoagulant", "warfarin", "term"),            # drug name as replacement
    ("dose", "5 mg", "term"),                         # number / unit
    ("hypertension", "high bp 140/90", "term"),       # number
    ("procedure", "CPT 99213", "term"),               # code
    ("htn", "high blood pressure 20%", "term"),       # percentage
    ("x", "take 2 tablets", "term"),                  # unit word
])
def test_unsafe_entries_are_refused(key, plain, cat):
    assert L.lexicon_entry_problem(key, plain, cat)


def test_safe_entries_pass_and_dose_abbreviations_stay_allowed():
    assert L.lexicon_entry_problem("dyspnea", "trouble breathing") is None
    assert L.lexicon_entry_problem("bid", "twice a day", "abbr") is None
    with pytest.raises(ValueError):
        L.assert_lexicon_safe({"lisinopril": ("blood pressure pill", "term")})


def test_new_entries_work_end_to_end_with_the_verifier():
    import spacy
    from sipg_rl.pipeline import SIPGRL, default_parts
    m = SIPGRL(default_parts(nlp=spacy.load("en_core_web_sm")))
    r = m.simplify("The patient had epistaxis and myalgia, and the efficacy of aspirin 81 mg was reviewed.", apply_all=True)
    assert "nosebleed" in r.text and "muscle pain" in r.text and "aspirin 81 mg" in r.text
