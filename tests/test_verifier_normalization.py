import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from sipg_rl.verifier import FidelityVerifier
from sipg_rl.normalize import normalize_number_unit, number_multiset
from sipg_rl import metrics as M

V = FidelityVerifier()


@pytest.mark.parametrize("a,b", [("Take 50mg daily.", "Take 50 mg daily."), ("Take 50 mg daily.", "Take 50 milligrams daily."),
                                 ("Take 50mg daily.", "Take 50 Milligrams daily."), ("Up to 4,000 mg a day.", "Up to 4000 mg a day."),
                                 ("Give 0.50 mg.", "Give 0.5 mg."), ("Give 500 mcg.", "Give 500 µg."), ("Give 500 mcg.", "Give 500 micrograms."),
                                 ("Give 5 mL now.", "Give 5 milliliters now."), ("Take 2 tablets.", "Take 2 tablet."),
                                 ("Wait 24 hours.", "Wait 24 hrs.")])
def test_equivalent_forms_have_identical_number_multisets(a, b):
    assert number_multiset(a) == number_multiset(b)
    assert V.check(a, b, use_nli=False).ok, V.check(a, b, use_nli=False).violations
    assert V.check(b, a, use_nli=False).ok


def test_thousands_not_split():
    assert normalize_number_unit("1,204 adults") == [(1204.0, "")]
    assert V.check("A study of 1,204 adults.", "A study of 1204 adults.", use_nli=False).ok
    assert M.numbers_kept("A study of 1,204 adults.", "A study of 1204 adults.")


@pytest.mark.parametrize("a,b", [("Take 5 mg daily.", "Take 50 mg daily."), ("Take 5 mg daily.", "Take 5 g daily."),
                                 ("Take 5 mg daily.", "Take 5 mcg daily."), ("Take 500 mg.", "Take 500 mcg."),
                                 ("Take 5 mg daily.", "Take mg daily."), ("Take 4,000 mg.", "Take 400 mg.")])
def test_real_changes_are_still_caught(a, b):
    assert not V.check(a, b, use_nli=False).ok


def test_unitless_source_number_matches_any_unit():
    assert V.check("Take 2 daily.", "Take 2 tablets daily.", use_nli=False).ok       # pre-existing behaviour kept
