import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import collections, pytest
from sipg_rl import stress, drift, adversarial as A
from sipg_rl.verifier import FidelityVerifier, NullVerifier
from sipg_rl.pipeline import SIPGRL, default_parts


@pytest.fixture(scope="module")
def nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def test_stress_set_is_balanced_deterministic_and_self_consistent():
    s1, s2 = stress.build_stress_set(), stress.build_stress_set()
    assert s1 == s2 and len(s1) == 120
    assert collections.Counter(i.category for i in s1) == {c: 20 for c in ("numbers", "units", "negations", "hedges", "conditionals", "drug_names")}
    assert all(stress.check_item(i, i.text) for i in s1)                        # an unchanged sentence passes every item
    assert len({i.id for i in s1}) == 120


def test_stress_checker_detects_each_category_failure():
    by = {}
    for i in stress.build_stress_set():
        by.setdefault(i.category, i)                  # first item of each category
    assert not stress.check_item(by["numbers"], by["numbers"].text.replace("2.5", "3"))
    assert not stress.check_item(by["drug_names"], by["drug_names"].text.replace("warfarin", "aspirin"))
    assert not stress.check_item(by["negations"], by["negations"].text.replace("no ", ""))
    assert not stress.check_item(by["hedges"], by["hedges"].text.replace("Possible ", ""))
    assert not stress.check_item(by["conditionals"], by["conditionals"].text.replace(" if ", " when "))
    assert not stress.check_item(by["units"], by["units"].text.replace("mg", "doses"))


def test_verified_system_passes_the_stress_set_and_the_unverified_ablation_does_not(nlp):
    items = [i for i in stress.build_stress_set() if i.category in ("numbers", "negations", "drug_names")][:30]
    mk = lambda ver, mode: SIPGRL(default_parts(nlp=nlp, verifier=ver, realizer=A.AdversarialRealizer(mode)))
    for mode in ("drop_number", "drop_negation", "swap_drug"):
        guarded, bare = mk(None, mode), mk(NullVerifier(), mode)
        g = [stress.check_item(i, guarded.simplify(i.text, apply_all=True).text) for i in items if A.applies(mode, i.text)]
        b = [stress.check_item(i, bare.simplify(i.text, apply_all=True).text) for i in items if A.applies(mode, i.text)]
        assert g and all(g), mode                    # with the verifier: zero failures
        assert not all(b), mode                      # ablation: the corruption reaches the output


def test_drift_sets_are_well_formed_and_mostly_invisible_to_the_lexical_verifier():
    assert len(drift.DRIFT) == 42 and len(drift.FAITHFUL) == 24
    assert {k for k, _, _ in drift.DRIFT} == {"certainty", "quantifier", "causality", "temporal", "direction", "anatomy", "polarity"}
    v = FidelityVerifier()
    recall = sum(not v.check(s, d, use_nli=False).ok for _, s, d in drift.DRIFT) / len(drift.DRIFT)
    fp = sum(not v.check(s, f, use_nli=False).ok for s, f in drift.FAITHFUL) / len(drift.FAITHFUL)
    assert recall < 0.2, recall                      # the gap NLI is supposed to close
    assert fp < 0.1, fp


def test_nli_modes_entailment_contradiction_either():
    nli = lambda p, h: {"entailment": 0.3, "neutral": 0.6, "contradiction": 0.1}     # 'neutral': an explanation was added
    ok = lambda mode: FidelityVerifier(nli=nli, nli_mode=mode).check("a b c", "a b c d", use_nli=True).ok
    assert not ok("entailment") and ok("contradiction") and not ok("either")
    contra = lambda p, h: {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9}
    assert not FidelityVerifier(nli=contra, nli_mode="contradiction").check("a b", "a c", use_nli=True).ok


def test_greedy_driver_override_runs_and_is_a_real_alternative_policy(nlp):
    from sipg_rl.baselines import GreedyReadabilityPolicy
    m = SIPGRL(default_parts(nlp=nlp))
    t = "The hypertensive patient reports dyspnea and takes lisinopril 10 mg BID."
    g = m.simplify(t, driver=GreedyReadabilityPolicy())
    assert g.text != t and "lisinopril 10 mg" in g.text
