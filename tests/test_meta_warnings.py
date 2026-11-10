import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest, torch
from sipg_rl.pipeline import SIPGRL, default_parts
from sipg_rl.model import SIPGPolicy
from sipg_rl.realize import RoutedRealizer, RuleRealizer


@pytest.fixture(scope="module")
def nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def ckpt(tmp_path, meta=None):
    p = str(tmp_path / "p.pt")
    torch.save(SIPGPolicy().state_dict(), p)
    if meta is not None:
        json.dump(meta, open(p + ".meta.json", "w"))
    return p


GOOD = dict(realizer_in_loop="rules", scorer="BandSuitabilityScorer", encoder="ContentEncoder", beta=0.0)


def test_legacy_checkpoint_without_meta_warns(tmp_path, nlp):
    with pytest.warns(UserWarning, match="LEGACY reward"):
        SIPGRL(default_parts(nlp=nlp), checkpoint=ckpt(tmp_path))


def test_matching_checkpoint_is_silent(tmp_path, nlp):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SIPGRL(default_parts(nlp=nlp), checkpoint=ckpt(tmp_path, GOOD))


def test_rules_trained_policy_deployed_with_llm_is_flagged_as_inference_time_augmentation(tmp_path, nlp):
    parts = default_parts(nlp=nlp, realizer=RoutedRealizer(lambda p, t: t, RuleRealizer()))
    with pytest.warns(UserWarning, match="inference-time augmentation"):
        SIPGRL(parts, checkpoint=ckpt(tmp_path, GOOD))


def test_reward_component_mismatch_is_flagged(tmp_path, nlp):
    with pytest.warns(UserWarning, match="encoder=ConceptEncoder"):
        SIPGRL(default_parts(nlp=nlp), checkpoint=ckpt(tmp_path, dict(GOOD, encoder="ConceptEncoder")))


def test_checkpoint_trained_with_sim_in_the_reward_is_flagged(tmp_path, nlp):
    """Audit item 1: policies trained with beta=0.5 (Sim in the objective) are not comparable to the S-only reward."""
    with pytest.warns(UserWarning, match="beta=0.5"):
        SIPGRL(default_parts(nlp=nlp), checkpoint=ckpt(tmp_path, dict(GOOD, beta=0.5)))
