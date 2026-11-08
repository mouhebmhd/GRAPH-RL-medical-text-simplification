import os, sys, json, threading, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pytest
from sipg_rl.checks import check_all, CATEGORIES
from sipg_rl.sanitize import accept_span_rewrite
from sipg_rl.select import select_candidate
from sipg_rl.mined import Blocklist, MinedLexicon, audit_lexicon, load_mined
from sipg_rl.stats import holm_bonferroni, paired_bootstrap
from sipg_rl.pipeline import default_parts
from sipg_rl.env import make_env, INVOKE
from sipg_rl.realize import CopyLexiconRealizer, RuleRealizer
from sipg_rl import lexicon


def test_check_all_per_category():
    ok = check_all("Take 5 mg of warfarin BID. Do not drive if dizzy; it may cause bleeding.", "Take 5 mg of warfarin twice a day. Do not drive if dizzy; it may cause bleeding.")
    assert ok["all_pass"] and all(ok[c]["pass"] for c in CATEGORIES if ok[c]["applicable"])
    bad = check_all("Take 5 mg of warfarin BID. Do not drive if dizzy; it may cause bleeding.", "Take 50 mg of metformin. Drive. It causes bleeding.")
    assert not bad["all_pass"]
    assert bad["number"]["pass"] is False and "5" in bad["number"]["detail"]["lost"] and bad["negation"]["pass"] is False
    assert bad["hedge"]["pass"] is False and bad["conditional"]["pass"] is False and bad["drug"]["pass"] is False
    assert bad["drug"]["detail"]["lost"] == ["warfarin"] and bad["drug"]["detail"]["hallucinated"] == ["metformin"] and bad["abbrev"]["pass"] is False


@pytest.mark.parametrize("new,ok", [("heart muscle", True), ("", False), ("renal", False), ("a long explanation of the whole thing here", False),
                                    ("kidney, which filters blood", False), ("kidney. Also", False), ("kidney 5", False)])
def test_sanitizer(new, ok):
    assert accept_span_rewrite("renal", new)[0] is ok


def test_s_ranked_selection_with_count_tiebreak():
    S = {"a": 0.5, "b": 0.7, "c": 0.7}
    assert select_candidate([("a", 9), ("b", 1), ("c", 5)], S.get)[0] == "c"      # b ties c on S, c has the higher count
    assert select_candidate([("a", 9), ("b", 1)], S.get)[:3] == ("b", 1, 2)


def test_rule_realizer_uses_s_ranking_over_multiple_candidates(monkeypatch):
    monkeypatch.setitem(lexicon.CANDIDATES, "dyspnea", [("breathing difficulty and shortness of breath problems", 50), ("trouble breathing", 1)])
    parts = default_parts(); g = parts["builder"].build_text("The patient has dyspnea.")
    env = make_env(parts); env.reset(g); env.cur = next(n.id for n in g.nodes if n.meta.get("key") == "dyspnea"); env.step(INVOKE)
    assert "trouble breathing" in env.text and env.trail[-1]["candidate"]["candidate_rank"] == 1 and env.trail[-1]["candidate"]["n_candidates"] == 2


def test_blocklist_and_audit_never_admit_unsafe_entries(tmp_path):
    bl, lex = Blocklist(), MinedLexicon()
    bad_rows = [("Give warfarin now.", "Give blood thinner now."), ("Dose is 5 mg daily.", "Dose is five milligrams daily."),
                ("Code I10 applies.", "Code for high blood pressure applies."), ("Take it bid.", "Take it twice a day."),
                ("There was no edema.", "There was swelling."), ("Possible edema was seen.", "Swelling was seen.")]
    for s, r in bad_rows:
        lex.add_row(s, r, block=bl)
    assert not lex.cands                                                  # nothing unsafe entered
    for _ in range(3):
        lex.add_row("Prior to dosing, check hepatic function.", "Before dosing, check liver function.", ["hepatic"], ["liver"], bl)
    p = tmp_path / "ok.json"; lex.save(p)
    assert audit_lexicon({j: lex.ranked(j) for j in lex.cands}) == []
    d = json.load(open(p)); d["candidates"]["warfarin"] = [["blood thinner", 9]]; json.dump(d, open(tmp_path / "bad.json", "w"))
    with pytest.raises(ValueError, match="blocklist"):
        load_mined(str(tmp_path / "bad.json"))
    saved = dict(lexicon.PLAIN_TERMS), set(lexicon.TIER2)
    try:
        assert load_mined(str(p), min_count=2) == 1 and lexicon.PLAIN_TERMS["hepatic"][0] == "liver" or "hepatic" in lexicon.PLAIN_TERMS
    finally:
        lexicon.PLAIN_TERMS.clear(); lexicon.PLAIN_TERMS.update(saved[0]); lexicon.TIER2.clear(); lexicon.TIER2.update(saved[1]); lexicon.CANDIDATES.clear()


def test_copy_lexicon_realizer_only_does_top1_lexical_swaps():
    parts = default_parts(realizer=CopyLexiconRealizer())
    g = parts["builder"].build_text("The hypertensive patient, who takes lisinopril 10 mg BID, reports no chest pain.")
    env = make_env(parts); env.reset(g)
    for n in g.nodes:
        env.cur = n.id
        if env.invoked[n.id] == 0: env.step(INVOKE)
    assert "twice a day" in env.text and "hypertensive" in env.text and ", who takes" in env.text      # LEX only; AMOD + clause untouched


def test_holm_and_paired_bootstrap_resamples_by_example():
    r = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.06}); assert r["a"]["reject"] and r["b"]["reject"] and not r["c"]["reject"]
    assert holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.04})["c"]["adj_p"] == 0.04
    rng = np.random.default_rng(0); base = rng.normal(0, 5, 100); a, b = base + 0.3, base            # constant paired diff, huge between-example variance
    d, lo, hi, p = paired_bootstrap(a, b)
    assert hi - lo < 0.01 and lo > 0.29                                    # per-resample paired diffs: CI collapses (independent CIs would be ~+-1)


def test_truncation_is_tracked():
    parts = default_parts(); sent = "Findings: " + ", ".join(["edema", "dyspnea", "syncope", "pruritus", "erythema", "emesis", "pyrexia", "tachycardia", "bradycardia"] * 4) + "."
    g = parts["builder"].build_text(sent)
    assert g.n_total > g.n == 24 and g.truncated == g.n_total - 24


def test_llm_persistent_and_lexical_cache(tmp_path):
    import fake_llm as F
    from sipg_rl.llm import OpenAICompatLLM
    srv = F.serve(0); threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"; path = str(tmp_path / "c.jsonl"); F.CALLS.clear()
    a = OpenAICompatLLM(url, "fake-model:1b", cache_path=path)
    a("p1", "utilize", "Sentence one.", "lexical"); a("p2", "utilize", "A different sentence.", "lexical")
    assert a.stats["calls"] == 1 and a.stats["cache_hits"] == 1             # same span, different sentence -> hit
    n = len(F.CALLS); b = OpenAICompatLLM(url, "fake-model:1b", cache_path=path)
    b("p1", "utilize", "Sentence one.", "lexical")
    assert len(F.CALLS) == n and b.stats["cache_hits"] == 1                  # restart: served from disk
    srv.shutdown()
