import os, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import pytest
from sipg_rl.glossary import find_terms, collect, lookup
from sipg_rl.llm import OpenAICompatLLM, parse_explanation

TXT = "The patient with high blood pressure takes lisinopril 10 mg BID and warfarin; heart attack ruled out. Kidneys fine, po fluids."


def test_finds_body_parts_medicines_conditions_and_abbreviations():
    got = {m.surface: m.entry.type for m in find_terms(TXT)}
    assert got["high blood pressure"] == "condition" and got["lisinopril"] == "medicine"
    assert got["BID"] == "abbr" and got["heart attack"] == "condition" and got["Kidneys"] == "body"


def test_longest_match_wins_and_no_overlap():
    ms = find_terms("high blood pressure and chest pain")
    assert [m.surface for m in ms] == ["high blood pressure", "chest pain"]       # not 'blood pressure' / 'chest'


def test_abbreviations_are_case_sensitive():
    assert not any(m.surface.lower() == "po" for m in find_terms("po fluids and a po box"))
    assert any(m.surface == "PO" for m in find_terms("Give PO daily"))


def test_word_shape_guesses_are_flagged_and_stoplisted():
    ms = {m.surface: m.entry.source for m in find_terms("gastritis, nephropathy, pitavastatin, academia, analysis, April")}
    assert ms == {"gastritis": "pattern", "nephropathy": "pattern", "pitavastatin": "pattern"}
    assert find_terms("gastritis", guess=False) == []


def test_known_drug_without_entry_is_flagged_as_name_only():
    e = lookup("sildenafil_x") or lookup("tacrolimus")
    assert e and e.type == "medicine"
    from sipg_rl.lexicon import CRITICAL_TERMS
    assert all(find_terms(n)[0].entry.type == "medicine" for n in list(CRITICAL_TERMS)[:20])


def test_every_medicine_entry_carries_the_safety_tip_and_no_dose_numbers():
    import re
    for m in find_terms("warfarin lisinopril metformin methotrexate opioid"):
        if m.entry.type == "medicine":
            assert "pharmacist" in m.entry.extra
            assert not re.search(r"\d+\s?(mg|mcg|ml)\b", m.entry.definition + m.entry.extra)


def test_collect_dedupes_and_lists_surfaces():
    c = collect("Kidney and kidneys, kidney again")
    assert len(c) == 1 and set(c[0]["surfaces"]) == {"Kidney", "kidneys", "kidney"} and c[0]["type_label"] == "Body part"


def test_parse_explanation_is_forgiving():
    assert parse_explanation('ok {"type":"weird","definition":"x y","extra":null}')["type"] == "other"
    assert parse_explanation("I do not know.") == {} and parse_explanation('{"definition": ""}') == {}


@pytest.fixture(scope="module")
def fake_llm():
    import fake_llm as F
    srv = F.serve(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_llm_explain_caches_and_handles_unsure(fake_llm):
    llm = OpenAICompatLLM(fake_llm, "fake-model:1b")
    d = llm.explain("gastritis", "He has gastritis.")
    assert d["type"] == "condition" and "stomach" in d["definition"]
    llm.explain("gastritis", "He has gastritis.")
    assert llm.stats["cache_hits"] == 1
    assert llm.explain("zzzitis") == {}                               # unsure -> empty, no crash
    assert OpenAICompatLLM("http://127.0.0.1:1/v1", "x", timeout=1).explain("gastritis") == {}
