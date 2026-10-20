import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, pytest
from sipg_rl.pipeline import default_parts
from sipg_rl.env import SimplifyEnv, INVOKE, STOP, MOVE0, N_ACTIONS
from sipg_rl.realize import Realizer, CallableRealizer, RuleRealizer
from sipg_rl.verifier import FidelityVerifier
from sipg_rl.model import SIPGPolicy, collate
from sipg_rl.baselines import InvokeAllPolicy, run_episode

TXT = "The hypertensive patient, who takes lisinopril 10 mg BID, reports no chest pain."


@pytest.fixture(scope="module")
def parts():
    return default_parts()


def mkenv(parts, realizer=None):
    return SimplifyEnv(parts["scorer"], parts["encoder"], parts["verifier"], realizer or parts["realizer"], parts["cfg"])


def test_graph_structure(parts):
    g = parts["builder"].build_text(TXT)
    kinds = [n.kind for n in g.nodes]
    assert kinds[0] == "ROOT" and "RELCL" in kinds and "AMOD" in kinds and "LEX" in kinds
    relcl = next(n for n in g.nodes if n.kind == "RELCL")
    assert relcl.relation == "elaboration" and relcl.parent == 0
    assert (g.adj.diagonal() >= 0).all()


def test_invoke_all_preserves_critical_spans(parts):
    g = parts["builder"].build_text(TXT)
    env = run_episode(mkenv(parts), g, InvokeAllPolicy())
    assert parts["verifier"].check(env.src, env.text).ok
    assert "10 mg" in env.text and "no chest pain" in env.text and "lisinopril" in env.text
    assert "twice a day" in env.text and env.S > env.S0


def test_verifier_gates():
    v = FidelityVerifier()
    src = "Take warfarin 5 mg BID. Do not drive."
    assert v.check(src, "Take warfarin 5 mg twice a day. Do not drive.").ok          # approved substitution
    assert not v.check(src, "Take warfarin 50 mg twice a day. Do not drive.").ok     # dose altered
    assert not v.check(src, "Take warfarin 5 mg twice a day. Drive.").ok             # negation dropped
    assert not v.check(src, "Take a blood thinner 5 mg twice a day. Do not drive.").ok  # drug substituted
    assert not v.check(src, "Take warfarin 5 mg twice a day and aspirin. Do not drive.").ok  # hallucinated drug


class DropDose(Realizer):
    """A faulty 'LLM' that deletes the dose - the env must roll it back and penalise."""
    def apply(self, graph, node, st):
        for t in graph.doc:
            if t.text == "10":
                st.delete(t.i)
                return True
        return False


def test_env_rolls_back_unsafe_edit(parts):
    g = parts["builder"].build_text(TXT)
    env = mkenv(parts, DropDose())
    env.reset(g)
    r, done, info = env.step(INVOKE)
    assert info["event"] == "rejected" and env.text == env.src and r < -0.1 and env.n_reject == 1


def test_callable_realizer(parts):
    g = parts["builder"].build_text("The patient should utilize the inhaler.")
    llm = lambda prompt, text: "use" if text.strip() == "utilize" else text
    env = mkenv(parts, CallableRealizer(llm, RuleRealizer()))
    env.reset(g)
    node = next(n for n in g.nodes if n.kind == "LEX")
    env.cur = node.id
    env.step(INVOKE)
    assert "use the inhaler" in env.text


def test_policy_masks_and_shapes(parts):
    g = parts["builder"].build_text(TXT)
    env = mkenv(parts)
    ob = env.reset(g)
    pi = SIPGPolicy()
    logits, v = pi(collate([ob, ob]))
    assert logits.shape == (2, N_ACTIONS) and v.shape == (2,)
    probs = torch.softmax(logits[0], -1)
    assert probs[~torch.as_tensor(ob["amask"])].sum() < 1e-6
    assert torch.isfinite(logits[0][torch.as_tensor(ob["amask"])]).all()


def test_padding_invariance(parts):
    """Padded graph slots must not influence the encoding of real nodes."""
    g = parts["builder"].build_text(TXT)
    ob = mkenv(parts).reset(g)
    pi = SIPGPolicy().eval()
    l1, _ = pi(collate([ob]))
    ob2 = dict(ob); x = ob["x"].copy(); x[g.n:, :] = 5.0; ob2["x"] = x
    l2, _ = pi(collate([ob2]))
    assert torch.allclose(l1, l2, atol=1e-5)


# ---- regression tests for review items 1-3 ------------------------------------------------------
def test_no_entity_collapse_and_entities_are_attributes(parts):
    import spacy
    from sipg_rl.graph import GraphBuilder
    from sipg_rl.prompts import PromptInstantiator
    nlp = spacy.load("en_core_web_sm"); nlp.add_pipe("merge_entities")
    with pytest.raises(ValueError):
        GraphBuilder(nlp, PromptInstantiator())          # merging NER spans before graph build is refused
    g = parts["builder"].build_text("The patient was seen at Boston Medical Center for hypertensive urgency.")
    assert [t[1] for t in g.toks][4:7] == ["at", "Boston", "Medical"]   # entity tokens stay separate
    assert any(e.label_ == "ORG" for e in g.doc.ents)


def test_amod_structure_kept_and_np_prompt(parts):
    g = parts["builder"].build_text("Discontinue lisinopril 10 mg if the patient develops hypertensive urgency.")
    n = next(x for x in g.nodes if x.meta.get("key") == "hypertensive urgency")
    assert ("hypertensive", "amod", "urgency") in n.meta["structure"] and n.meta["modifier"] == ("hypertensive", "urgency")
    assert "replace the whole noun phrase" in n.prompt
    d = next(x for x in g.nodes if x.meta.get("key") == "discontinue")   # parser tags it amod; head is outside the term
    assert d.meta["modifier"] is None and "replace the whole noun phrase" not in d.prompt


def test_amod_llm_replaces_whole_noun_phrase(parts):
    g = parts["builder"].build_text("The patient has a hypertensive cough.")
    env = mkenv(parts, CallableRealizer(lambda p, t: "a cough from high blood pressure" if "hypertensive" in t else t))
    env.reset(g)
    env.cur = next(n.id for n in g.nodes if n.kind == "AMOD")
    env.step(INVOKE)
    assert "a cough from high blood pressure" in env.text and "hypertensive" not in env.text


def test_source_vector_cache_is_per_encoder(parts):
    from sipg_rl.scoring import ConceptEncoder
    g = parts["builder"].build_text(TXT)
    e1 = mkenv(parts); e1.reset(g)
    e2 = SimplifyEnv(parts["scorer"], ConceptEncoder(binary=True, bigrams=False), parts["verifier"], parts["realizer"], parts["cfg"])
    e2.reset(g)
    assert abs(e2.sim - 1.0) < 1e-6 and abs(e1.sim - 1.0) < 1e-6
    e2.step(STOP); assert e2.sim == 1.0


# ---- dashboard payload ---------------------------------------------------------------------------
def test_simplify_exposes_graph_steps_and_operations(parts):
    from sipg_rl.pipeline import SIPGRL
    m = SIPGRL(parts, checkpoint=os.path.join(os.path.dirname(__file__), "..", "sipg_policy_bonus.pt"))
    res = m.simplify(TXT + " Follow up as directed.")
    assert len(res.sentences) == 2
    s = res.sentences[0]
    assert {"graph", "steps", "operations", "skipped", "rejected", "verified"} <= set(s)
    assert s["graph"]["nodes"] and s["graph"]["edges"]
    assert all({"label", "before", "after", "kind", "step"} <= set(o) for o in s["operations"])
    assert [t["step"] for t in s["steps"]] == list(range(1, len(s["steps"]) + 1))
    assert res.sentences[1]["operations"] == [] and res.sentences[1]["output"] == "Follow up as directed."  # E8 no-op


# ---- LLM backend (tested against a stand-in server, NOT a real model) -----------------------------
@pytest.fixture(scope="module")
def fake_llm():
    import threading
    sys.path.insert(0, os.path.dirname(__file__))
    import fake_llm as F
    srv = F.serve(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1", F
    srv.shutdown()


def test_llm_client_sanitises_caches_and_reports_health(fake_llm):
    from sipg_rl.llm import OpenAICompatLLM, sanitize
    url, F = fake_llm
    llm = OpenAICompatLLM(url, "fake-model:1b")
    assert llm.health()["reachable"] and llm.health()["model_available"]
    assert llm("p", "dyspnea", "He has dyspnea.") == "trouble breathing"       # prefix, quotes, final dot stripped
    n = len(F.CALLS)
    assert llm("p", "dyspnea", "He has dyspnea.") == "trouble breathing" and len(F.CALLS) == n   # cached
    assert llm.stats["cache_hits"] == 1
    body = F.CALLS[-1]
    assert body["temperature"] == 0 and body["messages"][0]["role"] == "system"
    assert sanitize("Here is a long explanation of why. I changed it because of many reasons that follow.", "renal") == ""


def test_llm_errors_never_crash(parts):
    from sipg_rl.llm import OpenAICompatLLM
    llm = OpenAICompatLLM("http://127.0.0.1:9/v1", "x", timeout=1)
    assert llm("p", "renal", "ctx") == "" and llm.stats["errors"] == 1 and "Error" in llm.stats["last_error"]
    assert not llm.health()["reachable"]


def test_llm_pipeline_gate_and_sweep(parts, fake_llm):
    from sipg_rl.llm import OpenAICompatLLM
    from sipg_rl.pipeline import SIPGRL
    from sipg_rl.baselines import LearnedPolicy
    url, _ = fake_llm
    llm = OpenAICompatLLM(url, "fake-model:1b")
    P = default_parts(nlp=parts["nlp"], realizer=CallableRealizer(llm, RuleRealizer()))
    m = SIPGRL(P, checkpoint=os.path.join(os.path.dirname(__file__), "..", "sipg_policy_bonus.pt"), sweep=True)
    res = m.simplify("Take warfarin 5 mg orally. The patient should utilize the inhaler.")
    s0, s1 = res.sentences
    assert "5 mg" in s0["output"] and "50 mg" not in s0["output"]                 # LLM invented a dose -> verifier rolled back
    assert any(r["violations"] for r in s0["rejected"])
    assert "use the inhaler" in s1["output"]                                      # LLM edit accepted
    assert s1["operations"][0]["driver"] == "policy"
    assert all(o["via"] == "llm" for o in s1["operations"] if o["kind"] == "LEX")


def test_sweep_fires_term_nodes_the_policy_skipped(parts, fake_llm):
    from sipg_rl.llm import OpenAICompatLLM
    from sipg_rl.baselines import NoOpPolicy, LearnedThenSweep
    url, _ = fake_llm
    P = default_parts(nlp=parts["nlp"], realizer=CallableRealizer(OpenAICompatLLM(url, "fake-model:1b"), RuleRealizer()))
    g = P["builder"].build_text("The patient should utilize the inhaler.")
    env = SimplifyEnv(P["scorer"], P["encoder"], P["verifier"], P["realizer"], P["cfg"])
    env.reset(g); pol = LearnedThenSweep(NoOpPolicy()); drivers = []
    while not env.done:
        env.step(pol.act(env)); drivers.append(pol.last_driver)
    assert "use the inhaler" in env.text and "sweep" in drivers


def test_apply_all_fires_every_node_and_stays_verified(parts):
    from sipg_rl.pipeline import SIPGRL
    m = SIPGRL(parts, checkpoint=os.path.join(os.path.dirname(__file__), "..", "sipg_policy_bonus.pt"))
    t = "Discontinue warfarin 5 mg if dyspnea occurs, because the drug is contraindicated in patients with renal impairment."
    a, b = m.simplify(t).sentences[0], m.simplify(t, apply_all=True).sentences[0]
    inv = {s["at"] for s in b["steps"] if s["action"] == "INVOKE"}
    assert inv == {n["id"] for n in b["graph"]["nodes"]}          # every node got its prompt
    assert b["verified"] and "5 mg" in b["output"] and "if" in b["output"].lower()
    assert b["n_invokes"] >= a["n_invokes"]
    # policy that always stops: apply-all must still do all the work
    from sipg_rl.baselines import NoOpPolicy, LearnedThenSweep
    env = SimplifyEnv(parts["scorer"], parts["encoder"], parts["verifier"], parts["realizer"], parts["cfg"])
    g = parts["builder"].build_text(t); env.reset(g); pol = LearnedThenSweep(NoOpPolicy(), kinds=None)
    while not env.done:
        env.step(pol.act(env))
    assert env.invoked[:g.n].all() and env.n_changed >= 3
