"""End-to-end SIPG-RL: text -> sentence graphs -> learned traversal -> verified, audited output."""
from __future__ import annotations
from dataclasses import dataclass, field
import copy
import dataclasses
import spacy, torch
from .env import SimplifyEnv, EnvConfig, make_env
from .graph import GraphBuilder, EDGE_TYPES
from .prompts import PromptInstantiator
from .realize import RuleRealizer
from .scoring import FeatureSuitabilityScorer, ConceptEncoder
from .verifier import FidelityVerifier
from .model import SIPGPolicy
from .baselines import (LearnedPolicy, LearnedThenSweep, RandomPolicy, GreedyReadabilityPolicy, InvokeAllPolicy)
from .glossary import collect as collect_terms
from .patient_profile import (PatientProfile, ProfileSuitabilityScorer, profile_report, build_system_prompt,
                              span_style_suffix)


def default_parts(target_grade=6.0, cfg=None, nli=None, realizer=None, scorer=None, encoder=None,
                  instantiator=None, nlp=None, fidelity=None, calibration=None, verifier_enabled=True):
    """Assemble all pluggable components (override any with real models)."""
    nlp = nlp or spacy.load("en_core_web_sm")
    inst = instantiator or PromptInstantiator()
    _rz = realizer or RuleRealizer()
    if isinstance(_rz, RuleRealizer) and _rz.scorer is None:
        _rz.scorer = scorer or FeatureSuitabilityScorer(target_grade)
    return dict(nlp=nlp, builder=GraphBuilder(nlp, inst), scorer=scorer or FeatureSuitabilityScorer(target_grade),
                encoder=encoder or ConceptEncoder(), verifier=FidelityVerifier(nli=nli, enabled=verifier_enabled),
                realizer=_rz, cfg=cfg or EnvConfig(),
                fidelity=fidelity, calibration=calibration)


OP_LABELS = {
    "RELCL": "Split relative clause into its own sentence",
    "CONJ_CL": "Split at the conjunction into two sentences",
    "ADVCL": "Split subordinate clause (relation kept explicit)",
    "LEX": "Replace medical term with plain wording",
    "NOM": "Turn nominalization into a verb/gerund",
    "AMOD": "Turn jargon adjective into a plain clause",
    "CPLX": "Simplify long word", "ROOT": "Rewrite main clause",
    "CCOMP": "Simplify complement clause", "ACL": "Simplify participial modifier",
}
_EDGE_PRIORITY = ["contrast", "elaboration", "condition", "cause", "sequence", "syn_down", "next"]


def graph_json(g):
    nodes = [dict(id=n.id, kind=n.kind, text=n.text, relation=n.relation, parent=n.parent, depth=n.depth,
                  anchor=n.anchor, critical=bool(n.critical), prompt=n.prompt,
                  ctx=[k for k, v in n.meta.get("source_ctx", {}).items() if v and k != "any"]) for n in g.nodes]
    edges = []
    for i in range(g.n):
        for j in range(i + 1, g.n):
            names = {EDGE_TYPES[g.adj[i, j]] if g.adj[i, j] >= 0 else None,
                     EDGE_TYPES[g.adj[j, i]] if g.adj[j, i] >= 0 else None} - {None}
            if names:
                t = next((p for p in _EDGE_PRIORITY if p in names), sorted(names)[0])
                edges.append(dict(source=i, target=j, type=t))
    return nodes, edges


@dataclass
class SimplifyResult:
    text: str
    sentences: list = field(default_factory=list)   # per-sentence dicts incl. audit trail
    terms: list = field(default_factory=list)       # glossary entries found in the OUTPUT text (info boxes)
    profile: dict | None = None                     # Health Literacy module: the profile this run was conditioned on
    profile_report: dict | None = None              # FKGL / RVS / RTS vs the profile targets (+ needs_revision flag)
    profile_revision: dict | None = None            # document-level LLM revision attempt (accepted only if verified)


def _with_scorer(rz, scorer):
    """Shallow copy of a realizer whose candidate-choosing scorer is `scorer` (the shared realizer is never mutated)."""
    rz2 = copy.copy(rz)
    if hasattr(rz2, "scorer"):
        rz2.scorer = scorer
    rules = getattr(rz2, "rules", None)
    if rules is not None:
        r2 = copy.copy(rules)
        if hasattr(r2, "scorer"):
            r2.scorer = scorer
        rz2.rules = r2
    return rz2


class SIPGRL:
    def __init__(self, parts=None, policy=None, checkpoint=None, sweep=False, traversal="learned", seed=0):
        self.sweep, self.traversal, self.seed = sweep, traversal, seed
        self.parts = parts or default_parts()
        self.policy = policy or SIPGPolicy()
        if checkpoint:
            self.policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.policy.eval()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def simplify(self, text: str, apply_all: bool = False, profile=None, revise: bool = False) -> SimplifyResult:
        """apply_all=True: after the policy stops, fire every remaining node's prompt (verifier still gates each edit).

        profile (PatientProfile | dict | None): Health Literacy module. Conditions the GENERATION layer (LLM style hint)
        and the EVALUATION scorer (target FKGL band) only. The verifier object `parts["verifier"]` is used unchanged and
        never receives the profile. revise=True additionally tries ONE document-level LLM rewrite under the profile
        prompt (coherence / spiritual-first ordering); it is kept only if the profile-blind verifier accepts it."""
        profile = PatientProfile.from_dict(profile) if profile is not None else None
        p, llm, prev_suffix, styled = self.parts, None, "", False
        if profile is not None:
            sc = ProfileSuitabilityScorer(p["scorer"], profile)
            p = {**p, "scorer": sc, "realizer": _with_scorer(p["realizer"], sc)}
            llm = getattr(p["realizer"], "llm", None)
            styled = llm is not None and hasattr(llm, "style_suffix")     # plain callables just ignore the hint
            prev_suffix = llm.style_suffix if styled else ""
            if styled:
                llm.style_suffix = span_style_suffix(profile)
        try:
            res = self._simplify(text, apply_all, p)
        finally:
            if profile is not None and styled:
                llm.style_suffix = prev_suffix
        if profile is not None:
            self._apply_profile(res, text, profile, p, revise, llm)
        return res

    def _apply_profile(self, res, source, profile, p, revise, llm):
        rep = profile_report(res.text, profile)
        res.profile = profile.to_dict()
        res.profile_revision = dict(attempted=False, accepted=False, reason="")
        want = revise and (rep.needs_revision or profile.spiritual_first)
        if want and llm is not None and hasattr(llm, "rewrite_document"):
            cand = llm.rewrite_document(build_system_prompt(profile), res.text)
            rv = dict(attempted=True, accepted=False, before=res.text, after=cand, violations=[], reason="")
            if not cand:
                rv["reason"] = "LLM returned nothing"
            else:
                # profile-blind safety gate, checked against the ORIGINAL source text
                vr = p["verifier"].check(source, cand, use_nli=True)
                new = profile_report(cand, profile)
                rv["violations"] = vr.violations
                if not vr.ok:
                    rv["reason"] = "rejected by verifier"
                elif not profile.spiritual_first and len(new.reasons) > len(rep.reasons):
                    rv["reason"] = "revision did not improve the profile targets"
                else:
                    rv["accepted"], rv["reason"] = True, "ok"
                    res.text, rep = cand, new
                    res.terms = collect_terms(res.text)
            res.profile_revision = rv
        res.profile_report = rep.to_dict()

    def _simplify(self, text: str, apply_all: bool, p) -> SimplifyResult:
        doc = p["nlp"](text)
        outs, records = [], []
        for sent in doc.sents:
            if not sent.text.strip():
                continue
            g = p["builder"].build(sent)
            wide = apply_all or self.traversal == "invoke_all"
            cfg = dataclasses.replace(p["cfg"], max_steps_factor=6) if wide else p["cfg"]
            env = make_env(p, cfg)
            if self.traversal == "random":
                pol = RandomPolicy(self.seed + len(records))
            elif self.traversal == "greedy":
                pol = GreedyReadabilityPolicy()
            elif self.traversal == "invoke_all":
                pol = InvokeAllPolicy()
            else:
                base = LearnedPolicy(self.policy)
                pol = (LearnedThenSweep(base, kinds=None) if apply_all
                       else LearnedThenSweep(base) if self.sweep else base)
            env.reset(g); pol.reset()
            while not env.done:
                env.step(pol.act(env))
                env.trail[-1]["driver"] = getattr(pol, "last_driver", "policy")
            final = p["verifier"].check(env.src, env.text, use_nli=True)
            out = env.text if final.ok else env.src       # hard safety fallback: never emit unverified text
            outs.append(out)
            nodes, edges = graph_json(g)
            steps, prev, ops, skipped, rejected = [], env.src, [], [], []
            for t in env.trail:
                st = dict(step=t["step"], action=t["action"], at=t["node"], kind=t["kind"], event=t.get("event", ""),
                          reward=t["reward"], ret=t["ret"], before=prev, after=t["text"], prompt=t["prompt"],
                          driver=t.get("driver", "policy"), S_before=t.get("S_before"), S_after=t.get("S_after"),
                          verifier=t.get("verifier"), realizer=t.get("realizer"), candidate=t.get("candidate"))
                if t["action"].startswith("MOVE"):
                    st["to"] = int(t["action"].split("->")[1])
                steps.append(st)
                if t["action"] == "INVOKE":
                    base = dict(step=t["step"], node=t["node"], kind=t["kind"], label=OP_LABELS.get(t["kind"], t["kind"]),
                                span=nodes[t["node"]]["text"], prompt=t["prompt"], reward=t["reward"],
                                driver=t.get("driver", "policy"),
                                via=t.get("realizer") or ("llm" if getattr(p["realizer"], "handles", lambda k: False)(t["kind"]) else "rules"),
                                candidate=t.get("candidate"), S_before=t.get("S_before"), S_after=t.get("S_after"), verifier=t.get("verifier"),
                                before=prev, after=t["text"])
                    ev = t.get("event")
                    (ops if ev == "edited" else rejected if ev == "rejected" else skipped).append(
                        {**base, "violations": t.get("violations", [])})
                prev = t["text"]
            records.append(dict(graph=dict(nodes=nodes, edges=edges), steps=steps, operations=ops,
                                skipped=skipped, rejected=rejected, final_rejected=not final.ok,
                                final_verdict="pass" if final.ok else "fallback", fallback=not final.ok,
                                n_moves=env.n_move, n_invokes=env.n_inv, source=env.src, output=out, verified=final.ok, violations=final.violations,
                                S_before=env.S0, S_after=env.S if final.ok else env.S0, sim=env.sim,
                                trail=[t for t in env.trail if t["action"] == "INVOKE"]))
        text_out = " ".join(outs)
        return SimplifyResult(text_out, records, collect_terms(text_out))
