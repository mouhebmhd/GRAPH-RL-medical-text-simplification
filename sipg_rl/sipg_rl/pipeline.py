"""End-to-end SIPG-RL: text -> sentence graphs -> learned traversal -> verified, audited output."""
from __future__ import annotations
from dataclasses import dataclass, field
import dataclasses, json, os, warnings
import spacy, torch
from .env import SimplifyEnv, EnvConfig
from .graph import GraphBuilder, EDGE_TYPES
from .prompts import PromptInstantiator
from .realize import RuleRealizer
from .scoring import FeatureSuitabilityScorer, ConceptEncoder, BandSuitabilityScorer, ContentEncoder
from .verifier import FidelityVerifier
from .model import SIPGPolicy
from .baselines import LearnedPolicy, LearnedThenSweep
from .glossary import collect as collect_terms


def default_parts(target_grade=6.5, cfg=None, nli=None, realizer=None, scorer=None, encoder=None,
                  instantiator=None, nlp=None):
    """Assemble all pluggable components (override any with real models)."""
    nlp = nlp or spacy.load("en_core_web_sm")
    inst = instantiator or PromptInstantiator()
    return dict(nlp=nlp, builder=GraphBuilder(nlp, inst), scorer=scorer or BandSuitabilityScorer(target_grade),
                encoder=encoder or ContentEncoder(), verifier=FidelityVerifier(nli=nli),
                realizer=realizer or RuleRealizer(), cfg=cfg or EnvConfig())


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


class SIPGRL:
    def __init__(self, parts=None, policy=None, checkpoint=None, sweep=False):
        self.sweep = sweep
        self.parts = parts or default_parts()
        self.policy = policy or SIPGPolicy()
        self.meta = None
        if checkpoint:
            self.policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            self.meta = self._check_meta(checkpoint)
        self.policy.eval()

    def _check_meta(self, checkpoint):
        """B4b: make train/deploy mismatches visible instead of silently confounding results."""
        mp = checkpoint + ".meta.json"
        if not os.path.exists(mp):
            warnings.warn(f"{checkpoint} has no .meta.json: it was trained under the LEGACY reward (old Sim/S, gamma<=0.01, "
                          "possibly a coverage bonus). Retrain with scripts/train_policy.py before drawing conclusions.")
            return None
        meta = json.load(open(mp))
        rz = self.parts["realizer"]
        deployed = "llm" if hasattr(rz, "llm") else "rules"
        if meta.get("realizer_in_loop") != deployed:
            warnings.warn(f"policy was trained with the '{meta.get('realizer_in_loop')}' realizer but is deployed with '{deployed}': "
                          "treat LLM mode as an inference-time augmentation, not something the policy optimised for.")
        for key, now in (("scorer", type(self.parts["scorer"]).__name__), ("encoder", type(self.parts["encoder"]).__name__)):
            if meta.get(key) != now:
                warnings.warn(f"policy was trained with {key}={meta.get(key)} but is deployed with {now}.")
        return meta

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def simplify(self, text: str, apply_all: bool = False) -> SimplifyResult:
        """apply_all=True: after the policy stops, fire every remaining node's prompt (verifier still gates each edit)."""
        p = self.parts
        doc = p["nlp"](text)
        outs, records = [], []
        for sent in doc.sents:
            if not sent.text.strip():
                continue
            g = p["builder"].build(sent)
            cfg = dataclasses.replace(p["cfg"], max_steps_factor=6) if apply_all else p["cfg"]
            env = SimplifyEnv(p["scorer"], p["encoder"], p["verifier"], p["realizer"], cfg)
            base = LearnedPolicy(self.policy)
            pol = (LearnedThenSweep(base, kinds=None) if apply_all
                   else LearnedThenSweep(base) if self.sweep else base)
            env.reset(g); pol.reset()
            while not env.done:
                env.step(pol.act(env))
                env.trail[-1]["driver"] = getattr(pol, "last_driver", "policy")
            final = p["verifier"].check(env.src, env.text, use_nli=True)
            # The env already reverts to the source when ITS final check fails (event "final_reject"), after which this
            # re-check sees the source and passes. Count that revert, or the fallback is invisible in every report.
            env_rej = [t for t in env.trail if t.get("event") == "final_reject"]
            fell_back = bool(env_rej) or not final.ok
            f_viol = env_rej[0]["violations"] if env_rej else final.violations
            # the env already reverts to the source on a failed final check (event 'final_reject'); count both paths
            fell_back = (not final.ok) or any(t.get("event") == "final_reject" for t in env.trail)
            out = env.text if final.ok else env.src       # hard safety fallback: never emit unverified text
            outs.append(out)
            nodes, edges = graph_json(g)
            steps, prev, ops, skipped, rejected = [], env.src, [], [], []
            for t in env.trail:
                st = dict(step=t["step"], action=t["action"], at=t["node"], kind=t["kind"], event=t.get("event", ""),
                          reward=t["reward"], ret=t["ret"], before=prev, after=t["text"], prompt=t["prompt"],
                          driver=t.get("driver", "policy"))
                if t["action"].startswith("MOVE"):
                    st["to"] = int(t["action"].split("->")[1])
                steps.append(st)
                if t["action"] == "INVOKE":
                    base = dict(step=t["step"], node=t["node"], kind=t["kind"], label=OP_LABELS.get(t["kind"], t["kind"]),
                                span=nodes[t["node"]]["text"], prompt=t["prompt"], reward=t["reward"],
                                driver=t.get("driver", "policy"),
                                via=t.get("via") or ("llm" if getattr(p["realizer"], "handles", lambda k: False)(t["kind"]) else "rules"),
                                before=prev, after=t["text"])
                    ev = t.get("event")
                    (ops if ev == "edited" else rejected if ev == "rejected" else skipped).append(
                        {**base, "violations": t.get("violations", [])})
                prev = t["text"]
            records.append(dict(graph=dict(nodes=nodes, edges=edges), steps=steps, operations=ops,
                                skipped=skipped, rejected=rejected, final_rejected=fell_back,
                                n_moves=env.n_move, n_invokes=env.n_inv, n_dropped=getattr(g, "n_dropped", 0), source=env.src, output=out, verified=not fell_back, violations=f_viol,
                                S_before=env.S0, S_after=env.S if not fell_back else env.S0, sim=env.sim,
                                trail=[t for t in env.trail if t["action"] == "INVOKE"]))
        text_out = " ".join(outs)
        return SimplifyResult(text_out, records, collect_terms(text_out))
