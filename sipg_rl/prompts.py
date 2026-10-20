"""Component 2: local prompt instantiation. Prompts are generated per node from its syntactic role
and local context (no fixed operation library). `instantiate` can be swapped for a small LLM."""
from __future__ import annotations

TEMPLATES = {
    "ROOT": "Check whether the main clause ('{text}') is overly long; if so, move secondary information into its own sentence.",
    "RELCL": "Rewrite this modifying clause as a separate sentence if it improves readability; preserve the head noun '{head}' and every number, negation and drug name. Clause: \"{text}\"",
    "CONJ_CL": "This clause is joined by '{cc}' ({relation}). Split it into its own sentence, keeping both statements true. Clause: \"{text}\"",
    "ADVCL": "This subordinate clause expresses {relation}. Make it a separate sentence only if the {relation} relation stays explicit. Clause: \"{text}\"",
    "CCOMP": "This complement clause ('{text}') is embedded in '{head}'. Only simplify wording inside it; do not detach it.",
    "ACL": "This participial modifier ('{text}') attaches to '{head}'. Simplify only if meaning is unchanged.",
    "AMOD": "The adjective '{text}' modifies '{head}'. If this modifying adjective is clinical jargon, replace the whole noun phrase '{np}' with a patient-friendly equivalent that preserves the clinical meaning.",
    "LEX_NP": "The noun phrase '{text}' has the modifier '{mod}' (amod of '{head}'). If this modifying adjective is clinical jargon, replace the whole noun phrase with a patient-friendly equivalent that preserves the clinical meaning; keep doses, negations and drug names.",
    "LEX": "Replace the medical term '{text}' with plain words a patient would use, unless it is a drug name, a dose or a negation.",
    "NOM": "The noun '{text}' encodes an action; if clearer, express it as a verb or gerund.",
    "CPLX": "The word '{text}' is long and may be unfamiliar; offer a simpler synonym only if the meaning is identical.",
}


class PromptInstantiator:
    def instantiate(self, node, doc) -> str:
        t = doc[node.anchor]
        ccs = [c.lower_ for c in t.head.children if c.dep_ == "cc" and c.i < t.i]
        np_ids = node.meta.get("np_ids", node.ids)
        mod = node.meta.get("modifier")
        ctx = dict(text=" ".join(doc[i].text for i in node.ids[:25]), head=t.head.text,
                   relation=node.relation or "elaboration", cc=(ccs[-1] if ccs else "and"),
                   np=" ".join(doc[i].text for i in np_ids), mod=mod[0] if mod else "")
        kind = "LEX_NP" if node.kind == "LEX" and mod else node.kind
        return TEMPLATES[kind].format(**ctx)


class LLMPromptInstantiator(PromptInstantiator):
    """Adapter: `llm(role, context) -> str` (e.g. a small instruction-tuned model)."""
    def __init__(self, llm):
        self.llm = llm

    def instantiate(self, node, doc) -> str:
        base = super().instantiate(node, doc)
        try:
            return self.llm(node.kind, base) or base
        except Exception:
            return base
