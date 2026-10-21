"""SOURCE uncertainty / assertion context (NegEx/ConText-style).

Question answered here: *does the clinical fact in the source carry a negation, hedge, conditional,
historical or other-experiencer qualifier?*  This is a property of the INPUT, attached to graph nodes as
`node.meta["source_ctx"]`. It feeds preservation checks (verifier) only. It is deliberately independent of
`uncertainty.py`, which answers the different question 'is the model's rewrite of this span trustworthy?'."""
from __future__ import annotations
from .lexicon import NEGATION_CUES, HEDGE_CUES, COND_CUES, HISTORICAL_CUES, OTHER_EXPERIENCER, SCOPE_BREAKERS

LEFT_WINDOW = 5


def _left_triggers(doc, i, lo):
    """Cue words in the <=5 tokens before token i, stopping at punctuation / scope breakers."""
    found, k = [], i - 1
    while k >= lo and i - k <= LEFT_WINDOW:
        t = doc[k]
        if t.is_punct or t.lower_ in SCOPE_BREAKERS:
            break
        found.append(t.lower_)
        k -= 1
    return found


def source_context(node, doc, sent_lo: int) -> dict:
    """Assertion flags for a node. For terms: cues scoped over the term; for clauses: the clause's own
    marker (if/unless) and cues on its anchor."""
    first, anchor = min(node.ids), doc[node.anchor]
    left = _left_triggers(doc, first, sent_lo)
    flags = {
        "negated": any(w in NEGATION_CUES or w.endswith("n't") for w in left)
                   or any(c.lower_ in NEGATION_CUES and c.dep_ in ("neg", "det", "advmod") for c in anchor.head.children),
        "hedged": any(w in HEDGE_CUES for w in left) or any(c.lower_ in HEDGE_CUES for c in anchor.head.children),
        "conditional": False,
        "historical": any(w in HISTORICAL_CUES for w in left),
        "other_experiencer": any(w in OTHER_EXPERIENCER for w in left),
    }
    p = anchor
    while True:                                   # inside an if/unless clause (any ancestor)?
        if p.dep_ == "advcl" and any(c.dep_ == "mark" and c.lower_ in COND_CUES for c in p.children):
            flags["conditional"] = True
            break
        if p.head.i == p.i:
            break
        p = p.head
    flags["any"] = any(flags.values())
    return flags
