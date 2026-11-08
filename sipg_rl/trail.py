"""P2.2 - audit-trail export: one JSON record per (system, example) with the full traversal and per-step provenance."""
from __future__ import annotations


def build_trail_record(ex_id, system, source, output, res=None):
    """`res`: a pipeline SimplifyResult (None for copy/reference). Steps from every sentence are concatenated."""
    traversal, fallback, verdict = [], False, "pass"
    for si, s in enumerate((res.sentences if res else [])):
        nodes = s["graph"]["nodes"]
        for t in s["steps"]:
            node = nodes[t["at"]]
            rec = dict(sentence=si, step=t["step"], node=node["text"], kind=node["kind"],
                       action="MOVE" if t["action"].startswith("MOVE") else t["action"])
            if t["action"] == "MOVE" or t["action"].startswith("MOVE"):
                rec["to"] = nodes[t["to"]]["text"] if "to" in t else None
            if t["action"] == "INVOKE":
                c = t.get("candidate") or {}
                rec.update(prompt=t["prompt"], candidate_used=c.get("candidate_used"), candidate_rank=c.get("candidate_rank"),
                           candidate_count=c.get("candidate_count"), realizer=t.get("realizer"), verifier=t.get("verifier"),
                           outcome=t["event"], S_before=t.get("S_before"), S_after=t.get("S_after"), driver=t.get("driver"))
            traversal.append(rec)
        fallback |= bool(s.get("fallback")); verdict = "fallback" if fallback else verdict
    return dict(id=ex_id, system=system, source=source, output=output, traversal=traversal,
                final_verdict=verdict, fallback=fallback)


def explain_term_outcome(result, term: str) -> str:
    """What happened to the node for `term`?  'applied' | 'noop' (prompt fired, no change) | 'reverted' (edit blocked by the
    verifier) | 'skipped' (never fired) | 'absent'. Uses the TIGHTEST node containing the term (term-level kinds first): an
    enclosing clause node such as the `if ...` condition legitimately does nothing and must not be mistaken for the term's node."""
    TERM = ("LEX", "NOM", "AMOD", "CPLX")
    best = None
    for s in result.sentences:
        for n in s["graph"]["nodes"]:
            if term.lower() in n["text"].lower() and n["kind"] != "ROOT":
                key = (n["kind"] not in TERM, len(n["text"]))
                if best is None or key < best[0]:
                    best = (key, s, n)
    if best is None:
        return "absent"
    _, s, n = best
    evs = [t["event"] for t in s["steps"] if t["action"] == "INVOKE" and t["at"] == n["id"]]
    return "applied" if "edited" in evs else "reverted" if "rejected" in evs else "noop" if "noop" in evs else "skipped"
