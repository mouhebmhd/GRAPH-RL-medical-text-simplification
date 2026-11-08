"""Task 2 - per-category deterministic safety checks (rules only, no model). check_all(src, out) -> per-category verdicts.
Independent of FidelityVerifier so it can serve as the POST-HOC checker when the verifier is switched off."""
from __future__ import annotations
import re
from collections import Counter
from .normalize import normalize_number_unit, fmt
from .lexicon import CRITICAL_TERMS, NEGATION_CUES, HEDGE_CUES, COND_CUES, APPROVED_SUBS

CATEGORIES = ("number", "negation", "hedge", "conditional", "drug", "abbrev")
_W = lambda t: re.findall(r"[a-z']+", t.lower())


def _cnt(t, cues):
    return sum(1 for w in _W(t) if w in cues or (cues is NEGATION_CUES and w.endswith("n't")))


def check_all(src: str, out: str) -> dict:
    """{category: {"applicable": bool, "pass": bool|None, "detail": {...}}, "all_pass": bool}. pass=None when the source has no such span."""
    res = {}
    sn, on = Counter(normalize_number_unit(src)), Counter(normalize_number_unit(out))
    sv, ov = Counter(v for v, _ in sn.elements()), Counter(v for v, _ in on.elements())
    lost = [fmt(v) for v in (sv - ov)]
    unit_lost = [f"{fmt(v)} {u}" for (v, u), c in (Counter(p for p in sn.elements() if p[1]) - on).items()]
    added = [fmt(v) for v in ov if v not in sv]
    res["number"] = dict(applicable=bool(sv), pass_=(not lost and not unit_lost and not added) if sv else None,
                         detail=dict(lost=lost, unit_lost=unit_lost, added=added))
    for name, cues in (("negation", NEGATION_CUES), ("hedge", HEDGE_CUES), ("conditional", COND_CUES)):
        s, o = _cnt(src, cues), _cnt(out, cues)
        res[name] = dict(applicable=s > 0, pass_=(o >= s) if s else None, detail=dict(src=s, out=o))
    ws, wo = Counter(_W(src)), Counter(_W(out))
    drugs = [d for d in CRITICAL_TERMS if ws[d]]
    gone = [d for d in drugs if wo[d] < ws[d]]
    new = [d for d in CRITICAL_TERMS if wo[d] and not ws[d]]
    res["drug"] = dict(applicable=bool(drugs), pass_=(not gone and not new) if (drugs or new) else None, detail=dict(lost=gone, hallucinated=new))
    ab = [a for a in APPROVED_SUBS if ws[a]]
    bad = [a for a in ab if wo[a] + sum(out.lower().count(p) for p in APPROVED_SUBS[a]) < ws[a]]
    res["abbrev"] = dict(applicable=bool(ab), pass_=(not bad) if ab else None, detail=dict(lost=bad))
    out_ = {k: dict(applicable=v["applicable"], **{"pass": v["pass_"]}, detail=v["detail"]) for k, v in res.items()}
    out_["all_pass"] = all(v["pass"] is not False for v in out_.values() if isinstance(v, dict))
    return out_
