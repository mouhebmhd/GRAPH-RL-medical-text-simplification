"""Tasks 1, 5 - the {strong, weak} x {verifier on, off} engine. Every cell is scored by the SAME independent post-hoc checker
(checks.check_all), so turning the verifier off cannot change how safety is measured.
  weak   = CopyLexiconRealizer (top-1 lexicon swap only; safe by construction)
  strong = an aggressive realizer. Offline default: AdversarialRealizer(RuleRealizer) = a STAND-IN for a strong-but-unsafe LLM
           (edits a lot AND sometimes drops doses/negations/hedges/conditions/drugs). With a reachable LLM use kind='strong_llm'
           (HybridRealizer). The stand-in measures the gate, not any LLM."""
from __future__ import annotations
import numpy as np
from .checks import check_all, CATEGORIES
from .pipeline import SIPGRL, default_parts
from .realize import RuleRealizer, CopyLexiconRealizer, HybridRealizer
from .adversarial import AdversarialRealizer


def make_realizer(kind, llm=None, rate=0.5, seed=0):
    if kind == "weak":
        return CopyLexiconRealizer()
    if kind == "strong":
        return AdversarialRealizer(RuleRealizer(), rate=rate, seed=seed)
    if kind == "strong_llm":
        if llm is None:
            raise ValueError("strong_llm needs an LLM client")
        return HybridRealizer(llm, RuleRealizer())
    raise ValueError(kind)


def run_cell(sentences, kind, verifier_on, nlp=None, ckpt=None, llm=None, rate=0.5, seed=0, apply_all=True):
    parts = default_parts(nlp=nlp, realizer=make_realizer(kind, llm, rate, seed), verifier_enabled=verifier_on)
    m = SIPGRL(parts, checkpoint=ckpt)
    rows = []
    for s in sentences:
        r = m.simplify(s, apply_all=apply_all)
        out = r.text
        ca = check_all(s, out)
        rows.append(dict(src=s, out=out, unsafe=int(not ca["all_pass"]), cats={c: ca[c]["pass"] for c in CATEGORIES},
                         S_before=float(np.mean([x["S_before"] for x in r.sentences])), S_after=float(np.mean([x["S_after"] for x in r.sentences])),
                         edits=sum(len(x["operations"]) for x in r.sentences), rejected=sum(len(x["rejected"]) for x in r.sentences),
                         fallback=int(any(x["final_rejected"] for x in r.sentences)), changed=int(out != s)))
    return rows


def summarize(rows):
    pres = {}
    for c in CATEGORIES:
        v = [r["cats"][c] for r in rows if r["cats"][c] is not None]
        pres[c] = dict(rate=(sum(v) / len(v) if v else None), n=len(v))
    f = lambda k: float(np.mean([r[k] for r in rows]))
    return dict(n=len(rows), unsafe_rate=f("unsafe"), preservation=pres, S_gain=f("S_after") - f("S_before"), edits=f("edits"),
                rejected=f("rejected"), fallback_rate=f("fallback"), changed_rate=f("changed"))


def run_2x2(sentences, **kw):
    return {f"{k}|verifier_{'on' if on else 'off'}": summarize(run_cell(sentences, k, on, **kw))
            for k in ("strong", "weak") for on in (True, False)}


def fmt_table(cells, cols=CATEGORIES):
    h = f"{'cell':24}{'unsafe':>8}" + "".join(f"{c:>13}" for c in cols) + f"{'S gain':>8}{'edits':>7}{'rej':>6}{'fallb':>7}"
    out = [h]
    for name, s in cells.items():
        pr = "".join(f"{('n/a' if s['preservation'][c]['rate'] is None else '%.1f%%' % (100*s['preservation'][c]['rate'])):>13}" for c in cols)
        out.append(f"{name:24}{100*s['unsafe_rate']:7.1f}%{pr}{s['S_gain']:8.3f}{s['edits']:7.2f}{s['rejected']:6.2f}{100*s['fallback_rate']:6.1f}%")
    return "\n".join(out)
