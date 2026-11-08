"""P0.3 - example-level hazard injectors for measuring the safety gate.

Each injector takes examples [{src, out, ...}] and returns (perturbed_examples, n_injected): `out` is perturbed as if a
realizer had produced an unsafe edit. They RAISE when they cannot fire (empty vocabulary or zero injections) so a
mis-configured injector can never silently report 'injected = 0, leaked = 0'."""
from __future__ import annotations
import re
from .lexicon import CRITICAL_TERMS, NEGATION_CUES, HEDGE_CUES, COND_CUES

ANTONYMS = [("increase", "decrease"), ("increased", "decreased"), ("increases", "decreases"), ("high", "low"),
            ("higher", "lower"), ("raise", "lower"), ("before", "after"), ("left", "right"), ("mild", "severe"),
            ("benign", "malignant"), ("normal", "abnormal"), ("start", "stop"), ("avoid", "take"), ("safe", "unsafe"),
            ("hypertension", "hypotension"), ("hyperglycemia", "hypoglycemia"), ("always", "never"), ("often", "rarely"),
            ("more", "less"), ("some", "all"), ("children", "adults"), ("improved", "worsened"), ("worse", "better")]


def _case(repl: str, orig: str) -> str:
    return repl.capitalize() if orig[:1].isupper() else repl


def _sub_first(text, pattern, fn):
    m = re.search(pattern, text, re.I)
    return (text[:m.start()] + fn(m) + text[m.end():], True) if m else (text, False)


def _run(name, examples, edit, vocab=None):
    if vocab is not None and not vocab:
        raise ValueError(f"{name}: empty vocabulary - injector cannot fire")
    out, n = [], 0
    for ex in examples:
        new, hit = edit(ex["out"])
        out.append(dict(ex, out=new, injected=hit, inject_kind=name if hit else None))
        n += hit
    if n == 0:
        raise RuntimeError(f"{name}: zero injections out of {len(examples)} examples - check the matcher / the data")
    return out, n


def inject_swap_drug(examples, drug_lexicon=None):
    lex = sorted(CRITICAL_TERMS if drug_lexicon is None else drug_lexicon)
    pat = r"\b(" + "|".join(map(re.escape, lex)) + r")\b" if lex else r"$^"

    def edit(t):
        def sw(m):
            other = next(d for d in lex if d != m.group(1).lower())
            return _case(other, m.group(1))
        return _sub_first(t, pat, sw)
    return _run("swap_drug", examples, edit, vocab=lex)


def inject_dose_x10(examples):
    from .normalize import fmt
    return _run("dose_x10", examples, lambda t: _sub_first(t, r"\b\d+(?:\.\d+)?\b", lambda m: fmt(round(float(m.group(0)) * 10, 6))))


def inject_drop_negation(examples):
    cues = sorted(c for c in NEGATION_CUES if c != "n't")
    return _run("drop_negation", examples, lambda t: _sub_first(t, r"\b(" + "|".join(cues) + r")\b\s*", lambda m: ""), vocab=cues)


def inject_drop_hedge(examples):
    cues = sorted(HEDGE_CUES)
    return _run("drop_hedge", examples, lambda t: _sub_first(t, r"\b(" + "|".join(cues) + r")\b\s*", lambda m: ""), vocab=cues)


def inject_drop_conditional(examples):
    cues = sorted(COND_CUES)
    return _run("drop_conditional", examples, lambda t: _sub_first(t, r"\b(" + "|".join(cues) + r")\b\s*", lambda m: ""), vocab=cues)


def inject_antonym(examples):
    """Meaning flip with no number / negation / hedge / drug change: the lexical gate is expected to LEAK on this."""
    amap = {}
    for a, b in ANTONYMS:
        amap[a], amap[b] = b, a
    pat = r"\b(" + "|".join(sorted(map(re.escape, amap), key=len, reverse=True)) + r")\b"
    return _run("antonym", examples, lambda t: _sub_first(t, pat, lambda m: _case(amap[m.group(1).lower()], m.group(1))), vocab=list(amap))


INJECTORS = {"swap_drug": inject_swap_drug, "dose_x10": inject_dose_x10, "drop_negation": inject_drop_negation,
             "drop_hedge": inject_drop_hedge, "drop_conditional": inject_drop_conditional, "antonym": inject_antonym}


def gate_report(examples, verifier, nli_gate=None):
    """For injected examples: how many did the gate reject (caught) vs accept (leaked)?
    Examples whose UNPERTURBED output already fails the gate are excluded (their rejection is not the injection's doing)."""
    inj = [e for e in examples if e.get("injected")]
    base_bad = caught = 0
    for e in inj:
        base_ok = verifier.check(e["src"], e["orig_out"], use_nli=False).ok if "orig_out" in e else True
        if not base_ok:
            base_bad += 1
            continue
        rej = not verifier.check(e["src"], e["out"], use_nli=False).ok
        if not rej and nli_gate is not None:
            rej = nli_gate(e["src"], e["out"])
        caught += rej
    n = len(inj) - base_bad
    return dict(injected=len(inj), excluded_base_rejected=base_bad, evaluated=n, caught=caught, leaked=n - caught,
                recall=caught / n if n else float("nan"))
