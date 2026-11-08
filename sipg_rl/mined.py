"""Mined lexicon (jargon -> everyday wording) learned from PARALLEL TEXT YOU SUPPLY, with a hard blocklist (Tasks 16, 27).
Safety rule: drug names, doses/units, numbers, procedure codes, dosing abbreviations and anything that flips negation / hedge /
conditional polarity NEVER enter, however often they align. `audit_lexicon` re-checks a finished file independently of the miner."""
from __future__ import annotations
import difflib, json, re
from collections import Counter, defaultdict
from .lexicon import PLAIN_TERMS, CRITICAL_TERMS, NEGATION_CUES, HEDGE_CUES, COND_CUES, UNITS
from .utils import syllables

_WORD = re.compile(r"[a-z]+(?:[-'][a-z]+)*|\d+(?:[.,]\d+)*")
_CODE = re.compile(r"\b[a-tv-z]\d{2}(?:\.\d{1,4})?\b|\b\d{4,5}\b|\b(?:icd|cpt|snomed|loinc)\b", re.I)
_DOSE_ABBR = {"bid", "tid", "qid", "qd", "prn", "qhs", "po", "iv", "im", "sc", "sl", "pr", "stat", "qod", "ac", "pc", "hs"}
_UNIT_RE = re.compile(rf"\b{UNITS}\b", re.I)
_FUNC = set("the a an of to and or in on at by for with as is are was were be been it this that these those".split())


def toks(s):
    return _WORD.findall((s or "").lower())


class Blocklist:
    def __init__(self, extra_terms=()):
        self.terms = set(CRITICAL_TERMS) | {t.lower() for t in extra_terms}

    def reason(self, jargon, layman):
        for side, txt in (("jargon", jargon), ("layman", layman)):
            w = toks(txt)
            if any(c.isdigit() for c in txt): return f"{side} contains a number"
            if _UNIT_RE.search(txt): return f"{side} contains a dose unit"
            if _CODE.search(txt): return f"{side} looks like a procedure/diagnosis code"
            if any(t in self.terms for t in w): return f"{side} contains a drug / critical term"
            if any(t in _DOSE_ABBR for t in w): return f"{side} contains a dosing/route abbreviation (curated tier only)"
        jw, lw = toks(jargon), toks(layman)
        if not jw or not lw: return "empty side"
        if jw == lw: return "identity"
        if len(jargon) < 4 or all(t in _FUNC for t in jw): return "jargon too short / function words only"
        if len(lw) > 8: return "layman side too long (explanation, not a substitution)"
        cnt = lambda w, cues: sum(1 for t in w if t in cues)
        for name, cues in (("negation", NEGATION_CUES), ("hedge", HEDGE_CUES), ("conditional", COND_CUES)):
            if cnt(jw, cues) != cnt(lw, cues): return f"changes {name} polarity"
        return None


def align_changes(src, ref, max_src=5, max_ref=8):
    a, b = toks(src), toks(ref)
    return [(" ".join(a[i1:i2]), " ".join(b[j1:j2])) for tag, i1, i2, j1, j2 in
            difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            if tag == "replace" and 0 < i2 - i1 <= max_src and 0 < j2 - j1 <= max_ref]


def _jargonish(span):
    return any(len(t) >= 7 or syllables(t) >= 3 for t in toks(span))


def pairs_from_row(src, ref, expert_terms=None, layman_terms=None, max_src=3):
    """Conservative: (1) annotated expert/layman lists of equal length paired by position, each verified in its text;
    (2) else an alignment region counts only if its source side EQUALS an annotated expert term (or, with no annotations, is
    <= max_src tokens and looks like jargon). Merged multi-change regions are discarded rather than guessed."""
    s_low, r_low = " ".join(toks(src)), " ".join(toks(ref))
    et = [" ".join(toks(t)) for t in (expert_terms or []) if toks(t)]
    lt = [" ".join(toks(t)) for t in (layman_terms or []) if toks(t)]
    if et and lt and len(et) == len(lt):
        out = [(e, l) for e, l in zip(et, lt) if f" {e} " in f" {s_low} " and f" {l} " in f" {r_low} " and e != l]
        if out:
            return out
    ch = align_changes(src, ref, max_src=max(max_src, 5))
    if et:
        return [(j, l) for j, l in ch if j in et]
    return [(j, l) for j, l in ch if len(toks(j)) <= max_src and _jargonish(j) and len(toks(l)) <= 5]


class MinedLexicon:
    def __init__(self):
        self.cands, self.rejected = defaultdict(Counter), Counter()

    def add_row(self, src, ref, expert_terms=None, layman_terms=None, block=None):
        block = block or Blocklist()
        for j, l in pairs_from_row(src, ref, expert_terms, layman_terms):
            r = block.reason(j, l)
            if r: self.rejected[r] += 1
            else: self.cands[j][l] += 1

    def top(self, jargon, k=5):
        return [w for w, _ in self.cands.get(jargon, Counter()).most_common(k)]

    def ranked(self, jargon):
        return self.cands.get(jargon, Counter()).most_common()

    def export_rules(self, min_count=2, min_share=0.5):
        out = {}
        for j, c in self.cands.items():
            (w, n), = c.most_common(1)
            if n >= min_count and n / sum(c.values()) >= min_share:
                out[j] = (w, "term")
        return out

    def save(self, path):
        json.dump({"candidates": {j: c.most_common() for j, c in self.cands.items()}, "rejected": dict(self.rejected)}, open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        d, m = json.load(open(path)), cls()
        for j, lst in d["candidates"].items(): m.cands[j] = Counter({w: n for w, n in lst})
        m.rejected = Counter(d.get("rejected", {}))
        return m


def audit_lexicon(cands, extra_blocked=()):
    """Independent re-check of a finished lexicon ({jargon: [layman,...] | [(layman,n),...]}). [] == pass."""
    bl, bad = Blocklist(extra_blocked), []
    for j, lay in cands.items():
        for l in lay:
            l = l[0] if isinstance(l, (tuple, list)) else l
            r = bl.reason(j, l)
            if r: bad.append((j, l, r))
        if j in bl.terms: bad.append((j, "", "jargon key is a critical term"))
    return bad


def load_mined(path, min_count=2, min_share=0.5):
    """Merge a mined lexicon into lexicon.PLAIN_TERMS (tier 2; curated tier-1 entries win) and lexicon.CANDIDATES (ranked lists).
    Refuses a file that fails the blocklist audit. Returns the number of entries added."""
    from . import lexicon
    m = MinedLexicon.load(path)
    rules = m.export_rules(min_count, min_share)
    bad = audit_lexicon({j: m.ranked(j) for j in rules})
    if bad:
        raise ValueError(f"mined lexicon {path} violates the blocklist ({len(bad)} entries, first: {bad[0]})")
    added = 0
    for j, (w, cat) in rules.items():
        lexicon.CANDIDATES[j] = m.ranked(j)
        if j not in PLAIN_TERMS:
            PLAIN_TERMS[j] = (w, cat); lexicon.TIER2.add(j); added += 1
    try:
        from . import scoring; scoring.refresh_jargon()
    except Exception:
        pass
    return added
