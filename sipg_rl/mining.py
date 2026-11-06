"""Lexicon mining from parallel (expert, simple) pairs (audit items 7 and 10).

Why: with the 127-entry hand lexicon the RuleRealizer leaves 77.8% of jargon in place (copy: 79.4%) and reaches a layman-term
hit rate of 48.8% against 81.1% for the references.  The references already contain the answer: for every expert term the
annotators chose a plain wording.  This module extracts those choices, counts them, and keeps the ones that are frequent,
consistent and SAFE (no drug names, numbers, doses, codes: `lexicon.lexicon_entry_problem`).

Leakage rule: mining uses the TRAIN split only.  `mine()` records the split in its metadata and `scripts/mine_lexicon.py`
refuses any other split; the recall diagnostic is computed on a different (held-out) split.

Two alignment sources, used in this order per example:
  1. the dataset's own `expert_terms` / `layman_terms` annotation, when both parse and have equal length (index-paired);
  2. a token-level diff (difflib) of source vs reference: a 'replace' block whose source side is 1-3 tokens containing a
     jargon-looking word and whose reference side is 1-6 tokens.
"""
from __future__ import annotations
import difflib, json, os, re
from collections import Counter, defaultdict
from . import metrics as M
from .lexicon import lexicon_entry_problem, CRITICAL_TERMS

_TOK = re.compile(r"[A-Za-z][A-Za-z\-']*|\d+(?:\.\d+)?|[^\sA-Za-z\d]")
_GREEK = re.compile(r"(?:itis|emia|aemia|osis|opathy|ectomy|otomy|plasty|scopy|algia|uria|pathy|genic|logy|iatric|cardi|hepat|nephr|pulmon|"
                    r"gastr|derm|neur|hyper|hypo|dys|brady|tachy|angi|arthr|myo|oste|cyst|rhin|ophth|oto)", re.I)
_COMMON = set("the a an of to and or in on for with by at as is are was were be been it its this that these those which who whom "
              "has have had not no may can will would should could also than then".split())


def toks(text: str) -> list:
    return [t.lower() for t in _TOK.findall(text or "")]


def jargon_like(word: str) -> bool:
    return len(word) >= 9 or bool(_GREEK.search(word))


def _norm_phrase(ws) -> str:
    return " ".join(ws).strip()


def pairs_from_annotation(raw: dict, src: str, ref: str):
    et, lt = M.parse_terms(raw.get("expert_terms")), M.parse_terms(raw.get("layman_terms"))
    if et and lt and len(et) == len(lt):
        low_s, low_r = (src or "").lower(), (ref or "").lower()
        return [(e, l, "annotation") for e, l in zip(et, lt) if e in low_s and l in low_r and e != l]
    return []


def pairs_from_diff(src: str, ref: str, max_src=3, max_ref=6):
    a, b = toks(src), toks(ref)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "replace":
            continue
        s, r = a[i1:i2], b[j1:j2]
        if 1 <= len(s) <= max_src and 1 <= len(r) <= max_ref and any(jargon_like(w) for w in s) and not (set(s) & _COMMON == set(s)):
            out.append((_norm_phrase(s), _norm_phrase(r), "diff"))
    return out


def example_pairs(ex: dict):
    """ex: {'src','ref','raw'} -> [(expert phrase, layman phrase, how)]"""
    return pairs_from_annotation(ex.get("raw") or {}, ex["src"], ex["ref"]) or pairs_from_diff(ex["src"], ex["ref"])


def mine(rows, split="train", min_count=2, min_share=0.4, max_alts=5):
    """-> (entries, report).  entries: key -> {'plain': top-1, 'alts': ranked list, 'count': n, 'share': top-1 share}.
    Candidates are ranked by frequency across `rows`; unsafe pairs are dropped and listed in the report."""
    cnt = defaultdict(Counter)
    for ex in rows:
        for e, l, _how in example_pairs(ex):
            cnt[e][l] += 1
    entries, rejected, dropped_low = {}, [], 0
    for key, c in cnt.items():
        n = sum(c.values())
        ranked = []
        for plain, k in c.most_common():
            prob = lexicon_entry_problem(key, plain, "term")
            if prob:
                rejected.append((key, plain, prob))
                continue
            ranked.append((plain, k))
        if not ranked:
            continue
        top, kt = ranked[0]
        if kt < min_count or kt / n < min_share or key in CRITICAL_TERMS or len(key.split()) > 3:
            dropped_low += 1
            continue
        entries[key] = dict(plain=top, alts=[p for p, _ in ranked[1:max_alts]], count=n, share=round(kt / n, 3))
    rep = dict(split=split, n_rows=len(rows), n_candidate_keys=len(cnt), n_entries=len(entries), n_unsafe_dropped=len(rejected),
               n_below_threshold=dropped_low, min_count=min_count, min_share=min_share, unsafe_examples=rejected[:15])
    return entries, rep


def save(entries, report, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(dict(meta=report, entries=entries), open(path, "w", encoding="utf8"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------------------------------------------------------------
# Item 10: does the mined lexicon contain the reference's own choice?  (computed on a HELD-OUT split)
# ---------------------------------------------------------------------------------------------------------------------
def _soft_match(a: str, b: str) -> bool:
    ta, tb = set(a.split()), set(b.split())
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.5


def recall_at_k(rows, entries, ks=(1, 3, 5), kind_of=None):
    """For every reference substitution (expert phrase -> layman phrase) found in `rows`:
         covered  : the expert phrase has ANY mined entry
         exact@k  : the reference's layman phrase is among the mined top-k for that phrase
         soft@k   : same, with token-overlap (Jaccard >= 0.5) instead of equality
    kind_of(ex, phrase) -> node type (LEX/NOM/AMOD/CPLX/OTHER) for the per-type breakdown."""
    stats = defaultdict(lambda: Counter())
    for ex in rows:
        for e, l, _how in example_pairs(ex):
            kind = kind_of(ex, e) if kind_of else "ALL"
            for K in (kind, "ALL"):
                st = stats[K]
                st["n"] += 1
                ent = entries.get(e)
                if not ent:
                    continue
                st["covered"] += 1
                cands = [ent["plain"]] + list(ent.get("alts", []))
                for k in ks:
                    top = cands[:k]
                    st[f"exact@{k}"] += int(l in top)
                    st[f"soft@{k}"] += int(any(_soft_match(l, c) for c in top))
    out = {}
    for K, st in stats.items():
        n = st["n"]
        out[K] = dict(n=n, covered=st["covered"] / n if n else float("nan"),
                      **{f"exact@{k}": st[f"exact@{k}"] / n if n else float("nan") for k in ks},
                      **{f"soft@{k}": st[f"soft@{k}"] / n if n else float("nan") for k in ks})
    return out
