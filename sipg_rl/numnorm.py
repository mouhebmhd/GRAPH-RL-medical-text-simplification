"""Number / unit normalisation for the fidelity verifier (audit item 3).

Why this exists.  The first verifier compared the raw strings "(number, unit)" of source and output.  Probing it showed
that "50mg" -> "50 mg" was NOT the problem (the old regex already allowed optional whitespace).  What it did reject, on
perfectly faithful rewrites, was:

    10 mg   -> 10 milligrams        (spelled-out unit)         5 mL  -> 5 milliliters
    7 days  -> a 7-day period       (hyphenated / singular)    20%   -> 20 percent   (worked only by accident)
    1,000 mg-> 1000 mg              (thousands separator: split into "1" and "000")
    2 tablets -> two tablets        (spelled number)           0.5 mg -> .5 mg        (leading-dot decimal)
    1/2 tablet -> half a tablet     (fraction)                 twice a day -> 2 times a day (frequency wording)

`canon_numbers(text)` maps all of these to one canonical (value, unit) form on BOTH sides before comparison.
It stays strict where it matters: a different value (10 -> 20), a different unit (mg -> g, mg -> tablets), or a number
that is simply gone is still a violation.  Spelled numbers are only converted when they sit next to a unit/dose noun, so
"one of the patients" is not read as a dose.
"""
from __future__ import annotations
import re
from collections import Counter
from decimal import Decimal, InvalidOperation

# ---- canonical unit forms -----------------------------------------------------------------------------------------
_UNIT_LONG = [  # (regex, canonical)  -- longest / most specific first
    (r"millimet(?:er|re)s?\s+of\s+mercury|mm\s*hg", "mmhg"),
    (r"international\s+units?", "iu"),
    (r"milligram(?:me)?s?", "mg"), (r"microgram(?:me)?s?", "mcg"), (r"kilogram(?:me)?s?", "kg"),
    (r"gram(?:me)?s?", "g"), (r"milliliters?|millilitres?", "ml"), (r"liters?|litres?", "l"),
    (r"per\s*cent|percent", "%"),
]
_UNIT_LONG_RX = [(re.compile(rf"(?<![a-z])(?:{p})(?![a-z])", re.I), c) for p, c in _UNIT_LONG]
# units the verifier tracks (singular canonical forms; a trailing s is stripped)
UNIT_RX = r"(?:mg|mcg|g|kg|ml|l|units?|iu|tablets?|capsules?|pills?|drops?|puffs?|%|mmhg|hours?|days?|weeks?|months?)"
_EXTRA_AFTER_NUMBER = r"(?:times?|doses?|years?|minutes?|injections?|sprays?|patches|patch|packets?|vials?|cups?|glass(?:es)?)"

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_WORDNUM = "|".join(sorted(list(_ONES) + list(_TENS), key=len, reverse=True))
_AFTER = rf"(?=[\s-]*(?:{UNIT_RX}|{_EXTRA_AFTER_NUMBER})(?![a-z]))"
_COMPOUND = re.compile(rf"\b({'|'.join(_TENS)})[\s-]+({'|'.join(k for k in _ONES if _ONES[k] < 10 and _ONES[k] > 0)})\b{_AFTER}", re.I)
_HUNDRED = re.compile(rf"\b({'|'.join(k for k in _ONES if 0 < _ONES[k] < 10)})\s+hundred\b{_AFTER}", re.I)
_SINGLE = re.compile(rf"\b({_WORDNUM})\b{_AFTER}", re.I)
_FRACTION = re.compile(r"(?<![\d/])(\d+)\s*/\s*(\d+)(?![\d/])")
_HALF = re.compile(rf"\b(?:(?:a|one)[\s-]+)?(half|quarter)(?:\s+of)?(?:\s+(?:an?|the))?\s+(?={UNIT_RX}(?![a-z]))", re.I)

_NUM = re.compile(rf"(?<![\w.])(?<![A-Za-z]-)(\d+(?:\.\d+)?)(?:\s*({UNIT_RX})(?![a-z]))?", re.I)

# frequency wording that may legitimately become "N times": src word -> (value, unit) the output may add
FREQ_ALLOW = {"twice": ("2", "time"), "bid": ("2", "time"), "thrice": ("3", "time"), "tid": ("3", "time"),
              "once": ("1", "time"), "qd": ("1", "time"), "daily": ("1", "time")}


def _dec(s: str) -> str:
    try:
        d = Decimal(s).normalize()
        out = format(d, "f")
        return out
    except InvalidOperation:
        return s


def canon_text(text: str) -> str:
    """Rewrite `text` so equivalent numeric expressions look identical (used only for comparison, never shown)."""
    t = text
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)                          # 1,000 -> 1000
    t = re.sub(r"(?<![\d\w])\.(\d)", r"0.\1", t)                            # .5 -> 0.5
    for rx, c in _UNIT_LONG_RX:                                             # milligrams -> mg, percent -> %
        t = rx.sub(c, t)
    t = _FRACTION.sub(lambda m: _dec(str(int(m.group(1)) / int(m.group(2)))) if int(m.group(2)) else m.group(0), t)
    t = _HALF.sub(lambda m: "0.5 " if m.group(1).lower() == "half" else "0.25 ", t)
    t = _COMPOUND.sub(lambda m: str(_TENS[m.group(1).lower()] + _ONES[m.group(2).lower()]), t)
    t = _HUNDRED.sub(lambda m: str(_ONES[m.group(1).lower()] * 100), t)
    t = _SINGLE.sub(lambda m: str(_ONES.get(m.group(1).lower(), _TENS.get(m.group(1).lower()))), t)
    t = re.sub(r"(?<=\d)\s*-\s*(?=" + UNIT_RX + r"(?![a-z]))", " ", t, flags=re.I)   # 7-day -> 7 day
    return t


def _unit(u: str) -> str:
    u = (u or "").lower()
    return u[:-1] if len(u) > 3 and u.endswith("s") else u                  # days->day, tablets->tablet, mmhg stays


def canon_numbers(text: str) -> Counter:
    """Counter of (normalised value, canonical unit or '')."""
    return Counter((_dec(m.group(1)), _unit(m.group(2))) for m in _NUM.finditer(canon_text(text)))


def allowed_from_frequency(src: str) -> set:
    """(value, 'time') pairs the output may introduce because the source said 'twice a day', 'BID', ..."""
    low = set(re.findall(r"[a-z]+", src.lower()))
    return {FREQ_ALLOW[w] for w in low if w in FREQ_ALLOW}


def compare(src: str, out: str):
    """-> (lost, hallucinated): lists of human-readable strings.  Unit may be absent from the OUTPUT side only for a
    number the source gave WITHOUT a unit (a unit absorbed by a word split is not a dose change)."""
    sn, on = canon_numbers(src), canon_numbers(out)
    lost, new = [], []
    for (num, unit), c in sn.items():
        have = sum(oc for (n2, u2), oc in on.items() if n2 == num and (u2 == unit or not unit))
        if have < c:
            lost.append(f"{num} {unit}".strip())
    src_nums = {n for n, _ in sn}
    allow_vals = {v for v, _ in allowed_from_frequency(src)}     # "twice a day" may become "2 times a day"
    for (num, unit) in on:
        if num not in src_nums and num not in allow_vals:
            new.append(num)
    return lost, new


def legacy_numbers(text: str) -> Counter:
    """The pre-normalisation extraction, kept so before/after rejection rates can be measured on the same data."""
    rx = re.compile(rf"(\d+(?:\.\d+)?)\s*({UNIT_RX})?\b", re.I)
    return Counter((m.group(1), (m.group(2) or "").lower()) for m in rx.finditer(text))


def legacy_compare(src: str, out: str):
    sn, on = legacy_numbers(src), legacy_numbers(out)
    lost = [f"{n} {u}".strip() for (n, u), c in sn.items()
            if sum(oc for (n2, u2), oc in on.items() if n2 == n and (u2 == u or not u)) < c]
    src_nums = {n for n, _ in sn}
    return lost, [n for (n, _) in on if n not in src_nums]


# ---- why did a number check fail?  (for the gold audit: format artefact vs. real change) ---------------------------
def diagnose(src: str, out: str) -> list:
    """For every number the LEGACY check flags, say whether normalisation explains it.  Returns dicts
    {item, status: 'format_artefact'|'still_flagged', cause}.  'still_flagged' = genuinely different / dropped."""
    l_lost, l_new = legacy_compare(src, out)
    n_lost, n_new = compare(src, out)
    rows = []
    for item in l_lost + l_new:
        base = item.split()[0]
        fixed = item not in n_lost and base not in n_new
        cause = "genuine change or dropped number"
        if fixed:
            if re.search(r"\d,\d{3}", src) or re.search(r"\d,\d{3}", out):
                cause = "thousands separator"
            elif re.search(rf"\b(?:{_WORDNUM}|hundred)\b", out, re.I) and not re.search(rf"\b{re.escape(base)}\b", out):
                cause = "spelled-out number"
            elif re.search(r"\d\s*-\s*(?:day|week|month|hour|year)", out, re.I):
                cause = "hyphenated duration"
            elif re.search(r"milligram|microgram|kilogram|gram|milliliter|millilitre|liter|litre|percent|per cent|millimet", out, re.I):
                cause = "spelled-out unit"
            elif re.search(r"/|half|quarter", src + out, re.I):
                cause = "fraction wording"
            elif re.search(r"\btwice|\bonce|\bbid\b|\btid\b", src, re.I):
                cause = "frequency wording"
            elif re.search(r"(?<!\d)\.\d", out):
                cause = "leading-dot decimal"
            else:
                cause = "unit spacing/case"
        rows.append(dict(item=item, status="format_artefact" if fixed else "still_flagged", cause=cause))
    return rows
