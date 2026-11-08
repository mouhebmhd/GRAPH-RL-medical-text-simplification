"""P0.2 - number/unit normalisation shared by the verifier and the strict metrics.

Makes these equivalent before any comparison:  50mg == 50 mg == 50 milligrams ;  4,000 == 4000 ;  0.50 == 0.5 ;  500 mcg == 500 µg.
Deliberately NOT done here (would need a decision, see README): number words ('three' -> 3), unit conversion (1 g -> 1000 mg)."""
from __future__ import annotations
import re
from collections import Counter

# raw surface (lowercase) -> canonical unit
_UNIT_MAP = {
    "mg": "mg", "milligram": "mg", "milligrams": "mg", "mgs": "mg",
    "mcg": "mcg", "µg": "mcg", "μg": "mcg", "ug": "mcg", "microgram": "mcg", "micrograms": "mcg",
    "g": "g", "gram": "g", "grams": "g", "gm": "g", "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml", "cc": "ml",
    "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "unit": "units", "units": "units", "u": "units", "iu": "iu",
    "tablet": "tablet", "tablets": "tablet", "tab": "tablet", "tabs": "tablet", "capsule": "capsule", "capsules": "capsule",
    "pill": "pill", "pills": "pill", "drop": "drop", "drops": "drop", "puff": "puff", "puffs": "puff",
    "mmhg": "mmhg", "%": "%",
    "hour": "hour", "hours": "hour", "hr": "hour", "hrs": "hour", "h": "hour",
    "day": "day", "days": "day", "week": "week", "weeks": "week", "month": "month", "months": "month",
    "year": "year", "years": "year", "minute": "minute", "minutes": "minute", "min": "minute", "mins": "minute",
}
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_GLUED = re.compile(r"(?<=\d)(?=[A-Za-zµμ%])")
_NUM_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(%|[A-Za-zµμ]+)?")


def normalize_text(text: str) -> str:
    """Thousands separators removed, digits and glued units separated, whitespace collapsed."""
    t = _THOUSANDS.sub("", text or "")
    t = _GLUED.sub(" ", t)
    return re.sub(r"\s+", " ", t)


def normalize_number_unit(text: str) -> list:
    """Multiset (as a list) of (value, canonical_unit). Unit is '' unless the word after the number is a known unit."""
    out = []
    for m in _NUM_UNIT.finditer(normalize_text(text)):
        out.append((float(m.group(1)), _UNIT_MAP.get((m.group(2) or "").lower(), "")))
    return out


def number_multiset(text: str) -> Counter:
    return Counter(normalize_number_unit(text))


def fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"
