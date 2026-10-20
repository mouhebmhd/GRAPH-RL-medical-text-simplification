import re
import zlib
import numpy as np

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")


def syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))


def hash_vec(tokens, dim: int, signed: bool = False) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        h = zlib.crc32(t.encode("utf8"))
        sign = 1.0 if (not signed or (h >> 31) & 1 == 0) else -1.0
        v[h % dim] += sign
    return v


def clean_spacing(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])\s*([.!?])", r"\2", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"^[,;:\s]+", "", s)
    return s


def capitalize_first(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s
