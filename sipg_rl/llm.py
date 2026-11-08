"""Free-LLM backend for the span rewriter.

Speaks the OpenAI-compatible /chat/completions protocol, which covers:
  * Ollama (local, free, no key)        base_url=http://localhost:11434/v1   model=qwen2.5:3b  (default)
  * LM Studio / llama.cpp server        base_url=http://localhost:1234/v1
  * Free hosted tiers (Groq, Gemini...) their OpenAI-compatible base_url + an API key (text leaves your machine)
Stdlib only. The LLM never decides *where* to edit (the graph policy does) and never has the last word on safety
(the verifier does): it only proposes a replacement for one span."""
from __future__ import annotations
import json, os, re, time, urllib.request, urllib.error

SYSTEM = (
    "You are an editor who simplifies ONE span of a clinical sentence for a patient with low health literacy. "
    "Rules: keep the exact meaning. Never change, drop or add numbers, doses, units, drug names, negations "
    "(no, not, denies, without) or hedges (may, possible, if, unless). Use short, everyday words. "
    "If the span is already plain or cannot be simplified safely, return it unchanged. "
    "Output ONLY the replacement span: no quotes, no explanation, no extra sentences."
)


SYSTEM_LEXICAL = (
    "You replace ONE word or short phrase in a clinical sentence with everyday words a patient understands. "
    "Return ONLY the replacement, same part of speech, at most 4 words. Do NOT restructure the sentence, add information, "
    "or change numbers, doses, units, drug names, negations (no, not, without) or hedges (may, possible). "
    "If no safe plain replacement exists, return the original span unchanged.")
SYSTEM_CLAUSE = (
    "You split ONE long clause of a clinical sentence into shorter sentences. Reuse the original words; only add a subject "
    "or a full stop where needed. Do NOT paraphrase, drop, or add facts. Keep every number, dose, unit, drug name, negation "
    "(no, not, without), hedge (may, possible) and condition (if, unless) exactly. Output ONLY the rewritten text on one line.")


def sanitize(raw: str, span: str) -> str:
    """Reduce a chatty model answer to a bare replacement span ('' = unusable -> treated as no change)."""
    s = (raw or "").strip()
    s = re.sub(r"^(replacement span|replacement|answer|output|rewritten span)\s*:\s*", "", s, flags=re.I)
    s = s.splitlines()[0].strip() if s else ""
    s = s.strip("`\"'“”‘’ ").strip()
    if not span.rstrip().endswith("."):
        s = s.rstrip(".")
    if not s or len(s.split()) > 2 * len(span.split()) + 6:   # empty or an explanation, not a span
        return ""
    return s


EXPLAIN_SYSTEM = (
    "You write one short glossary entry for a patient with low health literacy. Output ONLY a JSON object: "
    '{"type": "body|medicine|class|condition|test|other", "definition": "...", "extra": "..."}. '
    "definition: what the word means in plain everyday language, at most 25 words. "
    "extra: one useful general fact (what a medicine is used for, what a body part does, when to seek care), "
    "at most 30 words, or an empty string. Never give doses or personal advice. "
    'If you are not sure what the word means, use {"type": "other", "definition": "", "extra": ""}.'
)
_EXPLAIN_TYPES = {"body", "medicine", "class", "condition", "test", "other"}


def parse_explanation(raw: str) -> dict:
    """Pull {type, definition, extra} out of a model answer; {} when unusable (never raises)."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    clip = lambda v, n: " ".join(str(v or "").split())[:n].strip()
    out = dict(type=d.get("type") if d.get("type") in _EXPLAIN_TYPES else "other",
               definition=clip(d.get("definition"), 220), extra=clip(d.get("extra"), 260))
    return out if out["definition"] else {}


class OpenAICompatLLM:
    def __init__(self, base_url="http://localhost:11434/v1", model="qwen2.5:3b", api_key=None,
                 timeout=90, temperature=0.0, max_tokens=80, cache_path=None):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.timeout, self.temperature, self.max_tokens = timeout, temperature, max_tokens
        self.cache, self.cache_path = {}, cache_path
        self.lexical_context_free = True
        if cache_path and os.path.exists(cache_path):                 # Task 23: reproducible, cheap re-runs
            for line in open(cache_path):
                try:
                    k, v = json.loads(line); self.cache[tuple(k)] = v
                except Exception:
                    pass
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(calls=0, cache_hits=0, errors=0, seconds=0.0, last_error=None)

    # ------------------------------------------------------------------------------------------
    def _request(self, path, payload=None, timeout=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, None if payload is None else json.dumps(payload).encode(),
                                     headers)
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.load(r)

    def health(self) -> dict:
        info = dict(configured=True, model=self.model, base_url=self.base_url, reachable=False,
                    model_available=None, error=None)
        try:
            data = self._request("/models", timeout=3)
            names = [m.get("id") or m.get("name") for m in data.get("data", [])]
            info["reachable"] = True
            info["model_available"] = (self.model in names or any(n and n.split(":")[0] == self.model.split(":")[0]
                                                                  and ":" not in self.model for n in names)) if names else None
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
        return info

    def __call__(self, prompt: str, text: str, context: str = "", mode: str | None = None) -> str:
        """mode: None (legacy span rewrite) | 'lexical' (<=4-word swap) | 'clause' (split only). Lexical answers are cached
        per span text, ignoring the sentence (so repeated terms hit the cache); set lexical_context_free=False to disable."""
        system = {"lexical": SYSTEM_LEXICAL, "clause": SYSTEM_CLAUSE}.get(mode, SYSTEM)
        if mode == "lexical" and getattr(self, "lexical_context_free", True):
            key = ("lexical", text.strip().lower())
        else:
            key = (mode, prompt, text, context)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        user = (f"Sentence: {context or text}\nSpan to rewrite: {text}\nInstruction: {prompt}\nReplacement span:")
        t0 = time.time()
        try:
            data = self._request("/chat/completions", dict(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, stream=False,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]))
            out = sanitize(data["choices"][0]["message"]["content"], text)
        except Exception as e:                      # never crash the pipeline: no change + visible error
            self.stats["errors"] += 1
            self.stats["last_error"] = f"{type(e).__name__}: {e}"
            return ""
        finally:
            self.stats["calls"] += 1
            self.stats["seconds"] += time.time() - t0
        self.cache[key] = out
        if self.cache_path:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps([list(key), out]) + "\n")
        return out

    def explain(self, term: str, context: str = "") -> dict:
        """Short patient-friendly glossary entry for one term (cached). {} = model unsure / unreachable."""
        key = ("explain", term.lower(), context)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        user = f"Term: {term}" + (f"\nSentence it appears in: {context}" if context else "")
        try:
            data = self._request("/chat/completions", dict(
                model=self.model, temperature=0.0, max_tokens=200, stream=False,
                messages=[{"role": "system", "content": EXPLAIN_SYSTEM}, {"role": "user", "content": user}]))
            out = parse_explanation(data["choices"][0]["message"]["content"])
        except Exception as e:
            self.stats["errors"] += 1
            self.stats["last_error"] = f"{type(e).__name__}: {e}"
            return {}
        self.cache[key] = out
        return out
