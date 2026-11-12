"""Length-invariant fidelity between a source and a rewrite (logged column; optional reward component).

BERTScore-style greedy soft matching: P = mean over OUTPUT tokens of best match in source (anything invented?),
R = mean over SOURCE tokens of best match in output (anything lost?), F1 = harmonic mean. Averages, so no length effect.
Backends share one interface: prepare(src) -> prep ; score_prepared(prep, out) -> {p, r, f1} ; score(src, out).
  SoftTokenFidelity  offline fallback (concept-normalised tokens + char-trigram soft match; WEAK on open paraphrase)
  BertScoreFidelity  real BERTScore: pass `embedder` (tests) or hf_embedder("microsoft/deberta-v3-large")
  NLIFidelity        bidirectional entailment, f1 = min(P(src=>out), P(out=>src))
Fidelity versus the SOURCE is maximised by copying: use it as a constraint/diagnostic, never as the thing to maximise."""
from __future__ import annotations
import re
import numpy as np

_TOK = re.compile(r"§?[a-z0-9_.\-']+")
_KEEP = {"no", "not", "never", "without", "none", "denies", "deny", "cannot", "neither", "nor", "n't",
         "if", "unless", "may", "might", "possible", "possibly", "likely", "probably", "suspected"}
_STOP = set("the a an of to and who has have is are was were this that be been by in on with for it as at or "
            "which their its they them he she his her you your we our can will would should these those from".split())


def _trigrams(w):
    w = f"^{w}$"
    return frozenset(w[i:i + 3] for i in range(len(w) - 2))


class SoftTokenFidelity:
    def __init__(self, normalizer=None, min_sim: float = 0.6):
        if normalizer is None:
            from .scoring import ConceptEncoder
            normalizer = ConceptEncoder()
        self.norm, self.min_sim, self._tri = normalizer._norm, min_sim, {}

    def tokens(self, text):
        return [t for t in _TOK.findall(self.norm(text or "")) if t in _KEEP or t not in _STOP]

    def _sim(self, a, b):
        if a == b:
            return 1.0
        if a[0] == "§" or b[0] == "§" or a[0].isdigit() or b[0].isdigit():
            return 0.0
        ta = self._tri.setdefault(a, _trigrams(a)); tb = self._tri.setdefault(b, _trigrams(b))
        d = 2 * len(ta & tb) / (len(ta) + len(tb))
        return d if d >= self.min_sim else 0.0

    def prepare(self, src):
        return self.tokens(src)

    def score_prepared(self, s, out):
        o = self.tokens(out)
        if not s:
            return dict(p=1.0, r=1.0, f1=1.0)
        if not o:
            return dict(p=0.0, r=0.0, f1=0.0)
        M = np.array([[self._sim(a, b) for b in s] for a in o])
        p, r = float(M.max(1).mean()), float(M.max(0).mean())
        return dict(p=p, r=r, f1=2 * p * r / (p + r) if p + r else 0.0)

    def score(self, src, out):
        return self.score_prepared(self.prepare(src), out)


class BertScoreFidelity:
    def __init__(self, embedder):
        self.embed = embedder

    def _e(self, text):
        E = np.asarray(self.embed(text), dtype=np.float32)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)

    def prepare(self, src):
        return self._e(src)

    def score_prepared(self, S, out):
        if not out.strip():
            return dict(p=0.0, r=0.0, f1=0.0)
        C = self._e(out) @ S.T
        p, r = float(C.max(1).mean()), float(C.max(0).mean())
        return dict(p=p, r=r, f1=2 * p * r / (p + r) if p + r > 0 else 0.0)

    def score(self, src, out):
        return self.score_prepared(self.prepare(src), out)


def hf_embedder(model_name="microsoft/deberta-v3-large", layer=-4):
    """Token embedder from a Hugging Face encoder (needs `transformers`; network on first use)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, mdl = AutoTokenizer.from_pretrained(model_name), AutoModel.from_pretrained(model_name).eval()

    @torch.no_grad()
    def embed(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
        return mdl(**enc, output_hidden_states=True).hidden_states[layer][0][1:-1].cpu().numpy()
    return embed


class NLIFidelity:
    def __init__(self, nli):
        self.nli = nli

    def prepare(self, src):
        return src

    def score_prepared(self, src, out):
        if not out.strip():
            return dict(p=0.0, r=0.0, f1=0.0)
        fwd = float(self.nli(src, out).get("entailment", 0.0))
        bwd = float(self.nli(out, src).get("entailment", 0.0))
        return dict(p=bwd, r=fwd, f1=min(fwd, bwd))

    def score(self, src, out):
        return self.score_prepared(src, out)


def make_fidelity(kind="soft", embed_model="microsoft/deberta-v3-large", nli_model="microsoft/deberta-large-mnli"):
    if kind == "soft":
        return SoftTokenFidelity()
    if kind == "bertscore":
        return BertScoreFidelity(hf_embedder(embed_model))
    if kind == "nli":
        from .verifier import HFNLI
        return NLIFidelity(HFNLI(nli_model))
    raise ValueError(f"unknown fidelity backend {kind!r}")
