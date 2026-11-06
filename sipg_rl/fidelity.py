"""Length-invariant Sim for POST-HOC evaluation only (audit item 2). Not used in the training reward (beta = 0).

NLISim     : bidirectional entailment. fwd = P(entail | premise=src, hypothesis=out) (nothing invented),
             bwd = P(entail | premise=out, hypothesis=src) (nothing lost). sim = mean (agg='mean') or min (agg='min').
BERTScoreSim: BERTScore F1 of out against src (needs `bert_score`; pass model + num_layers explicitly for deberta-v3-large).
Neither can rank `copy` below the references against the SOURCE (copy == source is the maximum); they are meant to separate
faithful rewrites from corrupted ones, which `discrimination()` measures."""
from __future__ import annotations
import numpy as np


class NLISim:
    def __init__(self, nli, agg="mean"):
        self.nli, self.agg, self._c = nli, agg, {}

    def parts(self, src, out):
        k = (src, out)
        if k not in self._c:
            if src.strip() == out.strip():
                self._c[k] = (1.0, 1.0)
            else:
                self._c[k] = (float(self.nli(src, out).get("entailment", 0.0)), float(self.nli(out, src).get("entailment", 0.0)))
        return self._c[k]

    def __call__(self, src, out):
        f, b = self.parts(src, out)
        return min(f, b) if self.agg == "min" else 0.5 * (f + b)


class BERTScoreSim:
    def __init__(self, model="microsoft/deberta-xlarge-mnli", num_layers=None):
        from bert_score import BERTScorer
        self.sc = BERTScorer(model_type=model, num_layers=num_layers, lang="en")

    def __call__(self, src, out):
        return float(self.sc.score([out], [src])[2][0])


def discrimination(sim, pairs, corrupt_fns):
    """pairs: [(src, faithful_out)]; corrupt_fns: {name: f(text)->text}. -> {name: mean(sim(faithful) - sim(corrupted))}."""
    res = {}
    for name, fn in corrupt_fns.items():
        d = [sim(s, o) - sim(s, fn(o)) for s, o in pairs if fn(o) != o]
        res[name] = (float(np.mean(d)) if d else None, len(d))
    return res
