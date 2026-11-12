"""Executes an instantiated prompt on a node. `RuleRealizer` is a deterministic stand-in for the small
instruction model so the whole loop runs offline; `CallableRealizer` plugs in any LLM/T5."""
from __future__ import annotations
from . import lexicon as lexicon_mod
from .lexicon import PLAIN_TERMS, ADJ_REL, PERSON_NOUNS
from .editstate import EditState, Extra

SPLIT_ADVCL = {"although": "However, ", "though": "However, ", "while": "However, ",
               "whereas": "However, ", "because": "This is because "}
SPLIT_CC = {"and": "", "but": "However, ", "so": "So, "}


class Realizer:
    def apply(self, graph, node, st: EditState) -> bool:  # True iff the text changed
        raise NotImplementedError


class RuleRealizer(Realizer):
    scorer = None        # optional: S(text) -> float, enables S-ranked choice among several candidates (Task 11)
    last_via = None
    last_info = None     # {candidate_used, candidate_rank, candidate_count, n_candidates, tier} for lexical edits

    def apply(self, graph, node, st):
        self.last_via = self.last_info = None
        fn = getattr(self, "_" + node.kind.lower(), None)
        ok = bool(fn and fn(graph, node, st))
        if ok:
            self.last_via = "rules"
        return ok

    # -- term-level ---------------------------------------------------------------------
    def _lex(self, g, node, st):
        key = node.meta["key"]
        if key not in PLAIN_TERMS or any(i in st.edited for i in node.ids):
            return False
        cands = lexicon_mod.CANDIDATES.get(key) or [(PLAIN_TERMS[key][0], None)]
        chosen, rank, n, count = cands[0][0], 1, len(cands), cands[0][1]
        if n > 1 and self.scorer is not None:
            from .select import select_candidate
            def S_of(w):
                snap = st.snapshot(); st.replace_span(node.ids, w); sc = self.scorer(st.render()); st.restore(snap); return sc
            chosen, rank, n, count = select_candidate(cands, S_of)
        st.replace_span(node.ids, chosen)
        self.last_info = dict(candidate_used=chosen, candidate_rank=rank, candidate_count=count, n_candidates=n,
                              tier=2 if key in getattr(lexicon_mod, "TIER2", ()) else 1)
        return True

    _nom = _lex

    def _amod(self, g, node, st):
        adj = g.doc[node.anchor]
        head = adj.head
        if (adj.lower_ not in ADJ_REL or head.lower_ not in PERSON_NOUNS or adj.i in st.edited):
            return False
        phrase = ADJ_REL[adj.lower_][0]
        verb = "have" if head.tag_ in ("NNS", "NNPS") else "has"
        prev = st.after.get(head.i, [])
        if prev and prev[-1].startswith("who ha"):
            prev[-1] = prev[-1] + f" and {phrase}"
        else:
            st.insert_after(head.i, f"who {verb} {phrase}")
        st.delete(adj.i)
        self.last_info = dict(candidate_used=phrase, candidate_rank=1, candidate_count=None, n_candidates=1, tier=1)
        return True

    # -- clause-level -------------------------------------------------------------------
    @staticmethod
    def _strip_punct(g, ids):
        ids = [i for i in ids]
        while ids and g.doc[ids[-1]].is_punct:
            ids.pop()
        while ids and g.doc[ids[0]].is_punct:
            ids.pop(0)
        return ids

    @staticmethod
    def _drop_commas(g, st, lo, hi):
        """Delete a comma just before/after a moved span (non-restrictive clause punctuation)."""
        doc = g.doc
        if lo - 1 >= g.toks[0][0] and doc[lo - 1].text == "," and (lo - 1) not in st.deleted:
            st.delete(lo - 1)
        if hi + 1 < g.toks[-1][0] + 1 and doc[hi + 1].text == "," and (hi + 1) not in st.deleted:
            st.delete(hi + 1)

    def _relcl(self, g, node, st):
        doc, rel = g.doc, g.doc[node.anchor]
        if any(i in st.moved for i in node.ids):
            return False
        pron = [c for c in rel.children if c.lower_ in ("who", "which", "that")
                and c.dep_ in ("nsubj", "nsubjpass")]
        head = rel.head
        if not pron or head.pos_ not in ("NOUN", "PROPN") or head.i in st.moved:
            return False
        np_ids = [t.i for t in doc[head.left_edge.i:head.i + 1] if not t.is_punct]
        clause = self._strip_punct(g, [i for i in node.ids if i != pron[0].i and i not in st.moved])
        if not clause:
            return False
        self._drop_commas(g, st, node.ids[0], node.ids[-1])
        first = doc[np_ids[0]].lower_
        st.extras.append(Extra("", np_ids, clause, the_first=first in ("a", "an")))
        st.moved.update(node.ids)
        return True

    def _conj_cl(self, g, node, st):
        doc, v = g.doc, g.doc[node.anchor]
        ccs = [c for c in v.head.children if c.dep_ == "cc" and c.i < v.i]
        if not ccs or ccs[-1].lower_ not in SPLIT_CC or any(i in st.moved for i in node.ids):
            return False
        cc = ccs[-1]
        clause = self._strip_punct(g, [i for i in node.ids if i not in st.moved])
        if not clause:
            return False
        st.delete(cc.i)
        if cc.i - 1 >= 0 and doc[cc.i - 1].text == "," and cc.i - 1 not in st.deleted:
            st.delete(cc.i - 1)
        st.extras.append(Extra(SPLIT_CC[cc.lower_], [], clause))
        st.moved.update(node.ids)
        return True

    def _advcl(self, g, node, st):
        doc, v = g.doc, g.doc[node.anchor]
        marks = [c for c in v.children if c.dep_ == "mark" and c.lower_ in SPLIT_ADVCL]
        if not marks or v.i < v.head.i or any(i in st.moved for i in node.ids):
            return False
        mk = marks[0]
        clause = self._strip_punct(g, [i for i in node.ids if i != mk.i and i not in st.moved])
        if not clause:
            return False
        self._drop_commas(g, st, node.ids[0], node.ids[-1])
        st.delete(mk.i)
        st.extras.append(Extra(SPLIT_ADVCL[mk.lower_], [], clause))
        st.moved.update(node.ids)
        return True


import inspect

# Edit-type ownership (audit items 8/9).  Rules own every STRUCTURAL edit (clause splits) and every lexicon hit; the LLM owns only
# span-local LEXICAL substitution for spans the lexicon does not cover.  Neither performs the other's edit type.
RULE_OWNED_KINDS = ('RELCL', 'CONJ_CL', 'ADVCL')          # clause splitting: deterministic, structure-safe
LEXICAL_KINDS = ('LEX', 'NOM', 'CPLX', 'AMOD')
LLM_KINDS = LEXICAL_KINDS                                  # legacy LLM-first path no longer receives CCOMP/ACL (item 8)


class CallableRealizer(Realizer):
    """Adapter for a real LLM/T5 rewriter: `llm(prompt, text) -> str`.
    Rewrites the node's own span in place; the verifier (not the LLM) guarantees safety."""
    def __init__(self, llm, fallback: Realizer | None = None):
        self.llm, self.fallback = llm, fallback
        try:
            self._ctx = len(inspect.signature(llm).parameters) >= 3   # llm(prompt, text, context)
        except (TypeError, ValueError):
            self._ctx = False

    @staticmethod
    def handles(kind: str) -> bool:
        return kind in LLM_KINDS

    def _ask(self, prompt, text, st):
        return self.llm(prompt, text, st.render()) if self._ctx else self.llm(prompt, text)

    def apply(self, graph, node, st):
        if node.kind in LLM_KINDS and not any(i in st.edited for i in node.ids):
            ids = node.meta.get("np_ids", node.ids) if node.kind == "AMOD" else node.ids  # whole NP for adj+noun
            if any(i in st.edited or i in st.moved for i in ids):
                return False
            before = st.render_ids(ids)
            new = (self._ask(node.prompt, before, st) or "").strip()
            if new and new != before.strip():
                st.replace_span(ids, new)
                return True
            return False
        return self.fallback.apply(graph, node, st) if self.fallback else False


import re as _re

LEXICAL_KINDS = ("LEX", "NOM", "AMOD")
SPLIT_KINDS = ("RELCL", "CONJ_CL", "ADVCL")
_CONTENT = lambda t: {w for w in _re.findall(r"[a-z]+", t.lower()) if len(w) > 3}


class HybridRealizer(Realizer):
    """P1.1 / C3 - split responsibilities explicitly.
      * RULES do lexical replacement (deterministic, auditable) for every LEX/NOM/AMOD node that has a candidate, and
        clause splitting wherever the rule splitter applies.
      * The LLM is a CONSTRAINED fallback only where the rules cannot act: (a) a lexical swap (<= max_span_words words,
        <= max_growth extra words, no new digits, no sentence punctuation) for CPLX / uncovered terms, (b) a clause split
        that must keep >= min_keep of the original content words (re-use words; no paraphrase).
    Unlike `CallableRealizer` it never rewrites whole clauses freely, which is what drove deletions to 72% and SARI below copy.
    `last_via` ('rules'|'llm') and `last_info` are read by the env for the audit trail."""

    def __init__(self, llm, rules=None, lexical_llm=True, structural_llm=True, max_span_words=4, max_growth=3, min_keep=0.7):
        self.llm, self.rules = llm, rules or RuleRealizer()
        self.lexical_llm, self.structural_llm = lexical_llm, structural_llm
        self.max_span_words, self.max_growth, self.min_keep = max_span_words, max_growth, min_keep
        self.last_via = self.last_info = None
        try:
            n = len(inspect.signature(llm).parameters)
        except (TypeError, ValueError):
            n = 2
        self._argc = n

    @staticmethod
    def handles(kind):                       # which node kinds MAY be served by the LLM (for UI tags)
        return kind in ("CPLX", "LEX", "NOM", "AMOD") + SPLIT_KINDS

    def candidates(self, node):
        key = node.meta.get("key")
        if node.kind in ("LEX", "NOM") and key in PLAIN_TERMS:
            return [(PLAIN_TERMS[key][0], None)]
        if node.kind == "AMOD" and key in ADJ_REL:
            return [(ADJ_REL[key][0], None)]
        return []

    def _ask(self, prompt, text, st, mode):
        if self._argc >= 4:
            return self.llm(prompt, text, st.render(), mode)
        return self.llm(prompt, text, st.render()) if self._argc == 3 else self.llm(prompt, text)

    def apply(self, graph, node, st):
        self.last_via = self.last_info = None
        k = node.kind
        if k in LEXICAL_KINDS + ("CPLX",) and any(i in st.edited for i in node.ids):
            return False
        if (k in LEXICAL_KINDS and self.candidates(node)) or k in SPLIT_KINDS:       # the router: rules first
            if self.rules.apply(graph, node, st):
                self.last_via, self.last_info = "rules", self.rules.last_info
                return True
        if k in LEXICAL_KINDS + ("CPLX",) and self.lexical_llm:
            return self._llm_lexical(graph, node, st)
        if k in SPLIT_KINDS and self.structural_llm:
            return self._llm_split(graph, node, st)
        return False

    def _llm_lexical(self, g, node, st):
        ids = node.meta.get("np_ids", node.ids) if node.kind == "AMOD" else node.ids
        if any(i in st.edited or i in st.moved for i in ids):
            return False
        before = st.render_ids(ids).strip()
        if not before or len(before.split()) > self.max_span_words:
            return False
        new = (self._ask(node.prompt, before, st, "lexical") or "").strip()
        from .sanitize import accept_span_rewrite
        ok, _why = accept_span_rewrite(before, new)
        if not ok or len(new.split()) > len(before.split()) + self.max_growth:
            return False
        st.replace_span(ids, new)
        self.last_via = "llm"
        self.last_info = dict(candidate_used=new, candidate_rank=None, candidate_count=None, n_candidates=0, tier=None)
        return True

    def _llm_split(self, g, node, st):
        if node.relation == "condition":          # a condition must stay attached to what it conditions
            return False
        ids = [i for i in node.ids if i not in st.moved and i not in st.edited]
        while ids and g.doc[ids[-1]].is_punct:
            ids.pop()
        if len(ids) != len([i for i in node.ids if not g.doc[i].is_punct]) or not 6 <= len(ids) <= 40:
            return False
        before = st.render_ids(ids).strip()
        new = (self._ask(node.prompt, before, st, "clause") or "").strip()
        a, b = _CONTENT(before), _CONTENT(new)
        ok = (new and a and len(a & b) / len(a) >= self.min_keep and _re.search(r"[.!?]\s+\S", new)
              and not (set(_re.findall(r"\d+", new)) - set(_re.findall(r"\d+", before))))
        if not ok:
            return False
        st.replace_span(ids, new.rstrip(". "))
        self.last_via = "llm"
        self.last_info = dict(candidate_used=None, candidate_rank=None, candidate_count=None, n_candidates=0, tier=None)
        return True


class CopyLexiconRealizer(Realizer):
    """Task 4 - WEAK realizer: applies only the top-1 lexicon substitution for LEX / NOM nodes, nothing else (no clause splits,
    no AMOD, no LLM). Safe by construction apart from lexicon content; used as the weak arm of the 2x2 design."""
    last_via, last_info = None, None

    def apply(self, graph, node, st):
        self.last_via = self.last_info = None
        key = node.meta.get("key")
        if node.kind not in ("LEX", "NOM") or key not in PLAIN_TERMS or any(i in st.edited for i in node.ids):
            return False
        cands = lexicon_mod.CANDIDATES.get(key) or [(PLAIN_TERMS[key][0], None)]
        st.replace_span(node.ids, cands[0][0])
        self.last_via = "rules"
        self.last_info = dict(candidate_used=cands[0][0], candidate_rank=1, candidate_count=cands[0][1], n_candidates=len(cands), tier=2 if key in lexicon_mod.TIER2 else 1)
        return True
