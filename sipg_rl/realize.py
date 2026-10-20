"""Executes an instantiated prompt on a node. `RuleRealizer` is a deterministic stand-in for the small
instruction model so the whole loop runs offline; `CallableRealizer` plugs in any LLM/T5."""
from __future__ import annotations
from .lexicon import PLAIN_TERMS, ADJ_REL, PERSON_NOUNS
from .editstate import EditState, Extra

SPLIT_ADVCL = {"although": "However, ", "though": "However, ", "while": "However, ",
               "whereas": "However, ", "because": "This is because "}
SPLIT_CC = {"and": "", "but": "However, ", "so": "So, "}


class Realizer:
    def apply(self, graph, node, st: EditState) -> bool:  # True iff the text changed
        raise NotImplementedError


class RuleRealizer(Realizer):
    def apply(self, graph, node, st):
        fn = getattr(self, "_" + node.kind.lower(), None)
        return bool(fn and fn(graph, node, st))

    # -- term-level ---------------------------------------------------------------------
    def _lex(self, g, node, st):
        key = node.meta["key"]
        if key not in PLAIN_TERMS or any(i in st.edited for i in node.ids):
            return False
        st.replace_span(node.ids, PLAIN_TERMS[key][0])
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

LLM_KINDS = ('LEX', 'NOM', 'CPLX', 'AMOD', 'CCOMP', 'ACL')


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
