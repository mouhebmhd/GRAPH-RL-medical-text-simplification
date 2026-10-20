"""Component 1: syntax-induced graph. Nodes = clause subtrees + candidate terms; edges = dependency
structure + shallow discourse relations (cue-based). No external KB/AMR is used."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np
from .lexicon import PLAIN_TERMS, ADJ_REL, CRITICAL_TERMS, NEGATION_CUES
from .utils import syllables, hash_vec
from .context import source_context

KINDS = ["ROOT", "RELCL", "CONJ_CL", "ADVCL", "CCOMP", "ACL", "LEX", "NOM", "AMOD", "CPLX"]
CLAUSE_KINDS = {"ROOT", "RELCL", "CONJ_CL", "ADVCL", "CCOMP", "ACL"}
EDGE_TYPES = ["syn_down", "syn_up", "contrast", "elaboration", "condition", "cause", "sequence",
              "next", "prev", "self"]
ET = {n: i for i, n in enumerate(EDGE_TYPES)}
N_MAX = 24
PROMPT_DIM = 16
STATIC_DIM = len(KINDS) + 6 + PROMPT_DIM

ADVCL_REL = {"although": "contrast", "though": "contrast", "while": "contrast", "whereas": "contrast",
             "but": "contrast", "if": "condition", "unless": "condition", "when": "condition",
             "once": "condition", "because": "cause", "since": "cause", "as": "cause",
             "after": "sequence", "before": "sequence", "until": "sequence"}
CC_REL = {"but": "contrast", "and": "sequence", "or": "contrast", "so": "cause", "yet": "contrast"}
_NUM = re.compile(r"\d")


@dataclass
class Node:
    id: int
    kind: str
    anchor: int
    ids: list
    parent: int = -1
    depth: int = 0
    relation: str = ""
    critical: bool = False
    text: str = ""
    meta: dict = field(default_factory=dict)
    prompt: str = ""


@dataclass
class SentenceGraph:
    doc: object
    text: str
    toks: list           # (id, text, ws) for every token of the sentence
    nodes: list
    adj: np.ndarray      # [n, n] int edge type or -1
    static: np.ndarray   # [N_MAX, STATIC_DIM]
    src_vec: object = None

    @property
    def n(self):
        return len(self.nodes)


def is_critical_token(t) -> bool:
    lo = t.lower_
    return bool(_NUM.search(lo)) or lo in CRITICAL_TERMS or lo in NEGATION_CUES or lo.endswith("n't")


class GraphBuilder:
    def __init__(self, nlp, instantiator, max_cplx: int = 3):
        bad = {"merge_entities", "merge_noun_chunks", "merge_subtokens"} & set(nlp.pipe_names)
        if bad:  # merging collapses multi-word terms into atomic tokens and destroys internal syntax
            raise ValueError(f"spaCy pipes {sorted(bad)} would collapse clinical terms before graph construction")
        self.nlp, self.inst, self.max_cplx = nlp, instantiator, max_cplx

    # ------------------------------------------------------------------------------------
    def sentences(self, text):
        return [s for s in self.nlp(text).sents if s.text.strip()]

    def build_text(self, sentence_text: str) -> SentenceGraph:
        doc = self.nlp(sentence_text.strip())
        return self.build(doc[:])

    def build(self, sent) -> SentenceGraph:
        doc = sent.doc
        toks = [(t.i, t.text, t.whitespace_) for t in sent]
        nodes = self._clauses(sent) + self._terms(sent)
        nodes = nodes[:N_MAX]
        for k, n in enumerate(nodes):
            n.id = k
        owner = self._assign_parents(nodes, doc)
        for n in nodes:
            n.text = " ".join(doc[i].text for i in n.ids)
            n.critical = any(is_critical_token(doc[i]) for i in n.ids if i not in
                             self._child_clause_tokens(n, nodes))
            n.meta["source_ctx"] = source_context(n, doc, doc[n.ids[0]].sent.start)
            self._annotate(n, doc)
            n.prompt = self.inst.instantiate(n, doc)
        adj = self._edges(nodes, doc)
        g = SentenceGraph(doc, sent.text, toks, nodes, adj, self._static(nodes, doc))
        return g

    # ------------------------------------------------------------------------------------
    def _clauses(self, sent):
        out = [Node(0, "ROOT", sent.root.i, [t.i for t in sent.root.subtree if sent.start <= t.i < sent.end])]
        for t in sent:
            kind = None
            has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in t.children)
            if t.dep_ == "relcl":
                kind = "RELCL"
            elif t.dep_ == "conj" and t.pos_ in ("VERB", "AUX") and t.head.pos_ in ("VERB", "AUX") and has_subj:
                kind = "CONJ_CL"
            elif t.dep_ == "advcl" and has_subj:
                kind = "ADVCL"
            elif t.dep_ == "ccomp":
                kind = "CCOMP"
            elif t.dep_ == "acl":
                kind = "ACL"
            if kind:
                out.append(Node(0, kind, t.i, [x.i for x in t.subtree if sent.start <= x.i < sent.end]))
        return out

    def _terms(self, sent):
        doc, out, covered = sent.doc, [], set()
        words = [t.lower_ for t in sent]
        lemmas = [t.lemma_.lower() for t in sent]
        start = sent.start
        for i in range(len(words)):
            for L in (3, 2, 1):
                if i + L > len(words):
                    continue
                key = " ".join(words[i:i + L])
                if key not in PLAIN_TERMS and L > 1:   # tolerate plurals: lemma of the head (last) token
                    key = " ".join(words[i:i + L - 1] + [lemmas[i + L - 1]])
                if key in PLAIN_TERMS and not any(start + j in covered for j in range(i, i + L)):
                    ids = [start + j for j in range(i, i + L)]
                    kind = "NOM" if PLAIN_TERMS[key][1] == "nom" else "LEX"
                    out.append(Node(0, kind, ids[0], ids, meta={"key": key}))
                    covered.update(ids)
                    break
        for t in sent:
            if t.lower_ in ADJ_REL and t.dep_ == "amod" and t.i not in covered:
                out.append(Node(0, "AMOD", t.i, [t.i], meta={"key": t.lower_}))
                covered.add(t.i)
        n_cplx = 0
        for t in sent:
            if (n_cplx < self.max_cplx and t.i not in covered and t.is_alpha and not t.is_stop
                    and syllables(t.text) >= 4 and t.lower_ not in CRITICAL_TERMS
                    and t.pos_ in ("NOUN", "VERB", "ADJ", "ADV")):
                out.append(Node(0, "CPLX", t.i, [t.i], meta={"key": t.lower_}))
                n_cplx += 1
        return out

    @staticmethod
    def _annotate(n, doc):
        """Entity spans are ATTRIBUTES of nodes, never node merges. Internal syntax stays in the graph's
        token layer: meta['structure'] keeps (token, dep, head) for every token of a multi-token term."""
        if n.kind in CLAUSE_KINDS:
            return
        n.meta["structure"] = [(doc[i].text, doc[i].dep_, doc[i].head.text) for i in n.ids]
        n.meta["entities"] = sorted({(e.text, e.label_) for e in doc.ents
                                     if any(e.start <= i < e.end for i in n.ids)})
        if n.kind == "AMOD":
            adj = doc[n.anchor]
            n.meta["np_ids"] = list(range(adj.i, adj.head.i + 1)) if adj.i < adj.head.i else [adj.i]
        if n.kind == "LEX":
            inside = {doc[i].text for i in n.ids}
            n.meta["modifier"] = next(((w, h) for w, d, h in n.meta["structure"]
                                       if d in ("amod", "compound") and h in inside and h != w), None)

    @staticmethod
    def _child_clause_tokens(n, nodes):
        if n.kind not in CLAUSE_KINDS:
            return set()
        s = set()
        for m in nodes:
            if m.kind in CLAUSE_KINDS and m.id != n.id and m.anchor in n.ids and m.anchor != n.anchor:
                s.update(m.ids)
        return s

    def _assign_parents(self, nodes, doc):
        clause_nodes = [n for n in nodes if n.kind in CLAUSE_KINDS]
        by_anchor = {n.anchor: n for n in clause_nodes}

        def owner_of(tok_i, exclude=None):
            t = doc[tok_i]
            while True:
                n = by_anchor.get(t.i)
                if n is not None and n is not exclude:
                    return n
                if t.head.i == t.i:
                    return clause_nodes[0]
                t = t.head
        for n in nodes:
            if n.kind == "ROOT":
                n.parent, n.depth = -1, 0
            elif n.kind in CLAUSE_KINDS:
                n.parent = owner_of(doc[n.anchor].head.i, exclude=n).id
            else:
                n.parent = owner_of(n.anchor).id
            n.relation = self._relation(n, doc)
        for n in sorted(nodes, key=lambda m: m.anchor):  # depth (parents may be later in order)
            d, p = 0, n
            while p.parent != -1 and d < 10:
                p, d = nodes[p.parent], d + 1
            n.depth = d
        return by_anchor

    @staticmethod
    def _relation(n, doc):
        t = doc[n.anchor]
        if n.kind in ("RELCL", "CCOMP", "ACL"):
            return "elaboration"
        if n.kind == "ADVCL":
            for c in t.children:
                if c.dep_ == "mark" and c.lower_ in ADVCL_REL:
                    return ADVCL_REL[c.lower_]
            return "elaboration"
        if n.kind == "CONJ_CL":
            ccs = [c for c in t.head.children if c.dep_ == "cc" and c.i < t.i]
            if ccs:
                return CC_REL.get(ccs[-1].lower_, "sequence")
            return "sequence"
        return ""

    def _edges(self, nodes, doc):
        n = len(nodes)
        adj = -np.ones((n, n), dtype=np.int64)
        for a in nodes:
            adj[a.id, a.id] = ET["self"]
        order = sorted(nodes, key=lambda m: (m.anchor, m.id))
        for x, y in zip(order, order[1:]):
            adj[x.id, y.id] = ET["next"]
            adj[y.id, x.id] = ET["prev"]
        for a in nodes:
            if a.parent >= 0:
                adj[a.parent, a.id] = ET["syn_down"]
                adj[a.id, a.parent] = ET["syn_up"]
        for a in nodes:  # discourse overrides structural type between clause pairs
            if a.parent >= 0 and a.relation:
                adj[a.parent, a.id] = adj[a.id, a.parent] = ET[a.relation]
        return adj

    def _static(self, nodes, doc):
        X = np.zeros((N_MAX, STATIC_DIM), dtype=np.float32)
        for n in nodes:
            f = X[n.id]
            f[KINDS.index(n.kind)] = 1.0
            words = [doc[i].text for i in n.ids if doc[i].is_alpha]
            o = len(KINDS)
            f[o] = n.depth / 4.0
            f[o + 1] = min(len(n.ids), 30) / 30.0
            f[o + 2] = float(n.critical)
            f[o + 3] = (np.mean([len(w) for w in words]) / 10.0) if words else 0.0
            f[o + 4] = (max(syllables(w) for w in words) / 6.0) if words else 0.0
            f[o + 5] = float(n.meta.get("key", "") in PLAIN_TERMS or n.meta.get("key", "") in ADJ_REL)
            f[o + 6:] = hash_vec(n.prompt.lower().split(), PROMPT_DIM, signed=True) / 8.0
        return X
