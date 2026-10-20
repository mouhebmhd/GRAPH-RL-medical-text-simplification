"""Mutable token-level view of a sentence. Edits are attached to stable token ids so that
node prompts invoked in any order compose correctly, and every step can be rolled back."""
from __future__ import annotations
import copy
import re
from dataclasses import dataclass
from .utils import clean_spacing, capitalize_first


@dataclass
class Extra:
    """A sentence split off the main flow. Rendered lazily so later edits still apply."""
    prefix: str
    np_ids: list      # noun-phrase ids copied into the new sentence (rendered without insertions)
    clause_ids: list  # moved clause tokens (live)
    the_first: bool = False  # turn a leading a/an into 'The' (restrictive relative clauses)


class EditState:
    def __init__(self, toks):
        # toks: list of (id, text, whitespace)
        self.ids = [t[0] for t in toks]
        self.base = {t[0]: (t[1], t[2]) for t in toks}
        self.repl, self.deleted, self.moved, self.edited = {}, set(), set(), set()
        self.after = {}
        self.extras = []

    # ---- snapshot / restore -------------------------------------------------------------
    def snapshot(self):
        return (dict(self.repl), set(self.deleted), set(self.moved), set(self.edited),
                {k: list(v) for k, v in self.after.items()}, copy.deepcopy(self.extras))

    def restore(self, snap):
        self.repl, self.deleted, self.moved, self.edited, self.after, self.extras = (
            dict(snap[0]), set(snap[1]), set(snap[2]), set(snap[3]),
            {k: list(v) for k, v in snap[4].items()}, copy.deepcopy(snap[5]))

    # ---- primitive edits ----------------------------------------------------------------
    def replace_span(self, ids, new_text):
        """Replace tokens `ids` by `new_text` (first token takes the text, others deleted)."""
        ids = sorted(ids)
        first = self.base[ids[0]][0]
        if first[:1].isupper() and not (first.isupper() and len(first) > 1) and new_text:
            new_text = new_text[0].upper() + new_text[1:]
        self.repl[ids[0]] = new_text
        for i in ids[1:]:
            self.deleted.add(i)
        self.edited.update(ids)

    def delete(self, i):
        self.deleted.add(i)
        self.edited.add(i)

    def insert_after(self, i, text):
        self.after.setdefault(i, []).append(text)
        self.edited.add(i)

    # ---- rendering ----------------------------------------------------------------------
    def render_ids(self, ids, include_after=True, skip_moved=False):
        pieces, prev_txt, prev_edited = [], None, False
        for i in sorted(ids):
            if i in self.deleted or (skip_moved and i in self.moved):
                if self.base[i][1] and pieces and not pieces[-1][1]:   # hand the lost space to the previous piece
                    pieces[-1] = (pieces[-1][0], self.base[i][1])
                continue
            txt, ws = self.base[i]
            txt = self.repl.get(i, txt)
            ed = i in self.edited
            if ed and prev_txt and prev_txt.lower() in ("a", "an") and not prev_edited and txt:
                art = "an" if txt[0].lower() in "aeio" else "a"
                art = art.capitalize() if prev_txt[0].isupper() else art
                pieces[-1] = (art, pieces[-1][1])
            ins = self.after.get(i, []) if include_after else []
            if ins:
                pieces.append((txt, " "))
                for k, s in enumerate(ins):
                    pieces.append((s, ws if k == len(ins) - 1 else " "))
            else:
                pieces.append((txt, ws))
            prev_txt, prev_edited = txt, ed
        return "".join(p + w for p, w in pieces)

    @staticmethod
    def _finish(s):
        s = clean_spacing(s)
        s = s.rstrip(" ,;:")
        if s and s[-1] not in ".!?":
            s += "."
        return capitalize_first(s)

    def render(self):
        main = self._finish(self.render_ids(self.ids, skip_moved=True))
        out = [main] if main not in ("", ".") else []
        for ex in self.extras:
            body = self.render_ids(ex.np_ids, include_after=False, skip_moved=False)
            if ex.the_first:
                body = re.sub(r"^(?:a|an)\b", "the", body, flags=re.I)
            body = (body + " " if ex.np_ids else "") + self.render_ids(ex.clause_ids)
            body = clean_spacing(body).rstrip(" .,;:")
            if not body:
                continue
            out.append(self._finish(ex.prefix + body if ex.prefix else body))
        return " ".join(out)
