"""Local dashboard server (stdlib only). Runs the real SIPG-RL pipeline.

    python dashboard/server.py [--port 8000] [--ckpt sipg_policy_bonus.pt]
then open http://localhost:8000
"""
import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from sipg_rl.pipeline import SIPGRL, default_parts  # noqa: E402
from sipg_rl.llm import OpenAICompatLLM  # noqa: E402
from sipg_rl.realize import CallableRealizer, RuleRealizer  # noqa: E402
from sipg_rl import glossary  # noqa: E402

LOCK = threading.Lock()
STATE = {"models": {}, "ckpt": None, "llm": None}
MAX_CHARS = 6000


def jdefault(o):
    return o.item() if hasattr(o, "item") else str(o)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html")
        elif self.path == "/api/health":
            llm = STATE["llm"].health() if STATE["llm"] else {"configured": False}
            self._send(200, json.dumps({"ok": True, "checkpoint": STATE["ckpt"], "trained": bool(STATE["ckpt"]), "llm": llm}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _explain(self):
        """Info-box text for one term: built-in glossary first, otherwise (on request) the local LLM."""
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            term, ctx = (body.get("term") or "").strip(), (body.get("context") or "").strip()[:600]
            if not term or len(term) > 60:
                return self._send(400, json.dumps({"error": "give one term (max 60 characters)"}))
            e = glossary.lookup(term)
            if e and e.definition:
                return self._send(200, json.dumps(e.public([term])))
            llm = STATE["llm"]
            if not llm:
                return self._send(503, json.dumps({"error": "No LLM is configured, and this term is not in the built-in glossary."}))
            h = llm.health()
            if not h["reachable"] or h["model_available"] is False:
                return self._send(503, json.dumps({"error": f"LLM not available at {h['base_url']}. Start Ollama and pull {h['model']}."}))
            d = llm.explain(term, ctx)
            if not d:
                return self._send(200, json.dumps({"key": term.lower(), "title": term.capitalize(), "type": "other",
                                                   "type_label": glossary.TYPE_LABELS["other"], "definition": "", "extra": "",
                                                   "source": "llm", "model": llm.model, "surfaces": [term]}))
            self._send(200, json.dumps({"key": term.lower(), "title": term.capitalize(), "type": d["type"],
                                        "type_label": glossary.TYPE_LABELS[d["type"]], "definition": d["definition"],
                                        "extra": d["extra"], "source": "llm", "model": llm.model, "surfaces": [term]}))
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def do_POST(self):
        if self.path == "/api/explain":
            return self._explain()
        if self.path != "/api/simplify":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            text, mode = (body.get("text") or "").strip(), body.get("rewriter", "rules")
            apply_all = bool(body.get("apply_all", False))
            if mode not in STATE["models"]:
                return self._send(400, json.dumps({"error": f"rewriter '{mode}' is not available on this server"}))
            if not text:
                return self._send(400, json.dumps({"error": "empty input"}))
            if len(text) > MAX_CHARS:
                return self._send(400, json.dumps({"error": f"input too long (max {MAX_CHARS} characters)"}))
            if mode == "llm":
                h = STATE["llm"].health()
                if not h["reachable"]:
                    return self._send(503, json.dumps({"error": f"LLM server not reachable at {h['base_url']} ({h['error']}). "
                                                       "Start it (for Ollama: `ollama serve`) or switch to Rules."}))
                if h["model_available"] is False:
                    return self._send(503, json.dumps({"error": f"model '{h['model']}' not found on the LLM server. "
                                                       f"For Ollama run: ollama pull {h['model']}"}))
            t0 = time.time()
            with LOCK:
                if mode == "llm":
                    STATE["llm"].reset_stats()
                res = STATE["models"][mode].simplify(text, apply_all=apply_all)
                stats = dict(STATE["llm"].stats, model=STATE["llm"].model) if mode == "llm" else None
            out = {"input": text, "output": res.text, "sentences": res.sentences, "rewriter": mode, "apply_all": apply_all,
                   "llm_stats": stats, "terms": res.terms, "seconds": round(time.time() - t0, 3)}
            self._send(200, json.dumps(out, default=jdefault))
        except Exception as e:  # surface errors in the UI instead of a dead request
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def log_message(self, fmt, *a):
        sys.stderr.write("[dashboard] " + fmt % a + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-llm", action="store_true", help="disable the LLM rewriter option")
    ap.add_argument("--llm-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                    help="OpenAI-compatible base URL (default: local Ollama)")
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "qwen2.5:7b-instruct"))
    ap.add_argument("--llm-key", default=os.environ.get("LLM_API_KEY"), help="only for hosted APIs")
    a = ap.parse_args()
    cands = [a.ckpt] if a.ckpt else [os.path.join(ROOT, f) for f in ("sipg_policy_bonus.pt", "sipg_policy.pt")]
    ckpt = next((c for c in cands if c and os.path.exists(c)), None)
    print("loading spaCy + policy ...", flush=True)
    rules = SIPGRL(checkpoint=ckpt)
    STATE["models"]["rules"], STATE["ckpt"] = rules, (os.path.basename(ckpt) if ckpt else None)
    if not a.no_llm:
        STATE["llm"] = OpenAICompatLLM(a.llm_url, a.llm_model, a.llm_key)
        parts = default_parts(nlp=rules.parts["nlp"], realizer=CallableRealizer(STATE["llm"], RuleRealizer()))
        # same trained policy; sweep because it was trained on the rule rewriter (see README)
        STATE["models"]["llm"] = SIPGRL(parts, policy=rules.policy, sweep=True)
        h = STATE["llm"].health()
        print(f"LLM: {a.llm_model} @ {a.llm_url} -> " + ("reachable" if h["reachable"] else f"NOT reachable ({h['error']})"), flush=True)
    if not ckpt:
        print("WARNING: no checkpoint found - policy is untrained (run scripts/pilot.py first).")
    print(f"ready: http://{a.host}:{a.port}  (checkpoint: {STATE['ckpt']})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
