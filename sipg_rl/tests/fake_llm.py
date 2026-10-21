"""TEST-ONLY stand-in for an OpenAI-compatible server (Ollama etc.). Deterministic canned answers so we can test
the client, sanitising, caching, the sweep and the verifier gate WITHOUT a real model. Not a model."""
import json, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWERS = {"utilize": "use", "hypertensive": "with high blood pressure", "orally": "by mouth with 50 mg extra",  # unsafe on purpose
           "dyspnea": 'Replacement span: "trouble breathing."', "contraindicated": "not safe to take, because of risks to the kidneys and liver.",
           "bilateral": "on both sides"}
EXPLAIN = {"gastritis": 'Sure! {"type": "condition", "definition": "Swelling of the stomach lining.", "extra": "Often causes burning pain."}'}
CALLS = []


class H(BaseHTTPRequestHandler):
    def _j(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._j({"data": [{"id": "fake-model:1b"}]} if self.path.endswith("/models") else {}, 200 if self.path.endswith("/models") else 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        CALLS.append(body)
        user = body["messages"][-1]["content"]
        if user.startswith("Term:"):                      # glossary "explain" request
            term = user.splitlines()[0][5:].strip().lower()
            canned = EXPLAIN.get(term, "I do not know.")  # non-JSON answer = model unsure
            return self._j({"choices": [{"message": {"role": "assistant", "content": canned}}]})
        m = re.search(r"Span to rewrite: (.*)", user)
        span = (m.group(1).strip() if m else "")
        self._j({"choices": [{"message": {"role": "assistant", "content": ANSWERS.get(span.lower(), span)}}]})

    def log_message(self, *a):
        pass


def serve(port=0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    return srv


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    print("fake LLM on", port, flush=True)
    serve(port).serve_forever()
