"""Minimal inference server for the fine-tuned model.

Not vLLM (vLLM pins its own torch version tightly, which would very
likely re-trigger the exact same torch-upgrade-breaks-torchvision problem
we spent several rounds fixing for transformers/peft/trl - not worth the
risk for this deliverable).

The HTTP layer here never calls model.generate() directly - it shells
out to generate_once.py as a subprocess per request. See that file's
docstring for why: calling generate() in-process while an HTTP request's
socket is open hung indefinitely (0% GPU utilization) under every
framework tried (threaded FastAPI, async FastAPI, plain stdlib
http.server), while the exact same call with no open socket in the
process worked every time. The subprocess boundary costs ~15s of model
reload per request but guarantees correctness.
"""
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8000
GENERATE_SCRIPT = "/mnt/data/assignment/serve/generate_once.py"


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        print(f"Dispatching to subprocess for: {body[:80]!r}...", flush=True)

        result = subprocess.run(
            [sys.executable, GENERATE_SCRIPT],
            input=body,
            capture_output=True,
            timeout=240,
        )
        if result.returncode != 0:
            print("Subprocess failed:", result.stderr.decode()[-2000:], flush=True)
            self._send_json(500, {"error": result.stderr.decode()[-2000:]})
            return

        print("Subprocess completed.", flush=True)
        response = json.loads(result.stdout.decode())
        self._send_json(200, response)

    def log_message(self, fmt, *args):
        print("[http] " + (fmt % args), flush=True)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving on 0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
