"""Invoke generate_once.py as a subprocess, same as serve_model.py does,
but with no HTTP server anywhere in the picture - isolates whether the
subprocess mechanism itself is slow, independent of sockets."""
import json
import subprocess
import sys
import time

GENERATE_SCRIPT = "/mnt/data/assignment/serve/generate_once.py"
body = json.dumps({"prompt": "Write a Python function is_leap_year(year).", "max_new_tokens": 100}).encode()

print("Launching subprocess...", flush=True)
start = time.monotonic()
try:
    result = subprocess.run(
        [sys.executable, GENERATE_SCRIPT],
        input=body,
        capture_output=True,
        timeout=120,
    )
    print(f"Completed in {time.monotonic() - start:.1f}s", flush=True)
    print("STDOUT:", result.stdout.decode()[:500], flush=True)
    print("STDERR tail:", result.stderr.decode()[-1000:], flush=True)
except subprocess.TimeoutExpired as e:
    print(f"TIMED OUT after {time.monotonic() - start:.1f}s", flush=True)
    print("Partial STDOUT:", (e.stdout or b"").decode()[:1000], flush=True)
    print("Partial STDERR:", (e.stderr or b"").decode()[-2000:], flush=True)
