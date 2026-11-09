"""Tests for share_gate.py against a tiny fake OpenAI server in a thread.

Run from ``tools/``: ``python3 -m unittest tests.test_share_gate`` (or
``python3 -m pytest tests/test_share_gate.py -q``).
"""
import hashlib
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import share_gate


class FakeServer:
    """An http.server answering deterministic content (a hash of the prompt) with streamed ``data:`` lines."""

    def __init__(self, *, break_warm=False, break_concurrent=False, draft_differs=False):
        self.break_warm = break_warm
        self.warm_seen = set()          # prompts already answered once; the next is a warm (resumed) request
        self.break_concurrent = break_concurrent
        self.drafted_seen = False       # the drafted run ends the size phase, so later plain runs are concurrent
        self.draft_differs = draft_differs
        self.requests = []

    def handle(self, handler):
        if handler.path in ("/metrics", "/health"):
            body = b"kv_pool_free 1024\nprompt_cache_entries 0\n"
            handler.send_response(200)
            handler.send_header("Content-Type", "text/plain")
            handler.end_headers()
            handler.wfile.write(body if handler.path == "/metrics" else b"ok")
            return
        if handler.path != "/v1/chat/completions":
            handler.send_response(404)
            handler.end_headers()
            return
        length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(length))
        self.requests.append(body)
        if body.get("draft") is False:
            self.drafted_seen = True
        prompt = body["messages"][0]["content"]
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:32]
        if self.break_warm and body["messages"][0]["content"] in self.warm_seen:
            digest = "0" * 32
        self.warm_seen.add(body["messages"][0]["content"])
        if self.draft_differs and body.get("draft") is False:
            digest = "1" * 32
        if self.break_concurrent and self.drafted_seen and body.get("draft") is not False:
            digest = "2" * 32
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        for piece in (digest[:8], digest[8:16], digest[16:24], prompt[:8] + " end"):
            chunk = {"id": "c", "object": "chat.completion.chunk", "choices": [{"delta": {"content": piece}}]}
            handler.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        usage = {"prompt_tokens": 12, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}}
        final = {"id": "c", "object": "chat.completion.chunk", "choices": [], "usage": usage}
        handler.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()


def serve(fake):
    handler = type("H", (BaseHTTPRequestHandler,),
                   {"do_GET": lambda self: fake.handle(self), "do_POST": lambda self: fake.handle(self),
                    "log_message": lambda *a: None})
    # daemon_threads: the gate's urllib connections can outlive shutdown; without it the sockets warn
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


class ShareGateTest(unittest.TestCase):
    def run_gate(self, fake, argv):
        server, base = serve(fake)
        try:
            return share_gate.main([base, *argv])
        finally:
            server.shutdown()

    def test_equal_hashes_pass(self):
        rc = self.run_gate(FakeServer(), ["--sizes", "128", "--levels", "1", "--out", "/tmp/opencode/g1.json"])
        self.assertEqual(rc, 0)
        with open("/tmp/opencode/g1.json") as f:
            report = json.load(f)
        self.assertTrue(report["pass"])
        self.assertEqual(report["failures"], [])
        # resumed equals fresh: warm hashes the same as cold; the appended case is reported, not compared
        self.assertTrue(all(case["alone"]["warm"]["sha256"] == case["alone"]["cold"]["sha256"]
                            for case in report["cases"]))

    def test_differing_warm_hash_fails(self):
        rc = self.run_gate(FakeServer(break_warm=True), ["--sizes", "128", "--levels", "1",
                                                         "--out", "/tmp/opencode/g2.json"])
        self.assertEqual(rc, 1)
        with open("/tmp/opencode/g2.json") as f:
            report = json.load(f)
        self.assertFalse(report["pass"])
        self.assertIn("warm_hash_differs", report["failures"][0])

    def test_concurrent_comparison(self):
        rc = self.run_gate(FakeServer(), ["--sizes", "128", "--levels", "2", "--out", "/tmp/opencode/g3.json"])
        self.assertEqual(rc, 0)
        with open("/tmp/opencode/g3.json") as f:
            report = json.load(f)
        self.assertEqual(report["concurrency"][0]["level"], 2)
        self.assertTrue(all(r["equal_alone"] for cell in report["concurrency"] for r in cell["runs"]))
        # a concurrent reply that differs from the same prompt alone fails the run
        rc = self.run_gate(FakeServer(break_concurrent=True), ["--sizes", "128", "--levels", "2"])
        self.assertEqual(rc, 1)

    def test_report_schema(self):
        self.run_gate(FakeServer(), ["--sizes", "128", "--levels", "1", "--out", "/tmp/opencode/g4.json"])
        with open("/tmp/opencode/g4.json") as f:
            report = json.load(f)
        for key in ("schema_version", "base", "model", "sizes", "levels", "cases", "concurrency",
                    "census_before", "census_after", "census_delta", "pass", "failures"):
            self.assertIn(key, report)
        case = report["cases"][0]
        for key in ("requested_tokens", "alone", "drafted", "failures"):
            self.assertIn(key, case)
        for phase in ("cold", "warm", "appended"):
            for key in ("sha256", "ttft_s", "total_s", "usage", "cached_tokens", "prompt_tokens",
                        "output_sha256"):
                self.assertIn(key, case["alone"][phase])
        self.assertTrue(report["census_before"]["metrics"])
        self.assertIn("kv_pool_", " ".join(report["census_before"]["metrics"]))

    def test_dry_run(self):
        import io
        from contextlib import redirect_stdout
        out = io.StringIO()
        with redirect_stdout(out):
            rc = share_gate.main(["http://none:1", "--sizes", "128", "--levels", "1,2", "--dry-run"])
        self.assertEqual(rc, 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan["cells"][0]["requested_tokens"], 128)
        self.assertEqual(len(plan["cells"][0]["runs"]), 4)  # cold, warm, appended, drafted


if __name__ == "__main__":
    unittest.main()
