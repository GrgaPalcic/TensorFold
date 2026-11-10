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

CONTEXT = 8192  # the fake server's context limit; 600000 and 4096+64 do not fit under it


class FakeServer:
    """An http.server answering deterministic content (a hash of the prompt) with streamed ``data:`` lines.

    usage.prompt_tokens is proportional to the prompt's length (so calibration is exercised), /health
    carries a context limit, warm replies report cached_tokens and the metrics are tensorfold:-prefixed.
    """

    def __init__(self, *, break_warm=False, break_concurrent=False, draft_differs=False,
                 oversize_error=False, metric_prefix="tensorfold:"):
        self.break_warm = break_warm
        self.warm_seen = set()          # prompts already answered once; the next is a warm (resumed) request
        self.break_concurrent = break_concurrent
        self.draft_differs = draft_differs
        self.oversize_error = oversize_error
        self.metric_prefix = metric_prefix
        self.requests = []
        self.in_flight = 0              # requests being answered at once; concurrent copies overlap
        self.lock = threading.Lock()

    def handle(self, handler):
        if handler.path == "/health":
            body = json.dumps({"context": CONTEXT}).encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(body)
            return
        if handler.path == "/metrics":
            prefix = self.metric_prefix
            body = (f"{prefix}kv_cache_usage_perc{{stream=\"0\"}} 0.42\n"
                    f"{prefix}kv_cache_usage_ratio{{pool=\"0\"}} 0.10\n"
                    f"{prefix}kv_shard_part_free{{part=\"0\"}} 512\n"
                    f"{prefix}prompt_cache_entries 3\n"
                    f"unrelated_metric 1\n").encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "text/plain")
            handler.end_headers()
            handler.wfile.write(body)
            return
        if handler.path != "/v1/chat/completions":
            handler.send_response(404)
            handler.end_headers()
            return
        length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(length))
        with self.lock:
            self.requests.append(body)
            self.in_flight += 1
            concurrent_now = self.in_flight > 1  # another request is being answered at the same time
        prompt = body["messages"][0]["content"]
        # usage.prompt_tokens proportional to the prompt's length, as a real tokenizer would count it
        prompt_tokens = max(1, len(prompt.splitlines()) * 4)
        if prompt_tokens + body.get("max_tokens", 64) > CONTEXT and self.oversize_error:
            handler.send_response(400)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "prompt is longer than the context window"}).encode())
            return
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:32]
        seen = prompt in self.warm_seen
        if self.break_warm and seen:
            digest = "0" * 32
        self.warm_seen.add(prompt)
        if self.draft_differs and body.get("draft") is False:
            digest = "1" * 32
        if self.break_concurrent and concurrent_now and body.get("draft") is not False:
            digest = "2" * 32
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        for piece in (digest[:8], digest[8:16], digest[16:24], prompt[:8] + " end"):
            chunk = {"id": "c", "object": "chat.completion.chunk", "choices": [{"delta": {"content": piece}}]}
            handler.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        # a warm reply (the prompt was already answered once) reports the resumed tokens; the first
        # ever request reads 0, like the live lane's first cold request
        cached = 0 if not seen else max(0, prompt_tokens - 4)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 4,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        final = {"id": "c", "object": "chat.completion.chunk", "choices": [], "usage": usage}
        handler.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
        with self.lock:
            self.in_flight -= 1


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
        # calibration ran and the measured tokens are within +-5 % of the asked size
        self.assertIn("tokens_per_line", report["calibration"])
        self.assertTrue(all("measured_tokens" in case for case in report["cases"]))
        self.assertTrue(all("size_off" not in case["failures"] for case in report["cases"]))

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
        # each copy carries its own seed's alone hash: distinct seeds have distinct alone hashes,
        # so the comparison is not against one shared reference
        seen_alone = {run_["alone_sha256"] for cell in report["concurrency"] for run_ in cell["runs"]}
        self.assertGreater(len(seen_alone), 1)
        # a concurrent reply that differs from its own seed's alone reply fails the run
        rc = self.run_gate(FakeServer(break_concurrent=True), ["--sizes", "128", "--levels", "2"])
        self.assertEqual(rc, 1)

    def test_oversize_is_listed_failure(self):
        rc = self.run_gate(FakeServer(oversize_error=True),
                           ["--sizes", "600000", "--levels", "1", "--out", "/tmp/opencode/g5.json"])
        self.assertEqual(rc, 1)
        with open("/tmp/opencode/g5.json") as f:
            report = json.load(f)
        self.assertFalse(report["pass"])
        self.assertEqual(report["context_limit"], CONTEXT)
        # the oversized size is a listed size_over_context failure, not an exception
        self.assertIn("size 600000: size_over_context", report["failures"][0])
        self.assertEqual(report["cases"][0]["failures"], ["size_over_context"])
        self.assertNotIn("error", report["cases"][0])

    def test_census_finds_prefixed_metrics(self):
        self.run_gate(FakeServer(), ["--sizes", "128", "--levels", "1", "--out", "/tmp/opencode/g6.json"])
        with open("/tmp/opencode/g6.json") as f:
            report = json.load(f)
        names = " ".join(report["census_before"]["metrics"])
        # the tensorfold:-prefixed and bare prompt_cache names are matched verbatim, labels included
        self.assertIn("tensorfold:kv_cache_usage_perc{stream=\"0\"}", names)
        self.assertIn("tensorfold:kv_cache_usage_ratio{pool=\"0\"}", names)
        self.assertIn("tensorfold:kv_shard_part_free{part=\"0\"}", names)
        self.assertIn("prompt_cache_entries", names)
        self.assertNotIn("unrelated_metric", names)
        # a server without the prefix is matched the same way
        self.run_gate(FakeServer(metric_prefix=""),
                      ["--sizes", "128", "--levels", "1", "--out", "/tmp/opencode/g6b.json"])
        with open("/tmp/opencode/g6b.json") as f:
            report = json.load(f)
        self.assertIn("kv_cache_usage_perc{stream=\"0\"}", " ".join(report["census_before"]["metrics"]))

    def test_cached_reported(self):
        self.run_gate(FakeServer(), ["--sizes", "128", "--levels", "1", "--out", "/tmp/opencode/g7.json"])
        with open("/tmp/opencode/g7.json") as f:
            report = json.load(f)
        # cold reads 0 (never answered before), warm reports the resumed tokens; both recorded per run
        cold = report["cases"][0]["alone"]["cold"]
        warm = report["cases"][0]["alone"]["warm"]
        self.assertEqual(cold["resumed_from"], 0)
        self.assertGreater(warm["resumed_from"], 0)
        # the table prints the warm run's cached tokens as ``cached`` (a dash when the server omits it)
        self.assertIn("cached", share_gate.summary(report))

    def test_report_schema(self):
        self.run_gate(FakeServer(), ["--sizes", "128", "--levels", "1", "--out", "/tmp/opencode/g4.json"])
        with open("/tmp/opencode/g4.json") as f:
            report = json.load(f)
        for key in ("schema_version", "base", "model", "sizes", "levels", "cases", "concurrency",
                    "calibration", "context_limit", "census_before", "census_after", "census_delta",
                    "pass", "failures"):
            self.assertIn(key, report)
        case = report["cases"][0]
        for key in ("requested_tokens", "alone", "drafted", "failures", "measured_tokens"):
            self.assertIn(key, case)
        for phase in ("cold", "warm", "appended"):
            for key in ("sha256", "ttft_s", "total_s", "usage", "cached_tokens", "resumed_from",
                        "prompt_tokens", "output_sha256"):
                self.assertIn(key, case["alone"][phase])
        self.assertTrue(report["census_before"]["metrics"])
        self.assertIn("tensorfold:kv_cache_usage_perc", " ".join(report["census_before"]["metrics"]))

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
