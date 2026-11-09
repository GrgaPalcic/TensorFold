"""The served exactness gate for kept-state ports: resumed = fresh, concurrent = alone, drafted = plain.

Run against a live OpenAI-compatible server (tensorfold-native serve) before and after each engine change of
the KV-share port; the JSON report it writes is what decides whether a build ships. Deterministic filler
prompts from a fixed seed, temperature 0, streaming so time to first token is observed; every comparison is on
sha256 of the streamed content. Prompt sizes are calibrated against the live server's tokenizer before the
sized runs (one small probe fixes the tokens-per-line ratio), and the context limit is read from /health (or
/v1/models) so an oversized request is a listed failure, not an exception. A pool census (GET /metrics and
/health) is taken before and after the run and the deltas reported; the server's own idle line is in its log,
not the API, so the metric names found are printed for the operator to match. Standard library only, so it
runs on a bare host.

  python3 tools/share_gate.py http://127.0.0.1:8888 --model MODEL --out report.json
  python3 tools/share_gate.py http://127.0.0.1:8888 --model MODEL --sizes 128 --levels 1,2 --out t.json
  python3 tools/share_gate.py http://127.0.0.1:8888 --model MODEL --dry-run
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from bench_auth import authorize, redact, validate_api_key
from bench_evidence import Evidence

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]}); "
             "TensorFold's tools use PEP 604 unions. Try: python3.12 tools/share_gate.py ...")

PHASES = ("cold", "warm", "appended")
SCHEMA_VERSION = 2
PROBE_TOKENS = 256
PROBE_TOLERANCE = 0.05  # a measured prompt_tokens within +-5 % of the asked size passes
HEX64 = re.compile(r"[0-9a-f]{64}")
METRIC_PREFIX = "tensorfold:"
METRIC_KEYWORDS = ("kv_", "prompt_cache")


def filler(seed: int, tokens: int, per_line: float = 4.0) -> str:
    """A deterministic prompt of roughly ``tokens`` tokenizer tokens: numbered lines of words from a fixed list."""
    words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
             "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
             "xray yankee zulu").split()
    # per_line tokens per line as measured by the calibration probe (roughly 4 by eye before it runs)
    lines = max(1, round(tokens / per_line))
    out = []
    for i in range(lines):
        out.append(" ".join(words[(seed + i * 7 + j * 3) % len(words)] for j in range(4)) + f" line {i}")
    return (f"Fixture seed {seed}: count the lines below and reply with the number only.\n" + "\n".join(out))


def context_limit(base: str) -> tuple[int | None, str | None]:
    """The server's context limit: /health's ``context``/``max_context``, else /v1/models' ``max_model_len``."""
    try:
        with urllib.request.urlopen(base + "/health", timeout=30) as resp:
            health = json.loads(resp.read().decode())
        for key in ("context", "max_context"):
            if isinstance(health.get(key), (int, float)) and health[key] > 0:
                return int(health[key]), None
    except (ValueError, urllib.error.URLError, OSError) as exc:
        return None, f"health: {exc}"
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=30) as resp:
            models = json.loads(resp.read().decode())
        for model in models.get("data", []):
            if isinstance(model.get("max_model_len"), (int, float)) and model["max_model_len"] > 0:
                return int(model["max_model_len"]), None
    except (ValueError, urllib.error.URLError, OSError) as exc:
        return None, f"models: {exc}"
    return None, None


def metrics_snapshot(base: str) -> dict:
    """The KV-pool census: metric lines containing kv_ or prompt_cache after an optional tensorfold: prefix."""
    snap: dict = {"metrics": {}, "health": None, "error": None}
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=30) as resp:
            body = resp.read().decode()
        for line in body.splitlines():
            name = line.split(" ")[0]
            bare = name.split("{", 1)[0]
            if bare.startswith(METRIC_PREFIX):
                bare = bare[len(METRIC_PREFIX):]
            if "kv_" in bare or "prompt_cache" in bare:
                try:
                    snap["metrics"][line] = float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    snap["metrics"][line] = line.rsplit(" ", 1)[-1]
    except (urllib.error.URLError, OSError) as exc:
        snap["error"] = f"{type(exc).__name__}: {exc}"
    try:
        with urllib.request.urlopen(base + "/health", timeout=30) as resp:
            snap["health"] = resp.read().decode()[:200]
    except (urllib.error.URLError, OSError) as exc:
        snap["error"] = snap["error"] or f"{type(exc).__name__}: {exc}"
    return snap


def census_delta(before: dict, after: dict) -> dict:
    """Metric lines that appeared, disappeared or changed value between the two censuses."""
    out: dict = {"appeared": [], "disappeared": [], "changed": []}
    for line, value in after["metrics"].items():
        if line not in before["metrics"]:
            out["appeared"].append(line)
        elif before["metrics"][line] != value:
            out["changed"].append(line)
    for line in before["metrics"]:
        if line not in after["metrics"]:
            out["disappeared"].append(line)
    return out


def request_case(base: str, model: str, prompt: str, *, draft: bool, appended: bool, thinking: bool,
                 api_key: str | None = None, timeout: float = 3600.0) -> tuple[dict, str]:
    """One streamed chat request; returns its record and the sha256 of the streamed content."""
    body = {"model": model, "max_tokens": 64, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "messages": [{"role": "user", "content": prompt}]}
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    authorize(req, api_key)
    sent = time.perf_counter()
    evidence = Evidence()
    usage: dict = {}
    first = None
    complete = False
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                complete = True
                break
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            if "error" in chunk:
                raise RuntimeError("server_stream_error")
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                evidence.add(choice)
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    first = first if first is not None else time.perf_counter()
    total = time.perf_counter() - sent
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    details = details if isinstance(details, dict) else {}
    observed = evidence.finish(usage, {}, complete)
    if not complete:
        raise RuntimeError("incomplete_stream")
    record = {"phase": "appended" if appended else "sent", "draft": draft, "ttft_s": first - sent if first else None,
              "total_s": round(total, 3), "usage": usage,
              "cached_tokens": observed["cached_tokens"],
              "resumed_from": details.get("cached_tokens"),
              "prompt_tokens": observed["prompt_tokens"], "output_sha256": observed["output_sha256"],
              "cache_state": observed["cache_state"]}
    return record, observed["output_sha256"]


def alone(base: str, model: str, prompt: str, *, thinking: bool, api_key: str | None,
          timeout: float = 3600.0) -> dict:
    """cold, warm (should resume) and appended (one word more) runs of one prompt, sequentially."""
    out: dict = {}
    for phase in PHASES:
        text = prompt + " extra" if phase == "appended" else prompt
        record, sha = request_case(base, model, text, draft=True, appended=phase == "appended",
                                   thinking=thinking, api_key=api_key, timeout=timeout)
        record["sha256"] = sha
        out[phase] = record
    return out


def one_concurrent(base: str, model: str, prompt: str, seed: int, label: str, *, draft: bool,
                   thinking: bool, api_key: str | None, gate: threading.Barrier | None,
                   timeout: float = 3600.0) -> dict:
    """One of the ``level`` requests sent at once; the reply's hash is compared with its own seed's alone run."""
    try:
        if gate is not None:
            gate.wait(timeout=30)
        record, sha = request_case(base, model, prompt, draft=draft, appended=False, thinking=thinking,
                                   api_key=api_key, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - a failed request is reported, not fatal
        if gate is not None:
            gate.abort()
        return {"label": label, "seed": seed, "sha256": None, "equal_alone": None,
                "error": f"{type(exc).__name__}: {exc}"}
    return {"label": label, "seed": seed, "sha256": sha, **record}


def concurrency(base: str, model: str, reference: dict, level: int, *, draft: bool,
                thinking: bool, api_key: str | None, timeout: float = 3600.0) -> dict:
    """``level`` copies at once: seeds 1..level-1 distinct plus one seed-0 same-prompt run, each against its own seed's alone hash."""
    gate = threading.Barrier(level)
    with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
        futures = [pool.submit(one_concurrent, base, model, reference["prompts"][seed], seed, "distinct",
                               draft=draft, thinking=thinking, api_key=api_key, gate=gate, timeout=timeout)
                   for seed in range(1, level)]
        futures.append(pool.submit(one_concurrent, base, model, reference["prompts"][0], 0, "same", draft=draft,
                                   thinking=thinking, api_key=api_key, gate=gate, timeout=timeout))
        out = [f.result() for f in futures]
    for run in out:
        ref = reference["hashes"].get(run["seed"])
        run["alone_sha256"] = ref
        run["alone_prompt"] = reference["prompts"].get(run["seed"])
        run["equal_alone"] = None if run["sha256"] is None or not HEX64.fullmatch(ref or "") \
            else run["sha256"] == ref
    return {"level": level, "draft": draft, "runs": out}


def compare(alone: dict) -> list[str]:
    """The failing comparisons: a differing warm hash or a failed run; the appended case is reported, not compared."""
    fails = []
    cold, warm_sha = alone["cold"]["sha256"], alone["warm"]["sha256"]
    if not HEX64.fullmatch(cold):
        fails.append("cold_failed")
    elif not HEX64.fullmatch(warm_sha):
        fails.append("warm_failed")
    elif warm_sha != cold:
        fails.append("warm_hash_differs")
    if not HEX64.fullmatch(alone["appended"]["sha256"]):
        fails.append("appended_failed")
    return fails


def calibration_probe(base: str, model: str, *, thinking: bool, api_key: str | None,
                      timeout: float) -> tuple[float, int | None]:
    """One small probe prompt: the reply's usage.prompt_tokens fixes this server's tokens per filler line."""
    probe = filler(0, PROBE_TOKENS)
    try:
        record, _ = request_case(base, model, probe, draft=True, appended=False, thinking=thinking,
                                 api_key=api_key, timeout=timeout)
        tokens = record.get("prompt_tokens")
        if not tokens:
            raise RuntimeError("usage.prompt_tokens missing from the probe reply")
        lines = probe.count("\n")  # every counted line is one filler line
        return tokens / lines, tokens
    except Exception as exc:  # noqa: BLE001 - a failed probe is a listed failure, not a crash
        raise RuntimeError(f"calibration probe failed: {type(exc).__name__}: {exc}") from exc


def run(base: str, model: str, sizes: list[int], levels: list[int], *, thinking: bool, timeout: float,
        api_key: str | None, seed: int = 0) -> dict:
    """The whole gate: calibrate, per-size alone runs, drafted runs, concurrent cells and a pool census."""
    deadline = time.monotonic() + timeout
    report: dict = {"schema_version": SCHEMA_VERSION, "base": base, "model": model, "thinking": thinking,
                    "sizes": sizes, "levels": levels, "cases": [], "concurrency": [], "failures": []}
    report["census_before"] = metrics_snapshot(base)
    if report["census_before"]["error"]:
        report["failures"].append(f"census_before: {report['census_before']['error']}")
    # calibration: one small probe fixes this server's tokens per filler line before any sized run
    try:
        per_line, probe_tokens = calibration_probe(base, model, thinking=thinking, api_key=api_key,
                                                   timeout=timeout)
        report["calibration"] = {"probe_tokens": probe_tokens,
                                 "tokens_per_line": round(per_line, 4)}
    except RuntimeError as exc:
        report["calibration"] = {"error": str(exc)}
        report["failures"].append(f"calibration: {exc}")
        report["pass"] = False
        return report
    limit, limit_error = context_limit(base)
    report["context_limit"] = limit
    if limit_error:
        report["failures"].append(f"context_limit: {limit_error}")
    for size in sizes:
        prompt = filler(seed, size, per_line=per_line)
        entry: dict = {"requested_tokens": size, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        fits = limit is None or size + 64 <= limit
        if not fits:
            report["failures"].append(f"size {size}: size_over_context (context limit {limit})")
            entry["failures"] = ["size_over_context"]
            report["cases"].append(entry)
            continue
        try:
            entry["alone"] = alone(base, model, prompt, thinking=thinking, api_key=api_key, timeout=timeout)
            record, sha = request_case(base, model, prompt, draft=False, appended=False,
                                       thinking=thinking, api_key=api_key, timeout=timeout)
            entry["drafted"] = {**record, "sha256": sha}
        except Exception as exc:  # noqa: BLE001 - a failed run is caught and recorded with its message
            entry["error"] = f"{type(exc).__name__}: {exc}"
            report["failures"].append(f"size {size}: {entry['error']}")
            report["cases"].append(entry)
            continue
        measured_tokens = entry["alone"]["cold"]["prompt_tokens"] or 0
        entry["measured_tokens"] = measured_tokens
        # the asked size must land within +-5 % of what the tokenizer counted, or the report says size_off
        entry["failures"] = []
        if measured_tokens and abs(measured_tokens - size) / size > PROBE_TOLERANCE:
            entry["failures"].append("size_off")
        entry["failures"] += compare(entry["alone"])
        # drafted = plain: the plain-decode run must hash equal to the cold run of the same prompt
        if entry["drafted"].get("sha256") is not None:
            if entry["drafted"]["sha256"] != entry["alone"]["cold"]["sha256"]:
                entry["failures"].append("drafted_hash_differs")
        else:
            entry["failures"].append("drafted_failed")
        report["failures"].extend(f"size {size}: {f}" for f in entry["failures"])
        report["cases"].append(entry)
        if time.monotonic() > deadline:
            report["failures"].append(f"timeout after {timeout:g}s")
            break
    # every seed used in a level first gets its own alone reply; copies compare only with it.
    # seed 0 is the same-prompt copy, seeds 1..level-1 the distinct ones
    max_level = max(levels, default=0)
    reference: dict = {"prompts": {}, "hashes": {}}
    for seed_ in range(max(max_level, 1)):
        prompt_ = filler(seed_, PROBE_TOKENS, per_line=per_line)
        reference["prompts"][seed_] = prompt_
        try:
            reference["hashes"][seed_] = alone(base, model, prompt_, thinking=thinking, api_key=api_key,
                                               timeout=timeout)["cold"]["sha256"]
        except Exception as exc:  # noqa: BLE001 - a failed alone run is a listed failure
            reference["hashes"][seed_] = None
            report["failures"].append(f"concurrent reference seed {seed_}: {type(exc).__name__}: {exc}")
    for level in levels:
        for draft in (True, False):
            try:
                cell = concurrency(base, model, reference, level, draft=draft, thinking=thinking,
                                   api_key=api_key, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - a failed cell is a listed failure
                report["failures"].append(
                    f"concurrent level {level} draft {draft}: {type(exc).__name__}: {exc}")
                continue
            report["concurrency"].append(cell)
            for run_ in cell["runs"]:
                if run_["equal_alone"] is False:
                    report["failures"].append(
                        f"concurrent level {level} draft {draft}: seed {run_['seed']} hash differs from alone")
                elif run_["equal_alone"] is None and "error" not in run_:
                    report["failures"].append(
                        f"concurrent level {level} draft {draft}: seed {run_['seed']} unverified against alone")
            if time.monotonic() > deadline:
                report["failures"].append(f"timeout after {timeout:g}s")
                break
    report["census_after"] = metrics_snapshot(base)
    if report["census_after"]["error"]:
        report["failures"].append(f"census_after: {report['census_after']['error']}")
    report["census_delta"] = census_delta(report["census_before"], report["census_after"])
    report["pass"] = not report["failures"]
    return report


def plan(sizes: list[int], levels: list[int], *, thinking: bool, draft: bool = True) -> dict:
    """``--dry-run``: the requests the tool would send, no HTTP."""
    cells = []
    for size in sizes:
        prompt = filler(0, size)
        cells.append({"requested_tokens": size, "runs": [
            {"phase": phase, "draft": True, "prompt_sha256": hashlib.sha256(
                (prompt + (" extra" if phase == "appended" else "")).encode()).hexdigest()} for phase in PHASES
        ] + [{"phase": "drafted", "draft": False,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}]})
    return {"cells": cells, "concurrent_cells": [{"level": level, "draft": draft,
                                                  "requests": level} for level in levels for draft in (True, False)]}


def summary(report: dict) -> str:
    """The one-screen table: case, tokens (measured), cached, cold s, warm s, equal?."""
    rows = ["case                tokens  cached  cold s   warm s   equal?"]
    for case in report.get("cases", []):
        alone_ = case.get("alone")
        if not alone_:
            note = case["failures"][0] if case.get("failures") else case.get("error", "error")
            rows.append(f"{case['requested_tokens']:<19} {note}")
            continue
        equal = "yes" if not case.get("failures") else "NO"
        cached = alone_["cold"].get("resumed_from")
        rows.append(f"{case['requested_tokens']:<19} {str(case.get('measured_tokens', alone_['cold']['prompt_tokens'])):>6} "
                    f"{str(cached if cached is not None else '-'):>6} "
                    f"{alone_['cold']['total_s']:>8.2f} {alone_['warm']['total_s']:>8.2f}   {equal}")
    for cell in report.get("concurrency", []):
        equal = "yes" if all(r["equal_alone"] is True for r in cell["runs"]) else "NO"
        rows.append(f"concurrent L{cell['level']} draft={cell['draft']!s:<5} {'':>6} {'':>6} {'':>8} {'':>8}   {equal}")
    rows.append(f"pass: {report['pass']}")
    if report.get("failures"):
        rows.append("failing comparisons:")
        rows.extend(f"  - {f}" for f in report["failures"])
    return "\n".join(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", nargs="?", help="the server root, for example http://127.0.0.1:8888")
    p.add_argument("--model", default="test-model")
    p.add_argument("--key", metavar="K", help="bearer API key sent with every request")
    p.add_argument("--out", metavar="FILE", help="write the JSON report here")
    p.add_argument("--sizes", default="4096,126976,600000", help="prompt sizes in tokens, a comma list")
    p.add_argument("--levels", default="1,2,4,8", help="concurrent requests per cell, a comma list")
    p.add_argument("--timeout", type=float, default=3600, help="seconds for the whole run")
    p.add_argument("--thinking", action="store_true", help="send enable_thinking: true")
    p.add_argument("--dry-run", action="store_true", help="print the plan without sending anything")
    args = p.parse_args(argv)
    api_key = args.key
    if api_key is not None:
        try:
            validate_api_key(api_key)
        except ValueError:
            p.error("API key must contain printable ASCII characters without whitespace")
    sizes = [int(x) for x in args.sizes.split(",")]
    levels = [int(x) for x in args.levels.split(",")]
    if min(sizes) < 1 or min(levels) < 1:
        p.error("sizes and levels must be positive")
    if args.dry_run:
        print(json.dumps(plan(sizes, levels, thinking=args.thinking), indent=2))
        return 0
    if not args.base:
        p.error("the server base URL is required unless --dry-run")
    report = run(args.base.rstrip("/"), args.model, sizes, levels, thinking=args.thinking,
                 timeout=args.timeout, api_key=api_key)
    report = redact(report, api_key)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(encoded)
    print(summary(report))
    # the metric names the census matched, so the operator can pair them with the server's idle line
    print("pool metrics:", ", ".join(sorted(report["census_after"]["metrics"])) or "(none found)")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
