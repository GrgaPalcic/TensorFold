# T09 report — `tools/share_gate.py`, the served exactness gate

## What was done

Wrote the gate from scratch in this tree (no upstream file to port; upstream's own gate tools
`bench_openai.py`, `bench_concurrent.py`, `bench_cache.py` and PLAN.md section 4 "Gates" were the
reference for request style, streaming parse and comparison discipline):

- **`tools/share_gate.py`** (353 lines, standard library + the tree's `bench_auth`/`bench_evidence`):
  - `filler(seed, tokens)` — deterministic prompts from a fixed seed: numbered lines of NATO words,
    roughly 4 tokens per line, one user message asking to count the lines and reply with the number only.
    Requests carry `temperature: 0`, `max_tokens: 64`, `stream: true` with `include_usage`,
    `chat_template_kwargs: {"enable_thinking": false}` unless `--thinking`.
  - Resumed = fresh: per size, three sequential runs — cold, warm (same prompt, should resume), appended
    (one word more, resumes at the last mark). Cold and warm `sha256(content)` must be equal; the appended
    case is recorded and reported, not compared.
  - Drafted = plain: every size once with `"draft": false` in the body, hash must equal the same prompt's
    cold run.
  - Concurrent = alone: per level in `--levels`, that many requests sent at once through a
    `threading.Barrier` + `ThreadPoolExecutor` (never more than the level's maximum at a time): level − 1
    copies with distinct seeds (no shared prefix) plus one copy of the same 4,096-token prompt as the
    alone reference, each hash compared with that reference; a None comparison (failed or unverified run)
    is also a failure, following `bench_evidence`'s "unverified is never a pass".
  - Pool census: `GET /metrics` and `GET /health` before and after the whole run; every metric line whose
    name starts with `kv_pool_` or `prompt_cache_` (kv parts included when present) is recorded verbatim,
    deltas (appeared / disappeared / changed) reported, and the matched metric names printed on stdout so
    the operator can pair them with the server's own `kv pool idle:` log line (which is not in the API).
  - Report: JSON file (`--out`) with per-request size, hash, TTFT (first content delta), total time,
    full `usage`, `resumed_from` (= `usage.prompt_tokens_details.cached_tokens` when the server reports
    it), cache state, plus top-level `pass: true|false` and the failing comparisons listed; a one-screen
    table on stdout (case, tokens, cold s, warm s, equal?).
  - Safety: `--dry-run` prints the plan (phase/draft/prompt hashes) without sending; `--timeout` bounds
    the whole run and each request's read; the run stops at the first HTTP/stream error and reports it;
    `--key` is validated with `bench_auth.validate_api_key` and redacted out of the report.
- **`tools/tests/test_share_gate.py`** (147 lines): a fake OpenAI server (`http.server` in a thread,
  streaming `data:` lines whose content is derived from the prompt hash, `/metrics` and `/health`
  stubs) and five tests: equal hashes pass (and warm == cold), a differing warm hash fails with
  `warm_hash_differs` and exit 1, the concurrent comparison passes and fails when a concurrent reply
  differs from alone, the report schema (all top-level and per-case keys, census metric names), and
  `--dry-run` (no HTTP, 4 planned runs per size, concurrent cells planned).

## Test commands run and results

From `tools/` (Python 3.14.9 host; the tool requires 3.11+ like the other tools, PLAN targets 3.12 —
only stdlib is used, no version-specific behaviour):

| command | result |
|---|---|
| `python3 -m unittest tests.test_share_gate -v` | **Ran 5 tests — OK** (0 failures, 0 errors), repeatedly green |
| `python3 -m unittest discover -s tests -p test_share_gate.py` | Ran 5 tests — OK |
| `python3 -m pytest tests/test_share_gate.py -q` | not available on this host (`No module named pytest`); the unittest entry covers it |
| `python3 -m py_compile share_gate.py tests/test_share_gate.py` | OK |
| CLI smoke: `share_gate.py --dry-run --sizes 128 --levels 1,2` | plan JSON printed, no HTTP |
| CLI smoke against the fake server, `--out` report | exit 0, table + `pool metrics:` line, JSON schema complete |
| fault-injection smokes: `break_warm`, `break_concurrent`, `draft_differs` | each fails with the named comparison and exit 1 |

No engine code was touched, so no `zig build` target applies. No `*.rej` files were produced (nothing
was patch-applied; the files did not exist upstream at our pin).

## What was adapted, and why

- **Reuses `bench_evidence.Evidence`** for hashing content and `bench_auth` for authorization/redaction,
  instead of hand-rolling, so the fingerprint semantics (network chunk boundaries cannot alter the hash)
  and credential hygiene match upstream's tools exactly.
- **`resumed_from`** is reported from `usage.prompt_tokens_details.cached_tokens` when present (our pin's
  server reports it; absent → `null`), not treated as a pass condition by itself — the hash comparison
  decides.
- **The concurrent cell's alone reference** is the cold run of the same 4,096-token prompt taken in the
  size phase (sizes containing 4096) or taken once before the cells otherwise, so the comparison is
  strictly "same prompt alone vs. that prompt at level N".
- **Failure counting**: any unverified comparison (`None` from a failed run or a missing reference hash)
  fails the gate, matching `bench_cache.qualify`'s stance that unknown cannot satisfy an assertion.

## Left / uncertain

- `filler`'s tokens-per-line ratio (≈4) is calibrated for the GLM tokenizer by eye, not measured; the
  first reply's real `usage.prompt_tokens` is recorded (`measured_prompt_tokens` and per-case), which is
  what the operator should compare against `--sizes`. If the ratio is off at 600k the prompt is still
  deterministic and the comparisons remain valid; only the nominal size label drifts.
- `--levels` concurrency uses one connection per request on the same socket pool; if the native server's
  listen backlog rejects simultaneous connects the failures are reported per run (following
  `bench_concurrent`'s "resets are reported, not hidden"), not retried.
- The census matches metric lines by name prefix only (`kv_pool_`, `prompt_cache_`); the recipe's part
  metrics (under 0110, `cuda_kv_shard` part pools) will be picked up as long as they carry one of those
  prefixes — to be confirmed on the pair's first baseline run.

## GPU checks needed

None to write it (per the task). Its first real run — the baseline on the DGX pair with today's engine,
before and after each step-1 engine patch — is the operator's:
`python3 tools/share_gate.py http://127.0.0.1:8888 --model <model> --out report.json` on the pair, and
the gate must keep `pass: true` across: tokens unchanged with the switch unset, resumed = fresh at
4,096 / 126,976 / 600k, concurrent = alone at 1, 2, 4, 8, drafted = plain, and the pool census deltas
matching the server's `kv pool idle` log lines (both parts).
