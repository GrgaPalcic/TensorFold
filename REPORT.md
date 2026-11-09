# T09b report — `tools/share_gate.py` fixed after its first live run

## What was done

Four defects from the first production-lane run (GLM-5.3-Flash TP=2, 2026-11-09 19:04), fixed in
`tools/share_gate.py` (schema bumped to 2) with the fake-server tests extended for each:

1. **Sizes ~3× low / 600k overflow.** Before any sized run the gate sends one small calibration
   probe (`filler(0, 256)` filler lines) and takes `usage.prompt_tokens / lines` from the reply as
   this server's tokens per line (`report["calibration"] = {probe_tokens, tokens_per_line}`); every
   sized prompt is built with that ratio. The measured `prompt_tokens` of the cold run is reported
   as `measured_tokens` beside `requested_tokens` and printed in the table's `tokens` column; when
   measured and asked drift more than ±5 % the case lists `size_off`. The context limit is read
   from `GET /health` (`context` or `max_context`), else `/v1/models` (`max_model_len`), and a size
   that does not fit with `max_tokens` (64) is a listed `size_over_context` failure — no request is
   sent for it, no exception escapes. Any exception anywhere in a run is caught and recorded with
   its full message (`type: message`), in the census, calibration, size runs and concurrent cells
   alike (the first run printed bare class names such as `RuntimeError`).
2. **Concurrent equals compared different prompts.** For every level the gate now first takes each
   seed's own alone reply (256-token prompts to keep the reference runs cheap: seed 0 = the
   same-prompt shape, seeds 1..level−1 = distinct prompts with no shared prefix), then each
   concurrent copy is compared only with its own seed's alone hash. Each run record carries
   `seed`, `alone_sha256`, its own `sha256` and `equal_alone` (plus `alone_prompt` for the
   operator), and a failed comparison is one JSON failure entry per seed, so L8 no longer repeats
   one line. Both concurrent shapes kept: draft=True and draft=False cells per level, as in T09.
   The alone references are all taken before the cells so no reference is polluted by them.
3. **Pool census found nothing.** The census matches any metric whose bare name (labels stripped,
   verbatim line kept) contains `kv_` or `prompt_cache` after an optional `tensorfold:` prefix —
   the lane's `tensorfold:kv_cache_usage_perc{stream="0"}`,
   `tensorfold:kv_cache_usage_ratio{pool="0"}`, `tensorfold:kv_shard_*` (part, part_free,
   part_live_pages, …) and any `tensorfold:prompt_cache_*` on other builds all match, with or
   without the prefix. Verbatim lines and the appeared/disappeared/changed deltas are kept.
4. **Warm case says where it resumed.** `usage.prompt_tokens_details.cached_tokens` is recorded
   per run (`resumed_from`, and `cached_tokens` via `bench_evidence.Evidence`), printed as a new
   `cached` column in the stdout table (a dash when the server omits it); the cold first request
   reads 0 as on the lane.

## Test commands run and results

From `tools/` (Python host):

| command | result |
|---|---|
| `python3 -m unittest tests.test_share_gate` | **Ran 8 tests — OK**, repeated 6×, stable (5 in T09 + 3 new: oversize listed failure, prefixed census, cached column) |
| `python3 -m unittest discover -s tests -p test_share_gate.py` | Ran 8 tests — OK |
| `python3 -m py_compile share_gate.py tests/test_share_gate.py` | OK |
| `zig fmt --check build.zig zig` | clean (no Zig files touched) |
| CLI smoke `--dry-run --sizes 128 --levels 1,2` | plan printed, 4 concurrent cells (2 levels × 2 drafts), no HTTP |
| fault injection `break_warm` | exit 1, `warm_hash_differs` + `drafted_hash_differs` listed |
| fault injection `break_concurrent` (levels 2) | exit 1, per-seed `hash differs from alone` entries |
| fault injection `draft_differs` | exit 1, `drafted_hash_differs` |
| fault injection `oversize_error` (server 400s) | exit 1, `size 600000: size_over_context (context limit 8192)`, no traceback |
| clean run, levels 2, both draft cells | exit 0, `pass: true`, census finds the four prefixed names + excludes `unrelated_metric` |

The fake server grew: `usage.prompt_tokens` proportional to the prompt's line count (exercises the
calibration ratio), `/health` with `context: 8192`, `cached_tokens` 0 on the first-ever request and
`prompt_tokens − 4` on warm ones, and `tensorfold:`-prefixed metrics plus one unprefixed and one
unrelated line. Its `break_concurrent` fault now detects real request overlap (in-flight counter
under a lock) instead of "any drafted run after the size phase", which the per-seed reference runs
had made ambiguous.

## What was adapted, and why

- **Reference prompts are 256 tokens, not the sized prompts.** Taking each level's alone replies
  from the sized runs would compare a 4096-token concurrent copy with, say, the 126,976 alone
  reply; the per-seed references are their own prompts at a size cheap enough to run 8× (max
  level) without dominating the gate's wall time. The same-prompt shape (seed 0) is unchanged in
  kind: one prompt, alone then concurrent.
- **`break_concurrent` semantics in the fake server changed** (overlap-based, see above) because
  the old "after the drafted run" rule would also break the per-seed alone references, making the
  fault inject nothing. The unittest still only asserts exit 1; at a level-2 cell exactly one of
  the two overlapping copies is broken by the fake server, which is enough to fail the cell.
- **Census matching is by substring after the prefix, not by prefix list**, so recipe-patch metric
  families (`kv_shard_*` parts under 0110, `prompt_cache_*` on builds with 0108) are picked up
  without enumerating them; `unrelated_metric` is excluded and labels stay in the verbatim line.
- **`report["measured_prompt_tokens"]` (schema 1) is replaced by per-case `measured_tokens`** plus
  `report["calibration"]`, since sizes now differ from what the tokenizer counts by construction
  of the ratio, not by a fixed guess.

## Left / uncertain

- The ±5 % tolerance assumes the probe's ratio holds at 600k lines; if the tokenizer's per-line
  cost drifts with very long prompts the gate lists `size_off` (a report statement, not a crash)
  and the comparisons themselves stay valid.
- The calibration probe counts lines as `probe.count("\n")` (filler lines, header excluded from
  the denominator but present in the prompt): the header's ~15 tokens are part of
  `usage.prompt_tokens`, so `tokens_per_line` is a hair above the pure filler ratio and the sized
  prompts land within the tolerance of the asked size — confirmed by the fake server's exact
  counting, to be re-confirmed on the lane's tokenizer.
- Concurrent cells still use one connection per request; backlog rejects are reported per run, not
  retried (unchanged from T09).

## GPU checks needed

None to write it; the operator reruns on the lane: `python3 tools/share_gate.py
http://127.0.0.1:8888 --model <model> --out report.json` — the calibration line should report a
`tokens_per_line` near the lane's real ratio (the first run's ~3× error must disappear), the
600k size must either fit (with the measured `tokens` within ±5 %) or be listed as
`size_over_context`, the L4/L8 cells must compare per seed, `cached` must read 0 cold and >0 warm
(`usage.prompt_tokens_details.cached_tokens` present), and the census must list the
`tensorfold:kv_*` / `tensorfold:kv_shard_*` names seen in the server's own log.
