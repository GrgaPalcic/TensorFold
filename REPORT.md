# T09b report — `tools/share_gate.py` fixed after its first live run

## What was done

Four defects from the first run against the production lane (GLM-5.3-Flash at TP=2, 2026-11-09
19:04) fixed in `tools/share_gate.py`, with the fake server in `tools/tests/test_share_gate.py`
extended for each and three new tests. Schema bumped to 2. Everything else (phases, drafted=plain,
dry-run, auth/redaction, census deltas, exit codes) kept as T09 left it.

1. **Token sizes ~3x low** — before any sized run the gate sends one small probe prompt (the filler
   at `PROBE_TOKENS = 256`) and reads `usage.prompt_tokens` from the reply; the ratio
   tokens-per-line is recorded in the report (`calibration: {probe_tokens, tokens_per_line}`) and
   `filler` sizes every prompt from it. Each sized run then reports `measured_tokens` beside
   `requested_tokens`; when they differ by more than `PROBE_TOLERANCE` (±5 %) the case lists
   `size_off`. The context limit is read from `GET /health` (`context`/`max_context`), else
   `/v1/models` `max_model_len`, into `report["context_limit"]`; a size that does not fit together
   with `max_tokens` (64) is skipped as a listed failure `size_over_context` (in the top-level
   `failures` list and the case) instead of crashing. Any exception during a run is caught and
   recorded with its full message (`{type}: {text}`), everywhere: sized runs, concurrent copies,
   census, calibration.
2. **Concurrent equals compared different prompts** — for every seed a level uses, the gate first
   takes that seed's own alone run (same body, sent alone) and stores its cold hash; each
   concurrent copy is then compared only with its own seed's alone hash. Shapes kept: seeds
   1..level−1 are distinct prompts (no shared prefix), seed 0 is the same-prompt copy (shared
   prefix with its own alone run). The report carries per copy: `seed`, `alone_sha256`,
   `alone_prompt`, `sha256` (concurrent), `equal_alone`. Failure lines name the seed, so the
   `failures` list has one entry per failing comparison — the repeated identical L8 lines are gone.
   Both draft shapes (draft=true and draft=false cells per level) still run, as in T09.
3. **Pool census found nothing** — the census matches any metric line whose name contains `kv_` or
   `prompt_cache` after an optional `tensorfold:` prefix, labels included (`kv_cache_usage_perc{stream="0"}`,
   `kv_cache_usage_ratio{pool="0"}`, `kv_shard_*` parts, `prompt_cache_*`, bare names on older
   builds). Verbatim lines and the appeared/disappeared/changed deltas are kept; the matched metric
   names still print on stdout for the operator to pair with the server's `kv pool idle:` log line.
   (`METRIC_PREFIX`/`METRIC_KEYWORDS` replace the old `POOL_METRIC`S startswith match, which only
   knew `kv_pool_`/`prompt_cache_` without a prefix.)
4. **Warm case says where it resumed** — `usage.prompt_tokens_details.cached_tokens` is recorded
   per run (`resumed_from`, already in T09's schema) and now printed as the `cached` column of the
   stdout table (a dash when the server omits it). The tokens column prints the measured
   `prompt_tokens`, not the nominal size.

Also: the stdout table row for a skipped/failed size prints the failure name (`size_over_context`,
or the error) instead of a bare `error`.

## Test commands run and results

From `tools/` (host Python 3.14.9; stdlib only, like the T09 tool):

| command | result |
|---|---|
| `python3 -m unittest tests.test_share_gate -v` | **Ran 8 tests — OK** (0 failures, 0 errors); green repeatedly (≥6 runs) |
| `python3 -m unittest discover -s tests -p test_share_gate.py` | Ran 8 tests — OK |
| `python3 -m py_compile share_gate.py tests/test_share_gate.py` | OK |
| fault-injection smokes: `break_warm`, `break_concurrent`, `draft_differs`, `oversize_error` | each fails with the named comparison, exit 1; clean server passes, exit 0 |
| CLI smoke `--dry-run --sizes 128 --levels 1,2` | plan JSON, 1 cell with 4 runs, 4 concurrent cells (2 levels × 2 drafts), no HTTP |
| `zig fmt --check build.zig zig` | clean (no Zig touched, rule honored) |

The 8 tests: T09's five (equal hashes pass, differing warm fails, concurrent compare passes/fails,
report schema, dry-run — the schema test extended for the new keys) plus the three new ones:
`test_oversize_is_listed_failure` (600000 against a context limit of 8192 → `size_over_context`
listed, no `error` key), `test_census_finds_prefixed_metrics` (tensorfold:-prefixed and bare names
matched, `unrelated_metric` not), `test_cached_reported` (cold `resumed_from == 0`, warm > 0,
`cached` in the table). Calibration is exercised by every test: the fake server reports
`usage.prompt_tokens` proportional to the prompt's line count, so the probe really fixes the ratio
(128-token asks measure 132, within ±5 %). No test was weakened, skipped or deleted.

## What was adapted, and why

- **Per-seed alone references replace the single shared reference.** T09 reused seed 0's 4,096-token
  cold run as the reference for every copy; the task's fix (one alone run per seed) supersedes it.
  References are taken once per seed before the cells (not per level), so a level-N run sends
  `max_level` alone runs plus the cells themselves.
- **Concurrent copies are repeats of their seed's alone prompt.** Under the per-seed fix the same
  prompt necessarily ran alone first; that is what makes the comparison meaningful (and warms the
  shared prefix for the seed-0 copy, which is the shape the lane cares about).
- **The fake server's `break_concurrent` now keys on genuinely overlapping requests** (an in-flight
  counter under a lock) instead of "any request after the drafted run": with per-seed references
  taken after the size phase, the old condition broke the reference alone runs too, which made the
  comparison vacuous. Only truly simultaneous copies are broken now; the alone runs stay intact.
  Like before, only drafted runs are broken (`draft is not False`), so the draft=false cell stays
  clean — drafted=plain remains asserted by `draft_differs`.
- **Warm/`cached` on the fake server**: the first-ever request for a prompt reads 0, repeats report
  `prompt_tokens − 4` — the live lane read 0 on its first cold request and positive on warm ones.
- **Calibration counts the header line**: `per_line = probe_tokens / filler_lines` absorbs the
  fixture header and the tokenizer's real ratio at once; sized prompts are built with
  `round(tokens / per_line)` lines.
- **`plan()` unchanged** (it still plans both draft shapes per level, now matching the real run
  again), `SCHEMA_VERSION` bumped to 2 so consumers can tell the reports apart.

## Left / uncertain

- The ±5 % tolerance and `PROBE_TOKENS = 256` are constants; if the operator wants a different
  probe size or tolerance they are one-line changes. Sizes are also refused only against the
  context limit the server reports — if `/health` and `/v1/models` both omit it, oversized
  requests fall back to the server's own error, recorded per case with its message.
- The census matches by name substring (`kv_`, `prompt_cache`); a metric that renamed away from
  both keywords would still be missed — to be re-checked on the next lane run.
- Concurrent cells keep one connection per request; connect refusals under load are reported per
  copy (with message) as unverified, not retried — as in T09.

## GPU checks needed

None to write it (per the task). The operator reruns on the lane:
`python3 tools/share_gate.py http://127.0.0.1:8888 --model <model> --out report.json` — expect
`calibration.tokens_per_line ≈ 3` (the ~3x-low defect) with measured tokens within ±5 % of each
asked size, `context_limit = 262144`-ish read from `/health` (600000 then listed as
`size_over_context`, not a crash), per-copy `alone_sha256`/`sha256`/`equal_alone` at levels 1, 2,
4, 8 for both draft shapes, the `tensorfold:kv_*` names in `pool metrics:`, the `cached` column
non-zero on warm runs, and one failure line per failing comparison.
