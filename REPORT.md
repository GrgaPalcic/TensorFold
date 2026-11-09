# T07: port upstream `TF_GLM_KV_LIVE_HALF=1` (upstream `7547c99c2`, PR #2181)

## What was done

Ported upstream `7547c99c2` onto the pin + the recipe's 112 engine patches. With
`TF_GLM_KV_LIVE_HALF=1` a kept prompt state carries the live half of the KDA
recurrent state alone (the plan's arena head splits into two parts around the
dead half, `rec + (1-h) * kda_layers * rec_layer` bytes wide); with the switch
unset every path is byte-for-byte as before and `Plan.len`/`count` are
unchanged (asserted by a dedicated test). The rule applies to the arena-head
(KDA) part only, never to 0110's latent parts, which keep their own
owner/`lat` handling in `keepAs`/`restore`/carries untouched.

The worktree was left mid-task by an earlier run: the five source changes and
the new test file were present and all `*.rej` files were already resolved, but
the port did not compile — `Plan.withHalf` had been ported with upstream's
`shared` slot dropped while the (upstream-verbatim) test file still passed six
arguments, and two 0110-era callers of `initFor`
(`cuda_disk_shard_test.zig`, `cuda_kv_shard_pages_test.zig`) still used the
old five-argument form. Finished the adaptation (see below), then ran the
tests. No `.rej` files remain.

Commits on `task/T07-live-half`:
- `f6496ea` — the port itself (the eight files upstream's commit touches,
  adapted; see the commit body).
- `299f925` — fixes for the lane's pre-existing test failures (below).
- this report.

## What was adapted, and why

- `cuda_snapshot_layout.zig` — 0110 renamed upstream's page-sharing slot to
  `latent` (owner-only latent rows). `initFor`/`sizeFor` therefore take the
  kept state's own `cur` bits beside `latent` (a restore must plan by the
  keep's parity, not the slot's, since the slot's `cur` may have moved on);
  `withHalf` takes `latent: bool` where upstream took `shared: usize`. `Plan`
  gains `half: ?u1`; `make` splits the head around the dead half only when a
  half is given.
- `cuda_snapshots.zig` — `keepAs`/`seal`/`restore`/`carryRecv` pass the kept
  state's `cur` (`e.cur.st.cur`, `s.cur`); 0110's `own`/latent-rows logic and
  carries unchanged.
- `cuda_ranks.zig` — 0110 added `kv_shard` words to the settings and the
  fingerprint; the new `kv_live_half` word is added beside them and the
  fingerprint `layout` version moves 15 → 16 (15 had no `kv_live_half`), so
  mixed ranks refuse by name. `Settings.read` accepts only `0`/`1` and names a
  bad value.
- Tests — new `cuda_snapshot_layout_test.zig` is upstream's file plus the
  task's switch-off test (plan `len`/`count` unchanged with the switch off);
  its `withHalf` calls pass `true` (latent) in the slot where upstream passed
  `shared = 0`. `cuda_ranks_test.zig` gains the switch's read/fingerprint test
  and expects layout 16; the mismatch-name test lists `kv_live_half`.

## Pre-existing failures, fixed without weakening (commit `299f925`)

NOTES.md records that these already failed at the pin + 112 patches, before
any port; each was reproduced at HEAD without this port before fixing. All
were stale constants or setups from the recipe's own patches, not test
weakening — every assertion keeps its strength, only the stale values move,
exactly as recipe patch 0046 itself adapted the same file's constants when
window_rows changed:

- `test-glm-snapshots-host` 4/26 and `test-glm-fp8-pages-host` 1/15: recipe
  patch 0094 (`prefill_rows` 2048 → 4096) moved `snap_grid` and `stateBytes`
  without updating these tests. Keeps and lookups now sit on the family's
  chunk grid (`glm.native.snap_grid`, 4,096 rows): `snap_min_bytes`' W16
  expectation becomes 260,046,848 (the extra 2,048 rows' caches, +36 MiB, on
  top of W16's +18 MiB over W8), the store tests keep/look up at the grid with
  prompts of `grid + 1` tokens, and the paged E4M3 store test keeps 4,096 rows
  with 9 page ids a sequence (8 rows' pages + 1 past them) in a 20-page pool.
  Eviction counts, page-accounting and restore-equality asserts are unchanged
  and still hold (budgets scale with `stateBytes(grid)`).
- `test-cuda-host` `draft_sources_cuda_test` 1/112: recipe patch 0014 made
  GLM's `facts()` read `e.drafts` (speculation on) instead of `e.mtp`; the
  test left `e.drafts` undefined. It now sets both, as `cuda_engine.heads()`
  derives them.
- `test-cuda-host` `cuda_mm` "1 errors were logged": recipe patch 0100's own
  `TF_GLM_MM_STAGES` test exercises the mistyped-value fallback, which logged
  `std.log.err` — the test runner fails any run that logs an error, so the
  recipe's test could never pass as written. The note is now a warning (a
  mistyped knob falls back to the default, as the codebase's other soft
  fallbacks log); the test's assertions are untouched.

## Test commands run and counts (all green)

- `zig build test-cuda-host --summary all -j8` — 274/274 steps succeeded,
  588/589 tests passed (1 skipped, pre-existing). This target is where
  `cuda.zig`'s test block runs, i.e. `cuda_ranks_test.zig`, the new
  `cuda_snapshot_layout_test.zig`, 0110's `cuda_kv_shard_*` and
  `cuda_disk_shard_test.zig`. The glm module test inside it: 232/232 passed,
  including the 4 new live-half layout tests, the new `TF_GLM_KV_LIVE_HALF`
  ranks test, and all of 0110's own kv-shard tests.
- `zig build test-glm-snapshots-host --summary all -j8` — 26/26 passed.
- `zig build test-glm-fp8-pages-host --summary all -j8` — 15/15 passed.
- `zig build test-glm-kv-pages-host --summary all -j8` — 30/30 passed.
- `zig build test-glm-mm-policy-host test-glm-prompt-settings-host
  test-glm-snapshots-host test-glm-fp8-pages-host test-glm-kv-pages-host
  --summary all -j8` — 16/16 steps, 71/71 tests passed.
- `zig build test-glm-expiry test-glm-gathers --summary all -j8` — 32/32
  passed (their "failed command" line is a test intentionally exercising a
  warning path; the steps succeed).
- `zig fmt --check build.zig zig` — passes.

There is no `test-glm-ranks-host` or `test-glm-kv-shard-host` target on this
tree; the ranks and kv-shard tests run inside `test-cuda-host` (and the
snapshot-side tests inside `test-glm-snapshots-host`), which is what was run.

Still failing on this lane, untouched by this task (per NOTES.md, pre-existing
and outside this task's files): `test-glm-prompts-host` 1/7 (patch 0095 raised
`dflash.MAX_CTX` to 4,096; the `DrafterContextTooWide` test is stale).

## GPU checks needed later

As TASK.md says: on the DGX boxes, with `TF_GLM_KV_LIVE_HALF=1`,
- resumed == fresh with drafts on at TP=2 (`tf-cuda-test glm-kv-shard-check`
  still byte-identical);
- a 126k conversation's second turn's token hash equals a fresh pass;
- and, since the keep now leaves the dead half out of the lease, a
  keep/restore across a slot whose `cur` has flipped since the keep (the
  restore must rebuild the slot's dead half from the kept parity, not the
  slot's);
- also with TF_GLM_KV_SHARD on: a sharded keep at half `h` restored on a slot
  at the other parity, with and without a carry across ranks.
