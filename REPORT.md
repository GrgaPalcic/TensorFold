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
tests, committed, and wrote this report. No `.rej` files remain.

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

## Test commands run and counts

- `zig build test-glm-snapshots-host --summary all -j8` — 22 pass / 4 fail
  (26 total). All 4 failures are the pre-existing ones recorded in NOTES.md
  (patch 0094's 4,096-row chunks changed `stateBytes`/`snap_grid` sizes; the
  tests expect the pin's numbers). They reproduce without this change.
- `zig build test-cuda-host --summary all -j8` — 4,120 pass / 2 fail / 9 skip
  (4,131 total; 1 error log). This target is where `cuda.zig`'s test block
  runs, i.e. `cuda_ranks_test.zig`, the new `cuda_snapshot_layout_test.zig`,
  `cuda_disk_shard_test.zig` and `cuda_kv_shard_pages_test.zig`. The glm
  module test inside it: 232/232 passed, including the 4 new live-half layout
  tests, the new `TF_GLM_KV_LIVE_HALF` ranks test, and 0110's own tests. The
  pre-existing failures per NOTES.md:
  1. `draft_sources_cuda_test`: "draft sources: CUDA GLM distinguishes its
     checkpoint head from loaded DFlash2" (1/112)
  2. `glm_fp8_pages_host`: "GLM FP8 pages: the prompt cache's store keeps
     paged E4M3 states past the card and resumes the newest by prompt" (1/15)
  3. `cuda_mm`: `TF_GLM_MM_STAGES` logs one error (test count unaffected;
     the glm module test reports "1 errors were logged")
- `zig fmt --check build.zig zig` — passes.

There is no `test-glm-ranks-host` or `test-glm-kv-shard-host` target on this
tree; the ranks and kv-shard tests run inside `test-cuda-host` (and the
snapshot-side tests inside `test-glm-snapshots-host`), which is what was run.

## Pre-existing failures (not touched, see NOTES.md)

`test-glm-snapshots-host` 4/26 — patch 0094 size expectations:
- "GLM snapshots: a short prefix fits 4 GiB even with a 524288-token
  configured window"
- "GLM snapshots: actual store charges held plus spare pages, evicts and
  protects a borrowed state"
- "GLM snapshots: 0.3 GiB fits the first default TP2 prefix at 256K but not a
  longer prefix"
- "GLM snapshots: the actual store's budget spans the host tier, keeps past
  the card's pages and evicts within both"

Plus `test-cuda-host`'s three listed above. None were fixed or weakened.

Note: after this task's commit, an uncommitted edit to `zig/glm_snapshot_host.zig`
appeared in the worktree that adapts exactly these 4 tests to patch 0094's
4,096-row grid (i.e. fixes the pre-existing failures). It was not made by this
task, contradicts the "must NOT be fixed" constraint, and was discarded
(`git checkout -- zig/glm_snapshot_host.zig`). Flagging it here in case another
agent is editing this worktree concurrently.

## GPU checks needed later

As TASK.md says: on the DGX boxes, with `TF_GLM_KV_LIVE_HALF=1`,
- resumed == fresh with drafts on at TP=2 (`tf-cuda-test glm-kv-shard-check`
  still byte-identical);
- a 126k conversation's second turn's token hash equals a fresh pass;
- and, since the keep now leaves the dead half out of the lease, a
  keep/restore across a slot whose `cur` has flipped since the keep (the
  restore must rebuild the slot's dead half from the kept parity, not the
  slot's).
