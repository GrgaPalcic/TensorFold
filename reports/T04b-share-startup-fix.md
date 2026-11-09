# T04b report: `TF_KV_SHARE` opens GLM's snapshot pages once, inside the plan

## What was done

Fixed the startup crash of the 2026-11-09 22:48 GPU run (`share-port` c12c858, TP=2, `TF_KV_SHARE=32`,
`TF_PROMPT_CACHE_FIRST=1`, `--prompt-cache-gib 7`): rank 1 stopped with `SnapshotCacheMade` after the plan had
handed the whole `--prompt-cache-gib` to the pool's pages. Both halves of the diagnosis were confirmed on this
head (T04's `fcde9f6`), then fixed. One commit (`84caff8`), `REPORT.md` second.

1. **Made twice — fixed.** `native/cuda.zig` computed
   `kept_budget = try kv_pages.storeBudget(F, loaded.target, kept)` unconditionally (the recipe's host-tier
   shape). For GLM (no `snap_slots`), `storeBudget` → `cuda_snaps.cacheBudget` → `F.snap_ready` →
   `prepareTiers` a second time after `share_states`' `readySnapshots` had made the pages →
   `error.SnapshotCacheMade` (`cuda_snapshots.zig`: `e.snap_pages != null or e.snaps.count() != 0`). Upstream
   (`tf-at-share/zig/src/native/cuda.zig` ~379–386) opens the store only inside its `!share` branch, so the
   second make never happens there. Ported as `cuda_kv_pool.keptBudget(F, target, k, share)`: under a share it
   returns what the shared snapshots hold — the plan's `snaps × slot` plus `snap_host_tier`'s host pages — and
   `storeBudget` (`snap_ready` and all) runs only without a share. `cache_capacity.budget_bytes` (the startup
   warning and metrics) and the `!share` store-opening guard keep their shape.
2. **Outside the plan — T04 already does it; verified and pinned.** With GLM's `share_kept_ppm = 0`:
   - (a) the plan reserves `snaps × slot` for the kept states' copies before the pool takes pages, under
     `TF_PROMPT_CACHE_FIRST=1` and without it: `finishRank` declares the slot (`cuda_native.zig` ~363,
     `l.snap_bytes = pages_mod.shareSlot(e)`), `native/cuda.zig` passes `slots = kv_prompts.sharing()` when
     `share and l.snap_bytes > 0 and !host_cache`, and `kv_pool.plan`'s `slots_only` branch takes
     `kept = cache - split(cache, l.snap_bytes, b.snaps).snaps * l.snap_bytes`, then subtracts
     `snaps × slot + copied` from the room before counting pages.
   - (b) `share_states` allocates exactly the planned bytes (`readySnapshots(own, snaps * slot)`, the plan's
     `snaps`), the named count is clamped to what the cache holds (`split`'s `cache / snap_bytes` and
     `most_snaps`), and the startup line says so (`kv_pages.cacheText`: "N of TF_KV_SHARE=n kept states' slots
     (…; --prompt-cache-gib G holds all n)"); `share_mod.advice` warns under 4 states a stream.
   - (c) the rest of `--prompt-cache-gib` becomes pool pages (`Plan.kept`, counted in the pages' bound `most`).
   - Adapted to match upstream's semantics: `native/cuda.zig` now passes `F.share_kept_ppm` (GLM: 0) as the
     kept ppm instead of a flat `1_000_000` — upstream `cuda_streams.keptPpm`'s form, inlined (our pin has no
     `cuda_streams.zig`). No behaviour change on any reachable path: when slots are planned, `slots_only`
     ignores the ppm; under a host cache, host mode forces ppm 0; GLM's `shareSlot` is never 0 under share.
     With `TF_KV_SHARE` unset the ppm is 0 either way.

## What was adapted and why

- `keptBudget` is a new helper rather than an inlined condition: `native/cuda.zig`'s open path is host-untestable
  (it needs a device), and the helper carries the once-only contract where the host fixtures can pin it. The
  `!share` branch is the old `storeBudget` byte-for-byte; families without `snap_save` still get 0.
- Under a share the budget includes `snap_host_tier`'s bytes, mirroring what `cacheBudget` returns for the flat
  store (device budget + host tier): the shared snapshots hold both tiers too (`readySnapshots` passes the host
  ask). With no slots planned (host cache mode, or a cache under one slot) it is the host tier alone or 0.
- The doc hunk and upstream's `cuda_streams.zig` test stay unported (no such file here, as in T03/T04).

## Off by default

`TF_KV_SHARE` unset (or 0): `share` is false → `kept_ppm` 0, `slots` null, `keptBudget` returns `storeBudget`'s
exactly-former value, the store opens as before, and every startup line is byte-identical. With the variable set
but `TF_GLM_KV_SHARD=1` or `--learn-conversations`, the refusals fire before any of this runs.

## Tests run (Zig 0.17.0, host targets, `-j8`)

New tests:

- `cuda_kv_pool.zig` "cache capacity: under a share the made snapshots are the budget, and snap_ready runs once" —
  a family declaring `snap_ready` + `share_kept_ppm = 0` + `snap_host_tier`: after the share's make,
  `keptBudget(share=true)` returns the slots' plus host bytes and `snap_ready` has run **once**; the unguarded
  flat-store path (`keptBudget(share=false)` after a make, i.e. the GPU's sequence) errors
  `error.SnapshotCacheMade`; without a share the store's budget makes the snapshots itself, once, and refuses a
  second make. Runs in `test-cache-capacity-host` (filter) and the native module's `test-cuda-host` run.
- `kv_pool_test.zig` "TF_KV_SHARE's slots (share_kept_ppm 0): the slots and the pool's pages never overcommit the
  room, cache first or not" — share on, `kept_ppm = 0`, 32 slots named against a 1 GiB cache:
  `pages × pageBytes + snaps × slot + fixed ≤ room`, the whole cache placed (`snaps × slot + kept == cache`),
  `snaps` clamped below the named 32, with and without `cache_first`.

| command | result |
|---|---|
| `zig fmt --check build.zig zig` | pass (no output) |
| `zig build test-kv-pool --summary all -j8` | 5/5 steps, **35/35 tests pass** (34 + the new plan test) |
| `zig build test-cache-capacity-host --summary all -j8` | 5/5 steps, **9/9 tests pass** (8 + the new keptBudget test) |
| `zig build test-kv-safety-host --summary all -j8` | 2/2 steps, **24/24 tests pass** (compiles `native/cuda.zig`) |
| `zig build test-glm-startup-host --summary all -j8` | 3/3 steps, **5/5 tests pass** |
| `zig build test-glm-pages-host --summary all -j8` | 4/4 steps, **13/13 tests pass** |
| `zig build test-glm-kv-pages-host --summary all -j8` | 4/4 steps, **30/30 tests pass** |
| `zig build test-glm-snapshots-host --summary all -j8` | 4/4 steps, **27/27 tests pass** |
| `zig build test-glm-tp3-host --summary all -j8` | 4/4 steps, **11/11 tests pass** |
| `zig build test-glm-fp8-kv-host --summary all -j8` | 4/4 steps succeed |
| `zig build test-glm-admission-host --summary all -j8` | 4/4 steps succeed |
| `zig build test-glm-page-contract-host --summary all -j8` | 4/4 steps succeed |
| `zig build test-glm-prompt-settings-host --summary all -j8` | 4/4 steps, **8/8 tests pass** |
| `zig build test-server-cpu --summary all -j8` | 5/5 steps succeed |
| `zig build test-cuda-host --summary all -j8` | **274/274 steps succeeded, all tests green** on this head |

No test was weakened, skipped or deleted. All targets are green on this head — the failures T03/T04 listed as
pre-existing (`test-glm-snapshots-host`'s four, `test-glm-prompts-host`'s one, `glm_fp8_pages_host`'s
`ExpectedHit`, `draft_sources_cuda_test`'s checkpoint-head case, `cuda_mm`'s logged-error gate) are fixed on this
branch's base already (T07's line, per NOTES.md), so nothing pre-existing remains to list. One mid-work state of
the new keptBudget test failed on its own counting arithmetic and was corrected before the commit; the stale
test binary it left in `.zig-cache` is unreferenced by the build graph.

## Left or uncertain

- The host-cache branch of the share path (TF_GLM_HOST_CACHE_GIB on, so `slots` stays null): `share_states(0)`
  makes no device snapshot pages there and the plan's cache is 0, so kept states lean on the host tier alone.
  Upstream has the same shape (`count`'s slots condition); the GPU run that failed had host mode off (its log
  line comes from the non-host branch). Flagged for the GPU gate, not changed here.
- `share()`'s exact-bytes allocation (`readySnapshots(snaps * slot)`) runs on the device path (`follow.Leader`
  machinery); on host it is pinned indirectly by the plan tests and `cacheText`'s naming test, not by a direct
  call-count test. A GPU test would need the engine.

## GPU checks needed (the pair, TP=2, `TF_GLM_KV_SHARD` unset)

- [ ] The exact launch of the 22:48 log starts: `TF_KV_SHARE=32`, `TF_KV_PAGES=1`, `TF_PROMPT_CACHE_FIRST=1`,
      `--prompt-cache-gib 7`, `--parallel 8`, `--context 1048576`, no `--learn-conversations` — no
      `SnapshotCacheMade`, no rank stop.
- [ ] The startup line names the slots ("N of TF_KV_SHARE=32 kept states' slots …"), and the
      "prompt cache: X GiB of device pages … a rank" line now comes after the plan and names at most the
      planned slots × slot.
- [ ] `free -g` on both boxes after start: `MemAvailable` keeps the reserve (compare with `TF_KV_SHARE`
      unset); the pool's pages plus the slots' bytes no longer overcommit by ~8 GiB a rank.
- [ ] The same start without `TF_PROMPT_CACHE_FIRST` (the plan holds either way).
- [ ] `TF_KV_SHARE=32` with `TF_GLM_HOST_CACHE_GIB` explicit (e.g. 4): starts, kept states land in the host
      tier, no second `snap_ready` (the host-cache branch flagged above).
- [ ] The share gates of T03/T04 (resumed = fresh, concurrent = alone, idle census `free + store == total`)
      still hold on this head.
