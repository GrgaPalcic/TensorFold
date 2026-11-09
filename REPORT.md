# T06: a family's host staging permit for kept states on disk (upstream PR #2337)

## What was done

Ported upstream `0e5085dca` (+~85) and the review commit `12dfa588a` onto the recipe pin
(`f350cd3e` + 112 engine patches), then wired GLM's own attach path to the new permit:

- `zig/src/native/cuda_disk.zig` — upstream's two commits, mechanically identical in shape:
  - `open()` takes `staging: ?u64` and sets `Options.staging_bytes` only when a family names
    one (review commit: 1 GiB lives only in `rows_store.Options`' default).
  - `stagingOf(K, target) ?u64` reads an optional `K.staging(target) u64` declaration;
    `attachKept` passes it through. A module that declares `attach` opens the store itself
    and its `staging` stays unread (upstream's comment kept).
  - Nemotron's `attach` passes `null`: no declaration, store default — byte-identical
    behaviour for Nemotron/Qwen/Flash Next.
- `zig/src/native/cuda_glm_disk.zig` — GLM attaches the store itself (recipe patch 0109),
  so upstream's `stagingOf` is never read there. `attach()` now reads
  **`TF_GLM_LEARN_STAGING_GIB`**: unset or empty keeps the store's default (1 GiB); a
  number gives that many GiB aligned down to whole 16 KiB rows pages; anything else is
  refused at start (`error.BadStagingGib`). Passed as `staging_bytes` in the `Options`
  literal. The startup log line adds `; staging N GiB` only when the switch is named, so
  unset keeps today's line byte-for-byte.
- `zig/src/families/glm/cuda_native.zig` — `error.BadStagingGib` added to the explain
  table, named like the other `TF_GLM_*` refusals, plus a test that the refusal names the
  switch.

## Adaptations and why

- **The expected reject at cuda_disk.zig ~line 159**: upstream's `attach()` calls
  `identity(..., o.model_type)` (a later upstream change we do not carry); ours calls
  `identity(..., loaded.backend, loaded.target)` per the recipe pin. Resolved by keeping
  our call shape and adding only the `, null` staging argument.
- Upstream's generic-path hunks reference `rows.attach` / `rows.Options` directly
  (review commit); our file imports `api.prompt_cache.rows`, so the same spelling works
  as-is.
- GLM's parse (`stagingOf` in cuda_glm_disk.zig) follows the house style of
  `cuda_snapshot_host.zig`'s `asked()` (parseFloat, finite, 0..4096 bound), but aligns to
  `rows.file.alignment` (16 KiB) instead of 2 MiB device pages, and empty/unset returns
  the store default rather than an "auto" sentinel. The default is read from
  `rows.Options`' field, not restated, so the two stay in step.
- Where upstream's review test builds a store through `attachKept` with a recording
  driver, our copy keeps the upstream shape verbatim (the identity path is ours but the
  fake driver vtable is the same surface).

## Tests run (host, Zig 0.17.0, `-j8`)

| target | result |
| --- | --- |
| `zig build test-cuda-disk-open-host --summary all` | 9/9 pass (includes the ported `staging permit` test and the `--spill-gib` test extended with `staging_bytes`) |
| `zig build test-cuda-host --summary all` | 588/591 pass, 1 skipped, 2 failed — **pre-existing**, see below |
| `zig build test-glm-startup-host --summary all` | success |
| `zig build test-rows-host --summary all` | success |
| `zig build test-inc4-rows-host --summary all` | 4/4 pass |
| `zig build test-window-rows-host --summary all` | success |
| `zig build test-rows-prefetch-host --summary all` | success |
| `zig build test-learn-recovery --summary all` | success |
| `zig build test-learn-warning --summary all` | success |
| `zig build test-nemotron-learning-host --summary all` | 10/10 pass |
| `zig build test-proposal-rows-host --summary all` | 5/5 pass |
| `zig build test-qwen27-prompt-rows-host --summary all` | 8/8 pass |
| `zig fmt --check build.zig zig` | clean |

**Pre-existing failures (not from this port):** `test-cuda-host` fails the same way with
the working-tree changes stashed and the tree at `390b692`:
`draft_sources_cuda_test "draft sources: CUDA GLM distinguishes its checkpoint head from
loaded DFlash2"` (expected true, found false) and `glm_fp8_pages_host "...keeps paged E4M3
states past the card..."` (`ExpectedHit` at glm_fp8_pages_host.zig:760); `cuda_mm`'s
`TF_GLM_MM_STAGES` test logs an expected refusal error but its assertions pass. None
touch the disk path; left for whoever owns those areas.

New tests added (all pass): attachKept passes a declared 3 GiB permit and else the store's
default (upstream review commit's test, on a recording driver); `TF_GLM_LEARN_STAGING_GIB`
parse — unset/empty/whitespace → store default, `"2"` → 2 GiB, `"12"` → 12 GiB,
`"4.0001"` → 4 GiB (whole pages), `"0"`/`"0.000001"` → 0 (off), `"x"`, `"-1"`, `"nan"`,
`"inf"`, `"5000"` refused; the refusal names the switch in `explain`.

## Left or uncertain

- `attachKept`'s `stagingOf` is compile-time per family, so a family that later declares
  `staging` gets it with no further wiring — nothing else to port from the PR.
- The 4096 GiB upper bound on `TF_GLM_LEARN_STAGING_GIB` matches `TF_GLM_HOST_CACHE_GIB`'s
  bound; upstream does not bound family declarations. Harmless: it only bounds the setting.
- Did not touch `rows_store.zig` — the PR does not, and our pin's `Options.staging_bytes`
  default already exists (1 GiB).

## GPU checks needed later (DGX)

1. `TF_GLM_LEARN_STAGING_GIB=2` + `--learn-conversations`: a ~126k-token prompt sent cold
   is written to disk (today: `StagingFull` skip); watch `MemAvailable` on rank 0 during
   the save.
2. Switch unset: startup log lines byte-identical to today (no `; staging …` suffix), and
   a >1 GiB save is still skipped as `StagingFull`.
3. A bad value (`TF_GLM_LEARN_STAGING_GIB=x`) refuses at start naming the switch.
