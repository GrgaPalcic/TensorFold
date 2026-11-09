# T04 report: the startup plan counts `TF_KV_SHARE`'s kept states, a slot each beside the pool's pages

## What was done

Ported upstream `389769733` (PR for #2307, +141/−42) onto our pin (`f350cd3e` + the recipe's 112 engine patches,
after T01–T03). With `TF_KV_SHARE=n`, the startup plan now takes `min(n, cache / slot)` slots of the prompt cache
for the kept states' parts past the pages (a slot = the state at the window's last mark, `snap_bytes(window)` =
`pages_mod.shareSlot`), and the rest of the cache goes to the pool's pages (`Budget.slots_only`, `Plan.kept`).
A value over 64 (`kv_pool.most_snaps`) is refused by name before any rank opens (`error.GlmShareTooMany`, words in
`explain()`); a cache too small for one slot keeps none. The startup line names the slots, and when fewer than n
fit, how many of the n and the `--prompt-cache-gib` that holds all n (`kv_pages.cacheText`); the warning under 4
states a stream names `TF_KV_SHARE`, the count kept and the `--prompt-cache-gib` their slots take
(`share_mod.advice` replacing `fewStates`). The doc hunk was skipped per the task. One commit on
`task/T04-startup-plan` (`f867c22`), `REPORT.md` second.

- **`core/kv_pool.zig`**: `holdingGib(states, slot)`; `Budget.slots_only`; `plan()`'s `kept` becomes
  `cache - split(cache, l.snap_bytes, b.snaps).snaps * l.snap_bytes` when `slots_only`.
- **`core/kv_pool_test.zig`**: upstream's slots-alone test (its local `const mib` dropped — the file declares
  `mib` at the top), plus the task's `cache_first = true` + slots case (below).
- **`families/glm/cuda_native.zig`**: `attach()` no longer readies snapshots; the new family decl
  `share_states(p, snaps, parallel)` readies the plan's `snaps` slots and warns through `advice`. `finishRank`
  declares the slot (`e.share_pages = true`, `l.snap_bytes = pages_mod.shareSlot(own.e)`). `open()` refuses
  `share_mod.tooMany(share_snaps)` by name before `startup.open`. Explain words and its test.
- **`families/glm/cuda_pages.zig`**: `shareSlot(e)` (upstream's, on our `snap_grid`).
- **`families/glm/cuda_share.zig`**: `fewStates` → `advice(states, parallel, slot)` + `tooMany`, with upstream's
  two tests.
- **`native/cuda.zig`**: the inlined counting (our pin has no `native/cuda_streams.zig`) computes `slots`
  (share and `l.snap_bytes > 0` and not host mode → `kv_prompts.sharing()`) and passes it to `kv_pages.plan`;
  the startup line goes through `cacheText`; the store opens over what its slots hold (`share_limit`); the
  capacity warning is skipped for share (no copied rows).
- **`native/cuda_kv_pool.zig`**: `plan()` takes `slots: ?usize` (`.snaps = slots orelse named(o)`,
  `.slots_only = slots != null`); `attach()` calls `F.share_states` after the family's `attach`; `cacheText` +
  its test; every `plan()` call site (18) gains `, null`.
- **`native/cuda_kv_safety_test.zig`**: the planner call gains `, null`.

## Tests run (Zig 0.17.0, host targets, `-j8`)

| command | result |
|---|---|
| `zig build test-kv-pool --summary all` | 5/5 steps, **34/34 tests pass** |
| `zig build test-glm-startup-host --summary all` | 3/3 steps, **5/5 tests pass** |
| `zig build test-glm-pages-host --summary all` | 4/4 steps, **13/13 tests pass** |
| `zig build test-cuda-host --summary all` | 270/274 steps; **474/476 tests pass on the re-run steps (2 failed, 1 skipped)** — see below |
| `zig fmt --check build.zig zig` | **pass** |

`test-cuda-host` includes `cuda_kv_safety_test` and the full GLM family module run (the advice/tooMany,
`cacheText`, `share_states` and refusal tests all execute there). The 2 failing tests and the `cuda_mm` error log
are **pre-existing on this branch's base**: I confirmed on a clean checkout of `c12c858` (fresh worktree) that
exactly the same three fail there, the same way:

- `glm_fp8_pages_host` "the prompt cache's store keeps paged E4M3 states past the card and resumes the newest by
  prompt" — `error.ExpectedHit` at `zig/glm_fp8_pages_host.zig:760` (1/15; 14/15 pass).
- `draft_sources_cuda_test` "CUDA GLM distinguishes its checkpoint head from loaded DFlash2" (1/116).
- `cuda_mm` "TF_GLM_MM_STAGES: 3, 5, 6, 8 by index…" logs one error (step fails, its tests pass).

`NOTES.md` lists the first and third as pre-existing at the pin (recipe patches 0094/0095) and says they were
fixed on T07's head; this branch (T03's line) still carries them, unchanged by T04. Not touched, per the rules.

## What was adapted and why

- **No `native/cuda_streams.zig` at this pin** (a later upstream refactor; T03 already inlined `opensStore`).
  `Count.slots`, `keptPpm()` and `opensStore()` were not ported as helpers: the slot decision runs where the
  counting is inlined in `native/cuda.zig` (`if (share and l.snap_bytes > 0 and !host_cache) slots =
  kv_prompts.sharing()` — upstream `count()`'s condition verbatim), and the store gate keeps T03's inlined form
  with upstream's new semantics: `share and (if (comptime F.share_kept_ppm == 0) share_limit > 0 else cache > 0)`,
  `share_limit` = the plan's `snaps` when slots were planned, else `kv_prompts.sharing()`. Upstream's
  `cuda_streams.zig` test is not ported (no such file here).
- **`Budget.asked` → recipe 0108's `cache_first`**: our Budget ends in `cache_first` (0108), not upstream's
  `asked`, and `held()` takes no `more` argument (no `Plan.taken` here — 0108's form). `slots_only` was added
  beside 0108's field and only the `kept` line changed; with `slots_only` unset every path is byte-identical.
- **`cuda_kv_pool.plan` keeps our `Facts` parameter** (T06's reserve/`min_state` work) and the retry for a device
  share under one state; `slots` went in as a new last parameter.
- **The host tier differs**: upstream's `share_states` reads `snapshots.pages.onHost()` (`TF_GLM_HOST_CACHE=1`:
  kept states entirely in host memory, outside the device plan, so it readies all n states whatever the plan
  holds). Our pin's 0109 host tier is `TF_GLM_HOST_CACHE_GIB` — two tiers, device pages first, then pinned host
  pages — and it is on by default (auto), so "states in host memory" is never all-or-nothing here. The branch is
  therefore not ported: `n` is always the plan's `snaps`; states past the slots spill to the host tier as they
  are kept, as before. Upstream's warning text is kept (it names the count the cache's slots hold).
- **The task's `cache_first` + slots test** (`kv_pool_test.zig`): the same small numbers as upstream's test
  (180 MiB fixed, 13,056 a token, 512 a page, 180 MiB a slot, 80 GiB room) with `lanes = 8` and
  `cache_first = true` + `slots_only = true`: five of the sixteen named slots, the rest pages; with
  `slots_only` unset the plan keeps today's shape (same snaps, the rest copied rows, `kept == 0` — asserted) and
  a 100 MiB cache (under one slot) keeps none.

## Off by default

`TF_KV_SHARE` unset (or 0): `kp.sharing()` is 0, `share` is false, `slots` stays null → `kv_pages.plan` gets
`slots = null` (`slots_only = false`, `snaps = named(o)`: the exact old Budget), `cacheText` returns the old
strings, `share_limit` is 0 so the store never opens, `share_states` returns before touching anything, and
`e.share_pages`/`l.snap_bytes` stay as before (finishRank's block is under `own.share > 0`). With the variable
set: `tooMany` refuses past 64 by name before any rank opens; the plan counts only the slots that fit and the
startup line says so.

## Left / uncertain

- The 0109 host-tier divergence above is a deliberate keep-the-recipe choice; upstream's `onHost()` semantics
  (all states in host, none planned) cannot arise under `TF_GLM_HOST_CACHE_GIB`'s two-tier design. If the recipe
  later adopts upstream's boolean switch, `share_states` should grow the branch back.
- The three pre-existing `test-cuda-host` failures listed above (fail identically on the clean base `c12c858`).

## GPU checks needed later (on the DGX boxes)

- The startup line at this lane's settings (20.2 GiB room, 8 lanes, 417 MiB fixed, 1.73 MiB a page) with
  `TF_KV_SHARE=16` and `=32`: the slots, pages and cache it names.
- `MemAvailable` after load on both ranks.
