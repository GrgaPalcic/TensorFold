# T02 report: port upstream `40c03bf13` — `TF_KV_SHARE=n` native-CUDA adapter over the core pool

## What was done

Ported upstream `40c03bf13` (+337/−14, PR #1755, part of #1657) onto our pin (`f350cd3e` + the recipe's 112 engine
patches, after T01's `prompt_radix` lookahead + grid). Commit `f28cb7a` on `task/T02-share-adapter`.

- **`zig/src/core/kv_prompts.zig`** (new, 89 lines, from the patch verbatim): the hooks a family on the core pool
  calls — `Prompts` (begin/keep/report/free vtable), `Plan` (resume point + marks), `Blob` (a kept state's copied
  rows in store-held pages, `pagesFor`/`each`) — plus `sharing()`, which reads `TF_KV_SHARE` (unset/empty/0: off;
  a count: on; other text: `error.BadKvShare`, refused by name at start) and the blob-walk test.
- **`zig/src/core/kv_pool.zig`**: `Budget.kept_ppm` (parts per million of the cache made pages past the windows),
  `Plan.kept` (the cache bytes made pages), the `plan()` arithmetic upstream's hunks 2–3 add (kept carved out of the
  cache before the slots/copies split, added to the page cap), and `Pool.hold(out)` — `out.len` free pages the store
  holds itself, the pool's `reclaim` asked first, exactly as `promise` does.
- **`zig/src/core/kv_pool_test.zig`**: upstream's three new cases (the kept-share plan; the store holding free pages,
  never promised ones; a hold past the free pages asking the store to reclaim, via a `Shed` test double).
- **`zig/src/core/root.zig`**: `kv_prompts` exported and referenced from the core test block.
- **`zig/src/native/cuda_prompt_pages.zig`** (new, from the patch + the reference tree's final forms):
  `wanted`/`could`/`hint` and `Adapter.open(F, gpa, target, page_tokens, snap_limit)` — one `radix.Store` over the
  family's `kv.Pool` via `radix.pagesOf(kv.Pool, pool, pool.page_bytes)`, rules from `snaps.cacheRules(F)` plus
  `page`, `planned`/`grid` from `F.share_grid` (final form, from fb1094b16), `min_gap = 256`, `tail = true`; limits
  `pages = pool.total()`, `snaps = n`; the pool's `reclaim` evicts the tree's idle pages first; `F.share_attach`
  gets the hooks (and `null` on `close`, which also clears the pool's `reclaim` and deinits the store). Two host
  tests (the Fake family: a kept prompt's pages resume a later one, adopted, every page back on close; and the
  `hint` note a state past the unpaged budget names).
- **`zig/src/native/cuda.zig`**: `Host.prompts`; `prompt_pages.wanted(F, loaded)` after load, refusing
  `--learn-conversations` together with it; `kv_pages.plan(..., kept_ppm)` with `1_000_000` when sharing (so the
  cache's budget becomes pool pages for that family); the startup line's cache part becomes
  ", the prompt cache over the pool's pages" when sharing; the flat `prompt_cache.Store` is skipped when sharing
  (`and !share` on our `kept_budget > 0` condition, upstream's `cache > 0 and !ownCache(loaded) and !share`);
  `Adapter.open` beside the store's errdefer, closed in `Host.close`.
- **`zig/src/native/cuda_host_close.zig`**: `if (h.prompts) |x| x.close()` before the family frees the pool.
- **`zig/src/native/cuda_kv_pool.zig`**: `plan()` takes `kept_ppm: u32` (0 in host mode) beside our `Facts`;
  `kept()` adds `p.kept` to the no-`snap_slots` family's copied budget (the store's budget held as pool pages) —
  beside our `skipped` field, which the recipe's host-tier work added and stays as it was.
- **`zig/src/native/cuda_kv_safety_test.zig`**: the planner call passes `0`.
- **`zig/build/cuda.zig`**: the `prompt_pages` module import on the `tf-cuda-test` runner, as upstream wires it (the
  adapter's tests compile in the runner; on the host they run inside the native module's `test-cuda-host` run).

## Rejects resolved (12 hunks)

`git apply --reject` landed 17 of 29 hunks; 12 rejected and were resolved by hand against the reference tree
(`/home/user/invt/glm-kv-share/tf-at-share`, final forms):

- `zig/build/cuda.zig` (1): our runner lacks the `moss_image_cuda`/`spark25` lines the context anchored on; the
  import went in after `gemma4_cuda`.
- `zig/src/core/kv_pool.zig` (2): 0108 added `cache_first` to `Budget` and the recipe's tree already had `skipped`
  in `Plan` — `kept_ppm`/`kept` added beside them; the `plan()`/`hold()` hunks applied to our `plan` (recipe's
  `cache_first` behaviour untouched: `windows()` still reads `cache_first`, `held()` keeps its one-argument form).
- `zig/src/native/cuda.zig` (4): our tree has no `ownCache`/`TF_QWEN_KV_SHARE` (that commit is not in the pin), and
  the `Host`/`text()`/store conditions differ (recipe's `warm_quiet`, `kv_parts`, `kept_budget`); each was adapted
  to the equivalent line here, keeping our forms.
- `zig/src/native/cuda_kv_pool.zig` (4): our `plan()` takes `Facts` and reports refusals via `refusal()` (recipe
  0108's fit clause); `kept_ppm` added after `facts`, `.kept_ppm` into the `Budget` literal beside `cache_first`;
  the three test call sites gained `0`.
- `zig/src/native/cuda_kv_safety_test.zig` (1): call site gained `0` after `. {}`.

## What was adapted and why

- **`sharing()` reads the environment directly** (`std.c.getenv("TF_KV_SHARE")`): the task says to, as our pin has
  no `core/profile.zig` registry (later upstream work). This matches the patch's own text; the style follows the
  recipe's other switches (`cuda_kv_pool.cacheFirst` for `TF_PROMPT_CACHE_FIRST`, 0110's `Switches.read`).
- **`cache_first` (0108) kept exactly**: `Budget.cache_first`, `windows()`'s room carve-out, `cacheFirst()`, the
  256 MiB short-fall log line and the 0108 test all stand unchanged; upstream's fields sit beside them.
- **`skipped` (GLM host-tier share) kept**: `Plan.skipped` and `Kept.skipped` still carry the share the plan gave
  the pages; `kept()` now adds `p.kept` alongside it.
- **Final forms where they only extend this commit**: `could()`/`hint()` and the second adapter test (c8ee168d6),
  and `share_grid` rules (fb1094b16). Not brought in: C7's reclaim chain (`kv_reclaim.zig`, `joinReclaim`/
  `leaveReclaim`, `enter/leave` notify depth) and C2's groups (`group`, `shape`, `unit`, `ofRows`) — later work we
  are not taking, per the task.
- **Store skip condition**: upstream guards the flat store with `cache > 0 and !kv_pages.ownCache(loaded)`;
  our tree prices it with `kv_pages.storeBudget` (`kept_budget`, card + host tier) after the recipe's host-tier
  work, so the guard is `kept_budget > 0 and !share`. The `hint` plumbing (`s.over`) from c8ee168d6 was **not**
  ported: it depends on a `prompt_cache.Store.over` field and a `store(F, store, loaded, kept)` signature change
  that belong to that later commit, and the task bounds the port at `40c03bf13` + final forms of these two files.
  `hint()` itself is present and tested, ready for whoever ports that line.

## Off by default

No family declares `share_pool`/`share_attach` after this task (T03 adds GLM's), and `TF_KV_SHARE` is read only
through `prompt_pages.wanted`, which returns false without the `share_attach` decl. With the variable unset — or
set, for that matter — every existing path is unchanged: the adapter is never constructed, `kept_ppm` is 0, and
`plan()`'s arithmetic with `kept_ppm = 0` reduces to the previous expression (`kept = 0`; `most` gains a `+ 0`).

## Tests run (all pass counts from `--summary all`, Zig 0.17.0, host targets only)

| command | result |
|---|---|
| `zig fmt --check build.zig zig` | pass (no output) |
| `zig build test-kv-pool --summary all -j8` | 5/5 steps, 32/32 tests passed (was 26/26 + T01's; +3 kv_pool cases, +1 kv_prompts blob, +2 from T01 already counted) |
| `zig build test-kv-safety-host --summary all -j8` | 7/7 steps, 24/24 tests passed (compiles `native/cuda.zig` and `cuda_kv_pool.zig`; runs `cuda_kv_safety_test`) |
| `zig build check-native-cuda-host --summary all -j8` | success (the real native CUDA server config entry point compiles) |
| `zig build test-cuda-host --summary all -j8` | 270/274 steps succeeded (3 pre-existing failures), 4129/4140 tests passed (8 skipped, 2 failed) — baseline at the same commit: 270/274, 4123/4134 (8 skipped, 2 failed). The +6 are this task's new tests (3 kv_pool, 1 kv_prompts, 2 cuda_prompt_pages), all passing, inside the native module's two runs (441→445 and 111→113). |
| `zig build test-server-cpu --summary all -j8` (fresh cache, twice) | 893/893 tests passed both with and without the change |

The three `test-cuda-host` failures are **pre-existing on the base commit**, verified by `git stash` runs with
fresh `--cache-dir`s on both trees; the failing test names are byte-identical before and after:

- `draft_sources_cuda_test.test.draft sources: CUDA GLM distinguishes its checkpoint head from loaded DFlash2`
  (`draft_sources_check.zig:15`, expected true found false)
- `glm_fp8_pages_host.test.GLM FP8 pages: the prompt cache's store keeps paged E4M3 states past the card and
  resumes the newest by prompt` (`glm_fp8_pages_host.zig:760`, `ExpectedHit`); also fails standalone via
  `test-glm-fp8-pages-host` (14/15) on both trees
- `cuda_mm.test.TF_GLM_MM_STAGES: 3, 5, 6, 8 by index; unset, empty or anything else the default` (logged-errors
  gate: the test deliberately logs `[default] (err): TF_GLM_MM_STAGES: ...`, and the no-error-logs gate trips)

None touch the files this task changes (draft-source metadata, the FP8 pages host harness, and the MM-stages env
knob); no test was weakened, skipped or deleted.

## Left or uncertain

- `hint()` is wired into no log line yet (see the adaptation note above); the adapter, `wanted`/`could` and both
  its tests are complete.
- The adapter's second test asserts the `hint` strings only; a fuller exercise of `Blob.each` against real pages
  arrives with the families that declare the hooks (T03).
- Upstream's later `cuda_kv_pool.store` signature (`loaded` instead of `target`) is not taken, so `s.over` cannot
  be set here without bringing that commit in; noted for whoever ports the "state past the whole budget" line.

## GPU checks needed

None yet: no family declares `share_pool`/`share_attach` until T03, so no GPU path can reach the adapter. Once a
family does (T03), the DGX checks to run are upstream's for #1755: a kept prompt's pages resuming a later prompt
(resumed == fresh), eviction under a promise (reclaim from the store's idle pages), and close giving every page
back (`pool.idle()` at shutdown), plus the flat-store-off startup line on a real load.
