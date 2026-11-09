# T03 report: GLM keeps prompts as the KV pool's pages in the shared store (`TF_KV_SHARE=n`), unsharded first

## What was done

Ported upstream `a6f5615d4` (PR #1732: GLM's own store over `prompt_radix`), `768170532` (TP: the begin step
names the cache's pages, the snapshot step their absence) and `d5b0b92f4` (PR #2211: GLM moves onto the T02
adapter, `cuda_share.zig` arrives) onto our pin (`f350cd3e` + the recipe's 112 engine patches, after T01's
radix work and T02's adapter). Ported the **end state** — `cuda_share.zig` and the shared-pages side of a GLM
pass — not the intermediate `cuda_prefix.zig` store. Three commits on `task/T03-glm-share`:

- `a84922c` — the layout/snapshot change (upstream a6f5615d4, d5b0b92f4).
- `aa2816a` — the lanes/pages/follow change (upstream a6f5615d4, 768170532, d5b0b92f4).
- `d8c9625` — `cuda_share.zig` + the declarations + the refusals (upstream d5b0b92f4).

- **`zig/src/families/glm/cuda_share.zig`** (new, upstream d5b0b92f4's file verbatim): `cacheKey` (an image's
  content key, else the token ids), `cacheable`, `advice`/`fewStates`, `moved` (`TF_GLM_KV_SHARE` refused),
  `of`, `Owner` (a resume waits for the pass's start), `begin` (the deepest kept state's pages into `adopt`),
  `Keeps`/`marks` (keeps at the store's marks, no table swaps).
- **`cuda_native.zig`**: declares `share_pool` (rank 0's pool), `share_attach`, `share_kept_ppm = 0` (rows past
  the pages stay each rank's `snap_*` copies in the cache's room) and `share_grid = prompt_rows` (**4,096**
  here; upstream's was 2,048). `attach()` sizes the snapshots at the window's last mark (cc-defer, the recipe's
  `readySnapshots`, so the 0109 host tier still takes its share) and warns under 4 states a stream. Refusals by
  name: `TF_GLM_KV_SHARE` moved to `TF_KV_SHARE`; `TF_KV_SHARE` without `TF_KV_PAGES`
  (`error.GlmShareNeedsPages`); `TF_KV_SHARE` with `TF_GLM_KV_SHARD=1` (`error.KvShareWithShard`, T05 lifts
  it), each with words in `explain()`.
- **`cuda_pages.zig`**: `openWith(slot, prompt, max_new, home, adopted)` — 0110's sharded branch byte-for-byte,
  the unsharded branch adopts the store's pages at the head of the lease (`pool.adopt`, the rest promised and
  covered, their tables uploaded first); `give()` returns them when no lease took them; `Pages.prompts` (the
  server's hooks); `map`/`mapWith` take the begin step's `kept` count and `mapped()` poisons only past it
  (`TF_KV_POISON` on a follower leaves the store's pages alone); the idle line admits the store's pages when
  `e.share_pages`.
- **`cuda_lanes.zig`**: `opening` begins through `share_mod` (the store's pages into the lease, the resume
  pending in the owner, restored by the prompt pass's start through the same `Resume` on every rank), `Lane.marks`
  (owned, freed with the lane), `seat`/`pagesFor` hand the adopted pages over and give them back on a refusal,
  `cached()` asks the attached store, `marksOf` keeps at the store's marks. 0110's home-part seating
  (`homeOf`), `resumeFrom`'s shard checks and `resumesHere` stay.
- **`cuda_follow.zig`**: the begin step carries the store's kept count (`Header.a`), the follower maps with it;
  the snapshots step carries `e.share_pages` in its flags. 0110's twin ids in `depth` and 0109's host-tier
  `agreeTiers` stay.
- **`cuda_snapshot_layout.zig`**: `Part.from` and `Plan.shared` — a state whose leading pages the store holds
  plans only past them (`initForShared`, `sizeForShared`; `initFor`/`sizeFor` keep 0110's owner-only latent
  flag); `piece()` reads from `part.from`. A resumed pass writes nothing below its mark.
- **`cuda_snapshots.zig`**: `heldPages()`; `keepAs` plans past the store's pages and records `Snap.shared`
  (`cuda_engine_types.zig`); `restore` replans with the state's own count. 0110's owner-only latent rows,
  carries, and 0109's tier agreement stay.
- **`native/cuda.zig`**: a family keeping its own copies (`share_kept_ppm = 0`) opens the store at any cache
  (upstream's `opensStore`, inlined — our pin has no `native/cuda_streams.zig`); `--learn-conversations` stays
  refused beside `TF_KV_SHARE` (T02's words, unchanged).
- **Build glue**: the GLM family module imports `engine_api` (the shared store's types, one module one set of
  types, as upstream), and the host glue takes it from the family's import table. The image-cache host step is
  **not** ported (our tree has no `glm_image_cache_host.zig`; that file belongs to upstream work outside these
  three commits).

## Rejects resolved

The three patches measured 29 + 9 + 25 rejected hunks, in exactly the files recipe patches 0110 and 0109
rewrote. All were resolved by hand against the reference tree (`/home/user/invt/glm-kv-share/tf-at-share`,
final files) with 0110's sharded branches kept beside upstream's shared-pages branch:

- `cuda_lanes.zig` (12): `opening`/`seat`/`pagesFor` merged (upstream's `adopted` beside 0110's `home`);
  `marksOf` reworked to `share_mod.marks` (upstream's final form — a6f's `prefix_mod.Keeps` form was never
  applied here); `cached()` via `share_mod.of`; the a6f-era `Lanes.prefix` field and `cuda_prefix.zig` store
  are **not** ported (d5b0b92f4 deletes them; we go straight to the adapter).
- `cuda_pages.zig` (6): `openWith` (home + adopted), `map`/`mapWith` (twin + kept), `mapped` (poison past
  kept), `prompts` field, the idle line.
- `cuda_snapshots.zig` (4) and `cuda_snapshot_layout.zig` (2): shared-pages plans beside `latent`.
- `cuda_native.zig` (7): `share()` at attach (upstream's, on our `readySnapshots`), the moved/shard refusals,
  the explain table, `prefix_mod.sharing()` → `kp.sharing()` + the moved refusal.
- `cuda_follow.zig` (2): begin's `kept` in `Header.a`, snapshots' flag.
- `cuda_engine.zig`/`cuda_engine_types.zig` (2): `share_pages`, `Snap.shared`.
- Build glue (4): `glmCuda` gains the `api` parameter (three call sites + the import), the host glue's `api`
  moves up beside the image-cache step (which we drop), `cuda.zig`'s `opensStore` inlined.
- Tests (9): every 0110 call site gains the new arguments (`&.{}`, `0`, `null`); `cuda_disk_image_test.zig`'s
  Fake gains `share_pages = false`; the TF_KV_SHARE pages test re-sized to our 4,096 grid (below).

## What was adapted and why

- **Grid 4,096, not upstream's 2,048**: `share_grid = prompt_rows = 4,096` per the task. `cuda_pages_test.zig`'s
  store test was re-sized accordingly: a 4,100-token prompt (mark at 4,096, 8 pages below it), a 12-page pool,
  the Fake's capacity raised for that one test (16,384) so the prompt past the mark fits the table. Every
  assertion keeps upstream's shape (cold marks, resume on the tree's pages, the miss at the mark, eviction).
- **`glm_snapshot_host.zig`'s rules test**: asserts `grid == 4096` and `warm == .waits` — our GLM declares the
  recipe's `warm_quiet`, upstream's does not (upstream asserted 2,048 and `.off`).
- **`snap_restore`** keeps the recipe's `RestoredInPromptPass` refusal unless the store is attached (upstream's
  d5b0b92f4 form, which we ported directly).
- **The snapshot slot arithmetic** is d5b0b92f4's (`attach` → `readySnapshots(own.share × stateBytes(...))`),
  not #2307's slot-per-state plan (`389769733`, `share_states`, `pages_mod.shareSlot`): the task names three
  commits and the reference tree's tip carries that later work; T05/T07 territory.
- **`opensStore` inlined** into `native/cuda.zig` (`share and (cache > 0 or F.share_kept_ppm == 0)`): the
  helper lives in upstream's `native/cuda_streams.zig`, a file our pin does not have (a later refactor).
- **No `cuda_prefix.zig`**: `a6f5615d4` creates it and `d5b0b92f4` deletes it; we ported the end state, so the
  file never landed and `cuda.zig` exports `share` directly.

## Off by default

With `TF_KV_SHARE` unset: `kp.sharing()` is 0, the store is never attached (`prompt_pages.wanted` false → no
`Adapter.open`), `own.share = 0` skips `share()`, `e.share_pages` stays false, `heldPages` is 0 (so every
snapshot plan is byte-identical), `adopt` is always empty (`openWith` reduces to the old promise/cover),
`beginOn`'s `kept` is 0 and the follower's poison skips nothing. With the variable set but the shard on, the
start refuses by name.

## Tests run (Zig 0.17.0 host targets, `--summary all -j8`)

| command | result |
|---|---|
| `zig fmt --check build.zig zig` | pass (no output) |
| `zig build test-kv-pool --summary all -j8` | 5/5 steps, 32/32 tests passed |
| `zig build test-kv-safety-host --summary all -j8` | 7/7 steps, 24/24 tests passed (compiles `native/cuda.zig`) |
| `zig build test-glm-pages-host --summary all -j8` | 4/4 steps, 13/13 tests passed |
| `zig build test-glm-kv-pages-host --summary all -j8` | 4/4 steps, 30/30 tests passed (sharded page tests, the adopted-lease cases, the E4M3 layouts) |
| `zig build test-glm-snapshots-host --summary all -j8` | 23/27 passed; the 4 failures are pre-existing at the base commit (verified by `git stash` runs, byte-identical names and messages): "a short prefix fits 4 GiB", "actual store charges held plus spare pages", "0.3 GiB fits the first default TP2 prefix", "the actual store's budget spans the host tier" |
| `zig build test-glm-startup-host --summary all -j8` | 3/3 steps succeeded |
| `zig build test-glm-tp3-host --summary all -j8` | 4/4 steps, 11/11 tests passed |
| `zig build test-glm-prompts-host --summary all -j8` | 6/7 passed; the 1 failure ("context window cut preserves sequence positions") is pre-existing at the base commit |
| `zig build test-glm-fp8-kv-host --summary all -j8` | 4/4 steps, 21/21 tests passed |
| `zig build test-glm-admission-host --summary all -j8` | 4/4 steps, 2/2 tests passed |
| `zig build test-glm-page-contract-host --summary all -j8` | 4/4 steps, 2/2 tests passed |
| `zig build test-glm-prompt-settings-host --summary all -j8` | 4/4 steps, 8/8 tests passed |
| `zig build test-server-cpu --summary all -j8` (fresh cache) | 5/5 steps, 893/893 tests passed |
| `zig build test-cuda-host --summary all -j8` | 270/274 steps succeeded, 469/471 tests passed — the 3 failures are the **same pre-existing three** as T02's baseline (`draft_sources_cuda_test`, `glm_fp8_pages_host`'s "resumes the newest by prompt", `cuda_mm`'s `TF_GLM_MM_STAGES` logged-errors gate); this task's new tests all pass, including the glm module's 233-test run with the ported TF_KV_SHARE cases |

0110's and 0109's tests still pass: the sharded page/keep/disk tests run inside the glm module's `test-cuda-host`
run (`cuda_kv_shard_pages_test.zig`, `cuda_kv_shard_keep_test.zig`, `cuda_disk_shard_test.zig`,
`cuda_ranks_test.zig`, `cuda_kv_shard_attn_test.zig` — all green), and `test-glm-kv-pages-host` covers the
sharded page cases standalone. No test was weakened, skipped or deleted; the two re-sized cases above assert
the same properties at our grid.

## Left or uncertain

- `share_states`/`advice` (#2307's slot plan, `pages_mod.shareSlot`) is **not** ported — the task bounds the
  port at the three named commits; the d5b0b92f4-era `share()` sizes the snapshots from `--prompt-cache-gib`'s
  room via `readySnapshots`. When a later task takes #2307, `cuda_share.advice` is already there (ported with
  the final `cuda_share.zig`).
- `fewStates` (d5b0b92f4's warning) stays; #2307 replaced it with `advice` upstream. Both are in
  `cuda_share.zig` verbatim; `share()` uses `fewStates`.
- The image-cache host test (`glm_image_cache_host.zig`) upstream extends is absent here; the glue lines that
  would wire it are left out, and the step's description lives only in upstream's tree.
- `TF_GLM_KV_SHARE` is refused before any rank opens (`open()`), naming `TF_KV_SHARE`; the profile registry
  (`docs/profiles.md`) does not exist on our pin, so that doc hunk was not ported.

## GPU checks needed (the pair, TP=2, `TF_GLM_KV_SHARD=0`)

Upstream's gates for #1732/#2211 plus the recipe's, run with `tools/share_gate.py` where it applies:

- [ ] Outputs equal with `TF_KV_SHARE` unset (the recipe's gate: byte-identical replies and startup lines).
- [ ] Resumed equal to fresh at marks 4,096, 126,976 and a 600k state (`share_gate.py` resumed = fresh).
- [ ] Concurrent equal to alone at 1, 2, 4, 8 (`share_gate.py` concurrent = alone).
- [ ] Drafted equal to `"draft": false` (`share_gate.py` drafted = plain).
- [ ] `kv pool idle` census the same first and last: `free + store == total` (rank 0's line now names the
      store's pages; `TF_GLM_KV_SHARD` off, so one part).
- [ ] `TF_KV_POISON=1` on both ranks gives the same hashes (the store's pages untouched on a follower).
- [ ] A kept prompt's pages resuming a later prompt: the startup line names the store, and a second server
      start with `TF_GLM_KV_SHARE=32` refuses naming `TF_KV_SHARE`.
- [ ] `--learn-conversations` with `TF_KV_SHARE` set refuses at start by name.
- [ ] `TF_KV_SHARE` with `TF_GLM_KV_SHARD=1` refuses at start (`KvShareWithShard`).
