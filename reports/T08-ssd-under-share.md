# T08: the SSD tier under shared pages — `--learn-conversations` together with `TF_KV_SHARE` (finalised 2026-11-10)

**Branch:** `task/T08-ssd-under-share`, on top of `share-port` `b71a686` (T05's report commit).
**Nature:** new design work — upstream refuses the combination (`native/cuda.zig`: "serve with one of
the two") because its disk image holds a snapshot's bytes while a shared state leaves its pages out, so
there is no upstream sha to port. Everything here follows the task's finalised brief against the merged
tree (T03/T04b/T05/T06 reports, recipe patches 0108/0109/0110).

## What was done

Pieces 1–4 of the task are in; piece 5 (the sharded case) is **refused by name**, per the task's escape
hatch. The land is three code commits plus this report:

1. `core rows: ...` — the `rows.Kept` view, the image carrying pages, write-behind in pieces under the
   staging permit, recall into pool pages (pieces 1–3, core side, every store's).
2. `glm cuda: ...` — GLM's pages layout beside the copy one, writes and recalls through the page holds
   (pieces 1–3, GLM side).
3. `native cuda: ...` — the refusal lift, the startup line, and the sharded pair refused by name
   (pieces 4–5).

**Why three commits and not one per piece:** the pieces share one spine end-to-end — the `Kept` view is
produced by the stores (piece 2), consumed by the writer (piece 2), and its image layout is what recall
reads back (pieces 1, 3–4); splitting by piece would mean hunk-surgery across ~27 files with no
intermediate state that compiles. The commits cut along the seams that do compile (core vs GLM family vs
the refusal), cite their pieces, and every target below ran on the final head.

### Piece 1 — the image carries the pages

- `rows.Image.gather`/`scatter` take `pages: []const u32` — the state's pool pages in token order.
  Empty means a copy holds every row: the flat stores' exact old shape, byte-identical calls and files.
- GLM's disk image grows a pages mode beside the copy one: `Words` carries `pages` and formats
  `pages_fmt=4` / `pages_sharded=5` (`families/glm/cuda_disk_image.zig`); a state's per-token row parts
  are addressed through its page ids (`pageParts`/`pageSpan` build the snapshot plan with the pages in),
  while the words and the fixed parts stay the slot's own. `cuda_disk_ranks.wordsOf` marks
  `pages = s.shared > 0` (the store's kept-state copy count *is* the page count under share);
  `cuda_snapshots.seal` sizes the slot past those pages via `layout.Plan.sizeForShared`.
- Rank 1's part under 0110's `disk_ranks` step is untouched: unsharded writes every rank's bytes through
  rank 0's files exactly as a copied state does.
- Old files read unchanged: formats 0–3 are untouched, `pages` defaults false, so a store written before
  this task still restores byte for byte.

### Piece 2 — write-behind on keep, the pages held, in pieces under the permit

- `rows.Kept` (`rows_store.zig`) is the view both stores hand the writer: `owner`, `tokens`, `spans`,
  `at`, `saved`, `pages`, and a `freed` hook. `prompt_cache.Store.keepBorn` queues `rows.Kept.of(e)` —
  built from its `Entry`, nothing else there moves, and only when a disk store is attached
  (`--learn-conversations`), so the flat path is off-by-default byte-identical.
- A paged store's save holds each of the node's pages once more while queued or running
  (`prompt_radix.Store.writeBehind` retains through the pool, `freed` releases on staged/ended/dropped),
  so the tree may drop the leaf but the pool cannot reuse the pages under the writer. `rows_write`'s
  `dropped`/`evicted`/`gave` match by `owner` (`*anyopaque` now, not `*pc.Entry`), and
  `Rows.deinit`/`forgetAll` give every queued save's holds back.
- An eviction with the save still queued stages it now: `Tree.on_evict` fires before the entry drops and
  `rows.write.evicting` writes that save immediately (its pages are held to the end).
- Piece staging (`rows_write.stageJob`): the whole state as one job when the staging permit holds it
  (`whole` checks `Worker.fits` *before* gathering, so nothing over the permit is ever staged);
  otherwise the missing 256-token blocks a piece at a time on block edges, the state's own file and
  metadata last — its publish makes the entry. `io.Job` grew `final`/`new_blocks`: non-final pieces
  publish nothing and name no block-group completion; `Worker.names` matches final jobs only.
- Blocks stay 256 tokens and are named by the kept tokens with the store's keying, so states sharing a
  prefix share block files on disk, as today.

### Piece 3 — recall on miss into pool pages

- `rows.read.readPages` blanks a state into pool pages the store holds (`Image.blank_pages`:
  `at/page_tokens` ids, each held once more for the caller) and scatters the file into those pages and
  the slot.
- `prompt_radix.Store.begin` on a miss (or when a disk state is longer than the tree's deepest match)
  picks the longest state on disk that pays (`select.savedSeconds` against `prefill_rate`), evicts to fit
  the limits, reads it into pages, `Tree.insert`s it and adopts the pages like any kept state's; the
  recall's own holds go once the tree holds them.
- A sharded pool can hold a recall's pages in one part: `pc.Pages.VTable.hold` (null for pools that
  cannot), `kv_prompts.Parts.holdPages` (global ids, the part with room, as `Parts.choose` does for a
  fresh stream), wired in `cuda_native.share_pool` with the `part × h` offset. Unused until piece 5,
  but the surface is in and typed.

### Piece 4 — identity word, refusal lift, startup line

- The layout folds into the store's identity: `cuda_glm_disk.attachShare` hashes the share facts
  (world, shape, drafter, **1 = the pages layout**) plus every rank's prefix into
  `cuda_disk.identityOf`, so the pages layout's store is a new directory under the same root and a
  restart with the switch flipped starts a new set — old and new never mix.
- `native/cuda.zig` lifts the refusal: `shareDisk(F, loaded)` allows `--learn-conversations` beside
  `TF_KV_SHARE` for GLM only and only unsharded; the refusal text names the lift ("GLM-5.3-Flash
  unsharded keeps both: its store's states on disk carry their pages"), so a sharded start is refused
  by name, not silently.
- `attachShare` opens `rows.Rows` under the learn root with the T06 staging permit
  (`TF_GLM_LEARN_STAGING_GIB`), attaches it to the paged store (`store.attachDisk`), and logs the
  startup line: `glm: conversation states on disk in <dir> (...; 1 ranks' parts, pages and parts)` —
  the "conversation states on disk ... (pages and parts)" the task asks for.

### Piece 5 — the sharded case: refused by name, what it needs

`shareDisk` returns false when `loaded.target.settings.kv_shard.on`, so share + shard + learn keeps
today's refusal (with the lift named). What a real piece 5 needs, in build order:

1. **Per-rank parts in the pages layout.** `Words.pages_sharded=5` is defined but nothing writes it.
   The shape is 0110's `disk_ranks` step reused with pages as the source: the owner rank's latent rows
   read from its pages, every rank's own parts from its lease, all written through rank 0's files. The
   unsharded `gatherPages` already reads a piece through pages; the sharded writer needs the same split
   `disk_ranks` applies to a copied state.
2. **Recall into one part.** `blank_pages` for a sharded pool must take the recall's `at/page_tokens`
   pages in the part with room (`Parts.choose`'s rule) as global ids — `Parts.holdPages` exists for
   exactly this — and the read scatters into that part's pages; the recalled state's home is that part,
   which need not be the part it was saved from. The owner's latent rows must restore on that part's
   owner only (T05's owner-computed resume covers a state that never leaves its part; a recalled state
   whose home part differs from its save part needs the restore to land on the new owner).
3. **The write path under rank 0.** `attachShare` opens one store on the lead rank as today, but the
   sharded gather has to walk ranks to assemble a block piece; the permit then bounds one rank's piece
   at a time, not the whole state.
4. **Tests.** The host tests here use a fake single pool; the sharded ones need a fake `Parts` (two
   parts, `choose` and `holdPages`) and a tiny per-rank image, asserting a recall lands in one part and
   the save splits by rank.

## Staging permit numbers (pieces under the permit)

T06 measured a 122,880-token **copied** state at 1,112 MiB (9.05 KiB a token) written in 7.2 s under a
2 GiB permit, and a 606,208-token one skipped with `StagingFull`. Under share the staged bytes are the
same rows read through pages, so a 606,208-token shared state is ≈ 5.4 GiB of rows plus its slot. With
piece staging:

- The whole-state attempt is only made when the permit holds it (`w.fits` before any gather), so a
  1 GiB permit never gathers more than 1 GiB at once.
- Otherwise the missing blocks go out a piece at a time, each piece permit-bounded on block edges, and
  the state's own file last. Under the pages layout the own file is just the slot's words and fixed
  parts (small); the rows are all block files.
- **Host RAM for a 600k save at the default 1 GiB permit: ≈ the permit** (one piece staged at a time,
  the staging buffer held at its high-water mark between steps) plus the small own file — no longer the
  whole state. A 600k save now succeeds where T06's whole-state path skipped it; the permit may be
  raised with `TF_GLM_LEARN_STAGING_GIB` but need not be.

## Tests run (host, Zig 0.17.0, `-j8`)

New host tests: `zig/src/core/prompt_radix_disk_test.zig` (fake pool, tiny image, engine_api test
module — run by the master `test` step and `test-rows-host -Drows-filter=`):

- a 2-page state written behind: the save runs, its pages back to the tree's alone, the state on disk;
- an eviction while the save is queued: the hook stages it, its pages held to the end, the pool whole
  again after;
- a restart (new `Rows.open` + `attachDisk`) recalling a 2-page state into adopted pages (rows byte for
  byte a fresh pass's), then a full pool refusing to recall (the pass runs from the start);
- two identities over one root: their own directories, their own sets.

| target | result |
| --- | --- |
| `zig build test-rows-host -Drows-filter=` | 421/421 pass (includes the 4 new tests; the default `rows_` filter's subset unchanged) |
| `zig build test-kv-pool --summary all` | 6/6 pass |
| `zig build test --summary all` | 445/445 steps succeeded, exit 0 (master step; the engine_api suite compiles the new test file in the full native graph) |
| `zig build test-server-cpu --summary all` | 5/5 steps succeeded (served from cache: none of its inputs change) |
| `zig build test-cuda-host --summary all` | 274/274 steps succeeded, 109/109 tests passed; 3 "failed command" lines are the benign stderr writers (each binary exits 0, verified last task) |
| `zig build test-inc4-rows-host` | 4/4 pass |
| `zig build test-window-rows-host` | 4/4 steps succeeded |
| `zig build test-rows-prefetch-host` | 35/35 pass |
| `zig build test-learn-recovery` | 3/3 steps succeeded |
| `zig build test-learn-warning` | 3/3 steps succeeded |
| `zig build test-nemotron-learning-host` | 20/20 pass |
| `zig build test-proposal-rows-host` | 5/5 pass |
| `zig build test-qwen27-prompt-rows-host` | 8/8 pass |
| `zig build test-cuda-disk-open-host` | 5/5 steps succeeded |
| `zig build test-glm-startup-host` | 3/3 steps succeeded |
| `zig build test-glm-pages-host` | 4/4 steps succeeded |
| `zig build test-glm-kv-pages-host` | 30/30 pass |
| `zig build test-glm-snapshots-host` | 27/27 pass |
| `zig fmt --check build.zig zig` | clean |

The flat store's tests are unchanged (the disk fakes in `lane_host_rows_test`, `lane_host_fetch_test`,
`rows_pool_select_test`, `rows_stage_test` only took the new `pages` parameter, unused) — the prompt
cache's flat store keeps writing the files it wrote before this task.

## Adaptations and why

- **`rows.Kept` instead of extending `Entry`:** the task asked for a view both stores produce. The flat
  store's `Entry` carries `decode_spans` and a copy's slot; a tree node's state carries page ids and a
  `freed` hook. The view keeps `rows_write`/`rows_read` type-agnostic and lets `prompt_cache` build it
  from `Entry` without moving anything else there (byte-identical requirement).
- **`pages` on `gather`/`scatter` rather than a second image:** every family's image keeps one vtable;
  the flat stores pass `&.{}` and compile to today's code. A paged image switches on the layout inside
  GLM's disk code, where the pages/copy difference actually lives.
- **Eviction hook on `Tree` (`on_evict`) instead of polling:** the store knows which leaves have saves
  queued (`saves`), and the tree knows when a leaf goes; the hook is the one line where both are true.
- **`io.Job.final`:** a piece job publishes nothing (no entry, no block-group completion) but must
  reserve, gather and write like any job; only the state's own file publishes. `Worker.names` matches
  final jobs so a state's collection keeps its blocks while later pieces run.
- **Recall adopts pages through `Tree.insert` + `take`, not a side channel:** the recalled state is
  stored as an ordinary node whose pages the tree holds, so resume, eviction, and a later restart all
  treat it like any kept state — the only new surface is `blank_pages`.
- **The layout word inside the share facts** rather than a new `Options` field: `identityOf` already
  hashes the share shape and rank prefixes, so the pages layout is one more fact and both layouts get
  distinct directories under one root with no new plumbing.

## Off by default

- Without `--learn-conversations`: `attachShare` never runs, `pages_mode` stays false, the paged store
  has no `disk`, no saves are queued, no pages are held twice; every file written is byte-identical.
- Without `TF_KV_SHARE`: the flat store's path is today's (`Kept.of(e)` when a disk store is attached,
  unchanged queue and files).
- `TF_KV_SHARE` alone (no learn): the Adapter opens as before; `Adapter.disk` stays null; the startup
  line is unchanged.

## GPU checks later (the pair, per the task)

1. A 126k conversation written under the pair: the `imprint: ...` log line, and the state's files under
   the pages layout's directory.
2. The pair restarted; the next turn resumes from disk with the cold reply's hash (the recalled state's
   pages adopted, the reply tokens byte-identical to a prefill-only run).
3. A state evicted by eight other conversations, then read back into pages (the recall path under pool
   pressure, `MemAvailable` and read rate during the read).
4. A 600k state written with the permit at its default 1 GiB (pieces) — and once with the permit raised,
   to compare wall time; block sharing checked against a second conversation sharing the system prefix.
5. The startup line and the refusal: unsharded lift logs "pages and parts"; a sharded start still
   refuses, naming the lift.

## Coordination

The `atc` CLI is not installed in this worktree's environment, so the board/claim/`atc overlap --json`
checks could not run here; the work is on `task/T08-ssd-under-share` and the issue should be linked in
the pull request description. Run `atc overlap --json` before opening the PR.
