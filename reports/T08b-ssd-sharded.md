# T08b: the SSD tier under shared pages with `TF_GLM_KV_SHARD=1`

## What was done

The last refusal is lifted: `--learn-conversations` beside `TF_KV_SHARE` **and**
`TF_GLM_KV_SHARD=1` now serves for GLM-5.3-Flash. Ported from upstream's
TF_KV_SHARE disk tier (`tf-at-share` `389769733`) onto the sharded pool, beside
the recipe's 0110 disk step (`families/glm/cuda_disk_ranks.zig`). One commit
carries the port, one its test callers, one the nvcc-only compile fixes; rebased
on `share-port` (clean, both sides kept: the attachShare probe fixes, the share
step, `Adapter.openWith`, `LaneHost.paged_disk`, and the write-behind pages fix).

- **The sharded writer** (`cuda_disk_ranks.zig`, `cuda_glm_disk.zig`): the pages
  layout gains `pages_sharded` (`Words.pages_sharded=5` folds its own word into
  `attachShare`'s identity — the copy store, the pages store and the sharded
  pages store never mix). `pageParts` takes the sharded flag: a page's latent
  codes and scales sit in its part's latent page, its pooled keys in its key
  page, so each row part's place (`Part.at`) is in its own page there. The
  gather walks the ranks a piece's bytes come from — the words and the fixed
  parts from rank 0's slot, the pooled keys from rank 0's key pages, the latent
  codes and scales from the home rank's latent pages (through the link when the
  home is rank 1), each other rank's prefix through the link — all written
  through rank 0's files. The read-back scatters the keys into every rank's key
  pages, the latent rows into the home rank's alone, the fixed parts into rank
  0's lease and past each rank's prefix into its own. The staging permit bounds
  one rank's piece at a time; blocks stay 256 tokens; a write in progress holds
  its pages through the parts (`Parts` retain/release by global id).
- **Recall into one part**: `blankPages` chooses the recall's home part
  (`Parts.choose`), holds its pages there as global ids (`Parts.holdPages`),
  records the shared page count and home on every rank (`ranks.blank` →
  `blankAs`), and the read scatters into that part's pages — the recalled
  state's home is that part, which need not be the part it was saved from; the
  latent rows restore on that part's owner only, the keys on both ranks. The
  recalled path stays in one part (T05d's invariant) and a later resume from it
  takes no twin, subject to T05c's adopt rule like any kept state.
- **Refusal lift + startup line**: `shareDisk` allows the sharded pair, so the
  `--learn-conversations`/`TF_KV_SHARE` refusal no longer names sharding; the
  startup line reads `glm: conversation states on disk in <dir> (...; 2 ranks'
  parts, pages and parts...)`.
- **Reclaim interplay**: the per-part reclaim path is the tree's
  `reclaim(want, in)` with a part filter (`evictLeaf(in)`), which the new tests
  drive; an evicted leaf whose save is still queued is staged by the eviction
  hook (`rows.write.evicting`, may_wait=false) and never blocks a promise — a
  promise short of pages waits as `LaneMemory` does (the writer holds its
  pages; the queued save's pages are held through the parts).
- The probe path was read end to end under sharding (TASK's wrinkle): the probe
  keeps whole copies with the store standing aside (`share_pages=false` and
  `sharePages(false)` on every rank); the probe's keep is a copy keep whose
  latent rows lie on its home rank alone, and the RanksDiffer comparison goes
  through `homeOf` + `img.heldShared`'s home filter, so it compares only what
  each rank holds. No change was needed there beyond the port.

## Adaptations, and why

- `Parts.at`/`pageParts`'s per-part offsets, the slot-lease `slotBytes`, and the
  per-rank disk ops (`key_put`, `lat_put`, `lat_take`, `slot_put`, `slot_take`,
  `page_put`, `page_take`) are new against upstream's flat store: the recipe's
  0110 layout splits a page by rank, so a row part's bytes live in its own page
  and another rank's bytes ride the follow link in pieces sized by the link's
  id buffer.
- `blankAs` carries the recall's shared page count (`shared`) beside `home`, so
  the slot's seal and restore size past the pages' rows on every rank; its test
  callers pass none (unsharded).
- The unsharded paths are byte-for-byte as before: `pageParts`'s new argument is
  `false` there, `gatherPages`/`scatterPages`/`poolRows` keep the whole-page
  layout, and `shareDisk`'s change only drops the sharded refusal (the copy
  store's identity word is unchanged).
- The new host test's rig needed real writer semantics to pass (it had never
  been run): a part holds 256 pages (a 512-token state is 128 pages and must lie
  in one part), the rows are wide enough that a block piece dwarfs the state's
  metadata, a keep's home comes from the stream's part (as `ranks.readHome`
  does), the permit is two block-pieces (the whole state's charge never fits it,
  so the save goes in pieces), the eviction test follows the writer's actual
  drain semantics (a full drain writes every stageable save, so the eviction
  comes first and the two-leaf reclaim last), and the reclaim filter's `free`
  reports the part's real availability.

## Tests (all commands run on this head, post-rebase)

| Command | Result |
| --- | --- |
| `zig build test-rows-host -Drows-filter= --summary all` | 426/426 pass (the 6 new sharded-disk-tier tests included) |
| `zig build test-rows-host --summary all` | 112/112 pass |
| `zig build test-kv-pool --summary all` | 35/35 pass (29 + 6) |
| `zig build test-cuda-disk-open-host --summary all` | 9/9 pass |
| `zig build test-glm-kv-pages-host --summary all` | 30/30 pass |
| `zig build test-glm-pages-host --summary all` | 13/13 pass |
| `zig build test-glm-snapshots-host --summary all` | 27/27 pass |
| `zig build test-cuda-host --summary all` | 274/274 steps succeeded, 2524/2527 tests (3 skips, pre-existing) |
| `zig fmt --check build.zig zig` | pass |
| `/home/user/invt/glm-kv-share/cuda-compile-check.sh` | **CUDA COMPILE OK** (201 s) |

The `test-cuda-host` "failed command" stderr lines are the benign stderr writers
noted in T08's report (each binary exits 0). The 3 skips are the same skips as
on the clean base. No NOTES.md failure remains (T07's report records the old
ones as fixed on share-port).

## CUDA compile check

Ran (the task touches `zig/src/native/cuda_glm_disk.zig` and
`cuda_disk_ranks.zig`). It caught two errors no host target compiles: a
`pageRows` call missing its shape argument and `poolRows` taking a mutable host
from a const read-back. Fixed; it then printed `CUDA COMPILE OK`.

## Left / uncertain

- The per-rank disk ops and the link piece-size (`link.ids.len * 4`) are
  exercised by compilation and by reading only — no host target runs a two-rank
  engine. The GPU checks below are the real coverage.
- A recall's `at` must be a whole page count of the **pages layout**
  (`at % sparse.pool == 0` in `pageParts`); the fake rig uses its own 4-token
  pages, so the GLM-sized arithmetic (a page's region per layer,
  `Words.pages_sharded`'s field order) is proven by the image tests and the
  compile only.

## GPU checks needed later (the pair: share + shard + learn)

Per TASK.md: a 126k and a 600k conversation written (log lines, pieces under a
1 GiB permit); the pair restarted; the next turn resumed from disk with the cold
reply's hash; a state evicted by other conversations read back (also into the
other part than it was saved from — the latent rows must land on the new
owner's rank alone); `MemAvailable` low-water on both ranks during write and
read at a 5 GiB reserve; and the startup line's `2 ranks' parts, pages and
parts` with the sharded identity (a restart after switching `TF_GLM_KV_SHARD`
off must start a fresh store, not mix sets).
