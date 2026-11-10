# T11: the disk tier under `TF_GLM_KV_LIVE_HALF=1` (`--learn-conversations`, sharded and unsharded)

## Cause (read, not guessed)

With `TF_GLM_KV_LIVE_HALF=1` a keep leases the live KDA half alone: `layout.Plan.make` splits the arena
head around the dead half, so the kept state's lease holds `prefix - half_len` bytes of prefix
(`half_len` = `kda_layers * rec_layer`, the dead half's span) and everything past it sits `half_len`
lower than the whole-head layout. The disk tier sized a kept state's image from `image.Shape`, which
knows only the whole head: the startup probe (`cuda_glm_disk.probe`) kept a state at its mark and
gathered the whole store image in one range; `ranks.copy` built `segmentsFor` on the whole-head shape
and `Copy.leased` read past the half-sized lease — `error.SnapshotSize` on both ranks, right after the
opt-in prompt numerics warning, before the pool line. Exactly the reported GPU symptom (`bc1b012`,
`TF_GLM_KV_SHARD=1`, `TF_KV_SHARE=32`, `--learn-conversations`).

The same mismatch sat on every disk path: the copy layout's `segmentsFor` (one prefix run from lease 0),
the pages layout's fixed-part walk in `cuda_glm_disk` (`lease` accumulating whole-head offsets into
`slot_put`/`slot_take`), `leaseBytes`, and `seal` — which sized a recalled state by the blank's `cur`
(all zeros) instead of the words' parity, so even the restored size was wrong under the switch.

## The change (one commit, `5b16b8c`)

The image keeps the dead half's place; the lease is what is shorter. With the switch off every path is
byte-for-byte as before (no half is derived, no identity fact is added, `segmentsFor` emits the same
runs, `leaseBytes` subtracts 0).

- `cuda_disk_image.zig` — `Shape` gains `half: ?u1`, `dead` (the dead half's first byte in the prefix)
  and `half_len`. `Shape.ofCur(st, c, capacity, head, cur)` derives them from a kept state's own parity
  (`layout.halfOf` over the state's `cur` bits, as `Plan.initFor` plans a restore by the keep's — the
  slot's may have moved on); `of` is `ofCur` over the live state's. `segmentsFor` lays the prefix
  section through `prefixSegs`: one run, or a live-half keep's two around the dead half's place
  (`.zeros` in the image — gather writes zeros there, scatter drops them — and no bytes in the lease);
  the layers' parts follow at the shifted lease. `leaseBytes` returns the half plan's length.
  `fixedLease(sh, x)` maps a pages-layout fixed part's image byte past the words to its slot-lease
  offset: `x` before the dead half, `null` in its place, `x - half_len` past it. The image layout
  itself (`parts`, `pageParts`, `storeParts`, `route`, `imageBytes`, `heldShared`) is unchanged — the
  file keeps the dead half's place as zeros, so the store's part plan, blocks and identity inputs stay
  state-independent.
- `cuda_disk_ranks.zig` — `shapeOf(e, s)`, the engine's shape with the state's own parity; `copy` (the
  copy layout's per-rank gather/scatter) builds its segments from it, so each rank's copy of a half
  keep reads and writes its real lease. `gatherWith`/`scatterWith`'s shape feeds `route` only (image
  layout) and stays `shape(e)`.
- `cuda_snapshots.zig` — `seal` sets the blank's `cur` from the words before sizing (the size and the
  half follow the keep's, not the slot's), and, under the switch, refuses a state whose words say both
  halves (`error.DiskWords`): a whole-image state has no lease here, so a recall of one is a clean
  `bad()` (the state forgotten, a cold pass) and never a silent mis-restore. With the switch off, both
  halves seal as before.
- `native/cuda_glm_disk.zig` — the four pages-layout walks (`gatherPages`, `gatherPagesSharded`,
  `scatterPages`, `scatterPagesSharded`) take the kept state's shape (`ranks.shapeOf`) and map fixed
  parts through `fixedLease`: the hole's bytes are zeros on the gather and skipped on the scatter; the
  rows and the rank prefixes are untouched (the rows sit in pages, half-independent). The identity
  facts moved into one `identityFacts` helper used by `attach` and `attachShare`, which appends one
  fact when `TF_GLM_KV_LIVE_HALF` is on — the store's identity changes, so a live-half store is a new
  directory under the same root and the two settings' sets never mix (the off identity is unchanged,
  so existing stores keep their directory).
- `cuda_snapshot_layout.zig` — `test_switch`, a host test's word for the switch (production never sets
  it; the environment rules), so the real keep/disk paths are testable with the switch on.

**What wins where recipe and upstream differ:** there is no upstream sha for this pair (upstream
refuses the combination; the live-half keep is `7547c99c2`, the disk tier `389769733`, this lane's
T08/T08b the pages layouts). The recipe's 0110 sharded disk step and T03's pages layout keep their
shapes; the half enters beside them (the head split is prefix-region only, never 0110's latent parts,
whose pages and owner handling are untouched).

**Why the identity fact and not a file-words flag:** within one store the half is per-state (the
words' `cur` already carries the parity), so no file needs a flag; what must never happen is a store
written under one setting read under the other — a half file read with the switch off would seal (its
whole-head size fits a whole lease) and restore through the wrong mapping. The identity fact separates
the directories, which the rows layer already keeps apart (`rows.Rows` one directory per identity
word). A half file read with the switch off *within one identity* is impossible by construction; the
seal's both-halves refusal covers the one whole-image case that could otherwise mis-restore silently.

## Tests (all host, Zig 0.17.0, foreground, `-j8`)

New tests (13 across three files):

- `cuda_snapshot_layout_test.zig` — with the switch on, `Plan.init` follows the slot's parity (a
  uniform one keeps a half — same `len`/`count` as `withHalf`; the switch off keeps both halves, and
  `Plan.size` — the disk store's charge — follows too).
- `cuda_disk_image_test.zig` — a live-half shape splits the prefix at the dead half and `leaseBytes`
  equals the half plan's at both parities (the image's own bytes unchanged); `fixedLease`'s mapping
  (before / hole / past); a half keep's image round-trips through its lease into a fresh state at both
  parities with every planned byte equal and the dead half's place zeros (gathered in ranges, as the
  store asks); a disk state's seal under the switch takes a half's parity, refuses both halves
  (`DiskWords`), and with the switch off seals both as before.
- `cuda_disk_shard_test.zig` — a live-half keep round-trips through a store image into a new stream,
  sharded at both homes (with the home part full: the rows carried to the other part, as from memory)
  and unsharded, restored bytes equal (`expectRestored`); the startup probe's shape under the switch —
  keep at the mark, the whole store image gathered in one range, each rank's shared bytes compared
  equal to the image's (`heldShared`, the probe's RanksDiffer check), sharded and unsharded. This is
  the exact path that ended at startup with `SnapshotSize`; against the pre-fix head it fails there.
  (`Parts.of` now reads each part's own `lat` flag instead of hardcoded whole-plan indexes, so the rig
  compares the right parts of a half plan.)
- `cuda_glm_disk.zig` — the identity facts: with the switch off the facts (and the identity) are as
  before; on, one fact more, after the pages layout's word and before the ranks' prefixes.

| command | result |
| --- | --- |
| `zig build test-cuda-host --summary all -j8` | 274/274 steps succeeded, 629/630 tests passed (1 skip, pre-existing) — includes the ranks, snapshot-layout, disk-image, disk-shard and glm-disk tests |
| `zig build test-glm-snapshots-host --summary all -j8` | 27/27 pass |
| `zig build test-glm-kv-pages-host test-glm-pages-host test-cuda-disk-open-host test-glm-startup-host test-glm-snapshots-host --summary all -j8` | 17/17 steps, 58/58 pass |
| `zig build test-glm-fp8-pages-host test-glm-gathers test-glm-expiry --summary all -j8` | 10/10 steps, 32/32 pass (gathers/expiry's "failed command" line is the intentional warning-path test) |
| `zig build test-rows-host -Drows-filter= --summary all` | 429/429 pass |
| `zig build test-kv-pool --summary all` | 35/35 pass |
| `zig fmt --check build.zig zig` | clean |
| `/home/user/invt/glm-kv-share/cuda-compile-check.sh` | **CUDA COMPILE OK** (652 s) — ran because the task touches `zig/src/native/cuda_glm_disk.zig` |

No NOTES.md failure was touched (none remain on this lane per T08b). The 1 skip in `test-cuda-host`
is the same skip as on the clean base. The `test-cuda-host` "failed command" stderr lines are the
benign stderr writers noted in T07/T08 (each binary exits 0).

## Adaptations, and why

- **Zeros in the image, not a smaller file:** the store's file layout comes from the image vtable's
  `parts`, which sees only `(engine, at)` — a state's half must not change the part plan, the block
  boundaries or the identity inputs. Keeping the dead half's place as zeros costs `half_len` bytes of
  zeros per state (about 1.7 MiB of 448 MiB at the 448-layer shape's 126k states — under 0.5%) and
  makes every mapping a pure lease-side concern.
- **The seal sizes by the words' parity, and refuses both halves under the switch:** the blank's
  `cur` is zeros until the words arrive, and the rows of a recall's blocks are scattered before the
  state's own file (the words) is read — so the pre-seal mapping is the half mapping the blank's zeros
  parity yields (right for every real file, since a serving pass keeps all KDA layers on one parity).
  A whole-image state (a keep with layers on both — no serving path produces one) is refused at the
  seal rather than mis-restored.
- **The probe-shaped test instead of calling `probe`:** `probe` needs a full backend and
  `glm.native`'s engine; the rig replays its exact steps (keep at the mark, one-range whole-image
  gather, per-rank `heldShared` compare) on the two-rank host engines.
- **The T08/T08b rows fakes needed no change:** the half lives in GLM's image and slot geometry, which
  the rows layer sees only as opaque parts; the rows-level behaviour of a state that does not fit its
  blank (`SnapshotSize` → refused → `bad()`, state forgotten, cold pass) is already covered by
  `rows_refusal_test`.

## Left / uncertain

- The pages-layout walks (`gatherPages`/`scatterPages`, sharded and not) are exercised by compilation,
  the `fixedLease` unit test and reading only — no host target runs them against a real engine's pool
  pages (as T08b already noted for the per-rank disk ops). The copy-layout round trips and the probe
  shape, which is what failed on GPU, are host-tested.
- A whole-image file (a keep with layers on both halves) is refused at recall under the switch. If a
  serving path ever produces mixed parities, such states save fine but recall as a cold pass; making
  them work needs the words read before the rows are scattered (a rows-layer ordering change).
- `atc` is not installed in this environment (as in T08's run), so the board/claim/`atc overlap --json`
  checks could not run here; the work is on `task/T11-live-half-disk` and the issue should be linked in
  the pull request description. Run `atc overlap --json` before opening the PR.

## GPU checks needed (the pair that failed, plus the shard)

On the DGX boxes, GLM-5.3-Flash two ranks:

1. **The incident's exact start must now pass:** `TF_GLM_KV_SHARD=1 TF_KV_SHARE=32
   TF_GLM_KV_LIVE_HALF=1 --learn-conversations --prompt-cache-gib 7` starts, logs the
   `conversation states on disk in <dir>` line — in a **new directory** beside the non-live-half store's
   (the identity fact) — and the probe's keep lands.
2. A 126k conversation written under the pair (its state's files under the live-half store's
   directory; the state file's dead half's bytes are zeros), the pair restarted, and the next turn
   resumed from disk with the cold reply's hash (tokens byte-identical to a prefill-only run) — the
   restore rebuilds the slot's dead half from the kept parity.
3. The same with `TF_GLM_KV_SHARD=0` (the disk tier alone under live-half): write, restart, resume
   byte-identical — the copy layout's paths.
4. A restart with `TF_GLM_KV_LIVE_HALF` unset over the same root: a fresh store (new directory), the
   live-half set's states not read, and vice versa — the sets never mix.
5. The recall of a state whose save raced a parity flip cannot arise on GPU (all layers flip
   together); if a `bad()` line for `DiskWords` ever appears under the switch, that invariant is broken
   and this report's seal refusal is why the run stayed up.
