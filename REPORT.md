# T05: `TF_KV_SHARE` together with `TF_GLM_KV_SHARD=1`

Branch `task/T05-share-sharded`, three commits, each with its tests green:

1. `75fd58d` — the store's pages over the parts (`kp.Pool.one`/`.parts`), piece 1.
2. `7ef8eda` — the leaf filter and the per-part reclaim, piece 2.
3. `54edb65` — resumes open in their adopted pages' part; the refusal lifted; pieces 3, 6, 7, 8.

## What was done

**Piece 1 — the parts are the store's pages.** `cuda_native.share_pool` now returns
`core.kv_prompts.Pool` — `.one` (rank 0's pool, byte for byte as before) or `.parts`, the sharded
pool's parts: a global page id routes to `parts.pools[partOf(id, h)]` with the part-local id
(`cuda_kv_shard.Parts.retain/release/holders`), `available()` is the parts' sum (the tree's logging
only), and the store's page limit is `2 × h`. `native/cuda_prompt_pages.Adapter` takes either shape.
`core.kv_prompts.Parts` carries the pages' four calls as flat function pointers because
`core` cannot import `prompt_radix.zig` (it lives in the `engine_api` module; importing it by path
is a Zig module-ownership error); the Adapter synthesizes the tree's `Pages` vtable from them, so
the core's types stay engine_api-free. `cuda_share.Keeps.keep` hands the store the parts slot's
global ids when sharded, the lease's unsharded.

**Piece 2 — per-part reclaim.** `prompt_radix_tree.Tree.evictLeaf` takes a `LeafFilter` (a predicate
on a leaf's first page id, plus the free measure to reclaim against; `null` is the old any-leaf
walk, and a pageless leaf is never a victim); `Store.reclaim` forwards it. The adapter's parts
variant gives each part's pool its own hook: a promise in part 0 evicts part-0 leaves until part 0
shows `want` free — never the other part's leaves, whose rank's streams do not wait on this one.
`cuda_share.Keeps.keep` asserts the pages it offers the tree lie in one part (a leaf's pages always
do; `Parts.onePart`).

**Piece 3 — home from the adopted pages.** `cuda_lanes.seat` derives the home part from the begin's
adopted pages (`Pages.partOf(adopted[0])`), a kept state's `homeOf` when there are none. With
adopted pages the placement is strict: `Parts.open` promises in that part alone and takes the
adopted ids as the head of the slot's lease (`kv.Pool.adopt` adds the lease's hold to pages the
tree already holds, as it does for one pool); a short home part is `error.LaneMemory` and the
request waits — it never moves to the other part, and part 0 is untouched by the wait. The adopted
ids are the slot's ids, so the begin step carries them, a follower maps them (`mapWith`'s `kept`
leaves the store's first rows unpoisoned), and `TF_GLM_KV_SHARD_CHECK`'s digest covers them.

**Piece 5 — carries off under share.** Nothing to enable: a share resume's stream always opens in
its adopted pages' part (strict, piece 3), which is the kept state's home part, so
`kv_keep.carry` sees `hold.owner == home` on the home rank and no rows on the peer — `.none` on
both. `TF_GLM_KV_SHARD_CARRY` keeps applying to copy-restores only, as the task says. What a
**page-sourced carry would take** (for a resume placed outside its adopted pages' part, which this
design never does): 0110's `carrySend` walks a snapshot lease's latent parts at fixed offsets; a
page-sourced one would build a `cuda_snapshot_layout.Plan` over the tree node's page ids instead —
each part's piece at `part.live + local_id × part.page` — and reuse `kv_keep.pieces`' cut so the
sender's pieces meet the receiver's; the receiver's side is today's `carryRecv` unchanged (it
already writes the live sequence's latent pages through its table).

**Piece 6 — the kept state's copy under sharding.** Already consistent, verified rather than
changed: `keepAs` leaves the store's pages out (`heldPages`), and their latent rows were never in
the copy; `latent_bytes` is the difference of the with- and without-latent plans over the rows the
copy actually holds, so it accounts for the owner-only rows exactly as T03/0110 left it.
`--learn-conversations` stays refused with share (`native/cuda.zig` refuses the pair
family-neutrally), so 0109's `blankAs`/`seal` never run under share+shard. The digest covers
adopted pages because they are the slot's ids — asserted by the new two-rank digest test.

**Piece 7 — followers.** `mapWith` already accepts a begin step whose leading ids are the store's
(any ids of one part, a twin no longer than the stream); the new test maps a begin with adopted
leading ids and no twin and checks the digest matches rank 0's.

**Piece 8 — the refusal goes.** `KvShareWithShard` and its `explain` line are removed. `TF_KV_SHARE`
with sharding needs `--parallel` ≥ 2, which sharding's own `KvShardNeedsParallel` at the attach
refuses on rank 0 regardless of share. `--learn-conversations` + share keeps its own refusal.

## Blocker (piece 4: twins for a resumed pass) — stopping here per the task

The task says: a resumed pass borrows twin pages for the **new** chunks only, the adopted prefix's
latent rows exist on the owner alone, and if a prompt kernel path needs the prefix on both ranks,
say so and stop. **It does.** Stopped.

Verified in code:

- `dsa_block.attend`'s dense pass (`denseAttention` → `latent.attentionFormat` with `nch` chunks)
  reads **every earlier chunk's** latent rows of the sequence through the latent table; the sparse
  pass reads the selected tokens' rows, which include prefix positions. For a resumed chunk, the
  earlier chunks are the adopted prefix. On the owner those rows are in its pages; off the owner the
  stream's latent table maps those entries at the **trash page** (`latentIds`), so a locally
  attended resumed chunk on the non-owner reads trash into its outputs.
- The owner-computed path (`cuda_kv_shard_attn`) routes **whole segments** (a chunk, or a window's
  rows of one stream) — `Role.local` for a segment both ranks hold, `take`/`give` for one only one
  holds. There is no way to say "this chunk's context rows are owner-only but its own rows are
  local": the kernels take one table and one `nch` per call, and a chunk is one segment.
- So a resumed pass under share must run **owner-computed** (no twin) — which is exactly what piece
  3's code does (`openWith` borrows no twin for an adopted resume), and what the GPU paths already
  support (the TWINS=0 prompt path: `engine.prefillChunk`/`cuda_prompts` unpieced, unpaired, with
  the exchanges on the main stream).

**Best proposal for the operator** (not implemented): make the trash-page reads harmless instead of
moving rows — on the non-owner, a resumed pass's chunk runs owner-computed *as a whole segment*
(prefix and new rows together), i.e. exactly today's TWINS=0 path, and the twin is dropped for
resumed passes entirely. That costs the exchange on every resumed chunk but adds no kernel. The
alternative the task sketched (twin the new chunks only, owner-compute the prefix rows inside the
same chunk) needs a kernel that attends a chunk's rows against two tables by position range — a
GPU-side change, deferred like 0110's batched owner pass.

Consequence for the GPU checklist below: under share+shard, resumed passes run owner-computed; the
"resumed equal to fresh" checks must pass with `TF_GLM_KV_SHARD_TWINS` at its default (a fresh pass
twins, a resumed one does not — both must be byte-correct).

## Adaptations, and why

- **`core.kv_prompts.Parts` is a flat function-pointer struct, not a `radix.Pages`.** `core` files
  cannot import `prompt_radix.zig` (module ownership: it belongs to `engine_api`, which `core`
  imports the other way). The Adapter bridges.
- **`Store.reclaim`/`Tree.evictLeaf` take the filter as a parameter, not per-store state.** One store
  serves both parts' hooks; the part is known at the hook call, so it rides the call.
- **`Parts.open` grew the `adopted` parameter** rather than a separate `openAdopted`: the twin
  decision and the strictness belong to the same placement, and the existing callers pass `&.{}`.
- **`openWith` borrows no twin for an adopted resume** (see the blocker) instead of the task's
  "twin the new chunks only" — that shape needs the kernel decision above.
- **`cuda_pages_test.zig`'s `Store` fixture** reads `share_pool(e).?.one` since `share_pool` returns
  the union now.

## Tests

| Command | Result |
| --- | --- |
| `zig build test-kv-pool --summary all -j8` | 5/5 steps, 34/34 tests pass |
| `zig build test --summary all -j8` (whole tree, first full run) | 445/445 steps, 3619/3626 tests pass (7 skipped, 0 failed) |
| `zig build test-cuda-host --summary all -j8` | 274/274 steps, 607/608 tests pass (1 skipped) — includes the glm module's runs with the new pieces-1/2/3 tests |
| `zig build test-server-cpu --summary all -j8` | 5/5 steps succeed |
| `zig build test-glm-pages-host --summary all -j8` | 4/4 steps, 13/13 tests pass |
| `zig build test-glm-kv-pages-host --summary all -j8` | 4/4 steps, 30/30 tests pass |
| `zig fmt --check build.zig zig` | pass |

New tests: the parts as the store's pages (routing, holders, the sum, the per-part hook);
`evictLeaf`'s filter evicting a part-0 leaf while the part-1 state stays (and no unfiltered leaf
taken for a filtered ask); the sharded adapter keeping and resuming a state whose pages lie in part
1; a keep in part 1 and a resume opening in part 1 with the holds moving to the lease; a full home
part waiting (LaneMemory) with part 0 untouched; a seat with adopted pages (owner part, no twin,
holds given back on a failed seat); the follower mapping a begin whose leading ids are the store's,
digests equal across ranks. Pre-existing failures: none — NOTES.md's 0094/0095 failures were fixed
on this branch's base (T07), and every target above is green.

## Left, and uncertain

- **Piece 4's decision is open** (the blocker above). Until the operator picks, resumed passes under
  share+shard run owner-computed; the twin counters never count them.
- A share resume whose adopted part lacks room waits as long as admission keeps asking (`fits` says
  a part fits when *either* does; the seat then fails and the request retries). The wait is not
  counted in any counter — if the operator wants `/health` to show it, it is one line in
  `Pages.noteWait`'s sharded path, left out here to keep the log surface unchanged.
- `Pages.fits` (sharded) does not know about a pending resume's adopted pages; a resume whose part
  is full can be admitted beside others and then wait at its seat. Same behaviour as the unsharded
  T03 path; noting it so the GPU checklist's "does not hold a new request back" check is read with
  this in mind.

## GPU checks needed later (the pair, both switches on)

- `tf-cuda-test glm-kv-shard-check` byte-identical, extended with a share path: a keep under share in
  part 1, a resume whose begin adopts those pages, owner-computed (no twin), against an unsharded
  twin run.
- Resumed (share) equal to fresh at 126,976 and 600k tokens in **both parts** — the resume must land
  in the kept pages' part and prefill the tail only.
- Concurrent equal to alone at 1–8 streams; drafted equal to plain (a resumed pass owner-computes —
  its tokens must still match a fresh pass's).
- Both parts' idle census `free + store == total` (the store's holds live in the parts' pools now);
  a part full of kept leaves evicts **only that part's** leaves under a promise there (watch
  `/metrics` `kv_shard` part frees and the store's evicted counts), and does not hold a new request
  back beyond LaneMemory's wait when the other part has room for a fresh (non-resume) stream.
- Decode within 2 % of T03's; a resumed prompt's owner-computed chunk cost within the TWINS=0 budget
  0110 already measured.
- With the piece-4 decision taken, whichever path it picks (owner-computed-always vs a range-routing
  kernel) needs its own byte-equality check on the GPU.
