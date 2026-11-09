# KV sharding on the 1104 recipe engine (`TF_GLM_KV_SHARD`)

Port of Robert Wojciechowski's (GitHub @wojo, X @rwojo) GLM KV sharding (GLM patch 0118; its Zig port on Mia's old
engine, `tensorfold-glm` `b337611d7` host side and `260af6686` device paths, GPU-validated as g1-g3) onto the engine
the recipe `MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104` serves: upstream TensorFold `f350cd3e` with the recipe's 106
`patches/engine/*.patch`. GLM code lives in `zig/src/families/glm/` here (`glm_cuda/` on the old engine).

At TP=2 every stream's MLA latent rows are normally stored on both ranks. Sharded, each stream's latent rows live on
one rank only, its *owner*; the KV pages hold about 1.8 times the tokens in the same memory. Off (the default), every
launch, buffer, page and step is as before.

## What is sharded, and what is not

A KV page here holds 512 positions of every DSA layer (and, with the target MTP head, its layer): per layer the E4M3
latent codes (512 x 512 B), their fp32 scales (512 x 4 B) and the 128 pooled index keys of those positions
(128 x 128 x 2 B). 11 DSA layers: 3,266,560 B a page, 6,380 B a token (`cuda_state.pageBytesOf`).

| Part of a page | Bytes a token (11 layers) | Sharded |
| --- | --- | --- |
| latent codes and scales | 5,676 | yes: on the stream's owner only |
| pooled index keys | 704 | no: on both ranks, every page |

The pooled keys stay replicated because the DSA indexer runs replicated (its weights are replicated,
`cuda_dsa_block.Index`; `TF_GLM_INDEX_SPLIT` splits a prompt chunk's rows, not its keys, and all-gathers the picks):
both ranks select the same tokens for every row, so the owner can attend for both ranks' heads with its own token
lists. The index rings (`ik`/`ig`, per slot in the sequence arena), the KDA states and conv windows, DFlash2's context
ring and the MTP position stay as they are: a slot's, on every rank.

Sharded memory a rank: every page's keys (`2H x 360,448 B`) plus its own part's latent pages and one trash page
(`(H + 1) x 2,906,112 B`), H pages a part. The pool's layout (`kv_pool.Layout`) then says 3,542 B a token
(`(704 + 5,676 / 2)`, exact), so the server's plan gives about 6,380 / 3,542 = 1.80 times the pages; `attach` drops
four pages and an odd one so the trash page and the halves fit inside what the plan counted.

## Ownership: parts of the pool

The page ids are cut in two equal parts, part r = ids `[r H, (r + 1) H)`, owned by rank r. A stream takes all its
pages from one part (rank 0 decides, as it decides every lease; the begin step carries the ids, so the follower's
owner is the same by construction). Part choice: the kept state's home part when the request resumes one and that
part has room, else the part with fewer live streams, then more free pages, then the lower index (the old
`Pool.placeIn` rule). A request needing more than H pages is refused at its opening (`error.PoolTooSmall`), so the
longest request is a part, not the pool (`attach` refuses a part under max(131,072 tokens, capacity / 4), the old
`PartTooSmall` floor).

Each paged sequence gets a second device table when sharded, its *latent table*: entry i is the part-local latent page
of key page i when this rank holds it, else the trash page (index H). The kernels take a table and a stride per call,
and the latent calls and the key calls are separate launches everywhere, so `dsa_block.Caches` grows two fields:
`lpt`/`lps` (latent table and page stride) beside `pt`/`ps` (keys). Unsharded both are the same table and stride, so
every launch's arguments are byte for byte as before. Writes of a row this rank does not hold land in the trash page;
reads of one never reach an output that is kept (below).

## Decode and verify windows: owner-computed attention

This engine launches every forward eagerly (no CUDA graphs here, unlike the old engine's captured decode), so the
host knows each round's segments and owners and can size every exchange exactly. Per DSA layer, after the replicated
projections, latent and key writes and token selection, for each segment (one stream's rows of the window):

- both ranks hold its latent rows (a prompt twin, below): each attends its own heads, as unsharded;
- one rank holds them: the other sends it its absorbed queries (`qa`, rows x 32 heads x 512 bf16) for those rows;
  the holder attends its own heads (into its `ol`) and the peer's (into an exchange buffer) with the same kernels,
  the same token lists and the same call shapes the peer would have used; the peer's outputs go back into the peer's
  `ol` rows. Then `expand_v`, `o_proj` and the gathers run as before.

The heads' attention is independent head by head and the inputs are byte-identical, so the result is bit for bit the
unsharded one. A window with any exchanged segment runs every segment on its own launches (`dsa_block.attend`, the
per-segment path the batched `attentionSegs` launch is bit-equal to): the batched kernels find a row's segment by
`row0` alone (`glm_attn::seg_of`), so a table holding only some segments is not safe. That costs launches, not bits; a
batched owner pass needs a kernel that skips rows outside its table (a GPU-side change, deferred). A rank that lacks
a segment's rows still writes its latents (into its trash page), its index keys and gates, and picks its token lists
(`dsa_block.attendNoLatent`), so the replicated keys stay equal and a split selection's all-gather has both ranks.
Code: `cuda_kv_shard_attn.zig` (`run`: the phases, generic over the device; `Gpu`: the CUDA one; `single` for
`Forward.mixer`, `shared` for `cuda_shared.mixer`). Exchanges are grouped NCCL send/recv of whole segment row ranges on the main stream, one group a phase
(queries, then outputs); a round with no exchanged segment posts none. No new kernel is needed, so there is no
`glm_shard.cu` here: the old one's counts/select kernels existed for the captured graphs, whose shapes could not vary
by round, and its mark/scan/compact/renumber kernels for the prompt history fetch, which this port replaces (below).

## DFlash2 (decision: supported)

The recipe serves with `DRAFTER=dflash2` by default, so sharding that refused it would be off in practice. DFlash2
composes:
- its verify windows are target forwards over the target's KV pages, the same `Forward.run` / `shared.mixer` path as
  any decode window (up to 16 rows a stream, 32 a shared window); owner-computed attention covers them unchanged;
- its own context ring (`dflash.ring_bytes` a slot) stays a slot's on every rank and replicated; its taps are hidden
  rows after the gathers, the same on both ranks;
- beside DFlash2 the target MTP layer is not loaded and its caches take no page part (`engine.heads`), so the only
  paged caches are the 11 DSA layers' and every one of them is sharded the same way;
- kept states carry DFlash2's ring as the plan's last part on every rank, as before.

What only the GPU can show: that DFlash2's selected tokens and acceptance are identical sharded and not (the check
program's `tokens` comparison), and the cost of the per-layer exchanges at verify widths.

The target MTP head (drafts without `--drafter`, `DRAFTER=mtp`) stays refused: its layer's caches are a separate part
of every page and its chains run their own DSA block (`Forward.mtp`) outside the segment path.

## Prompt chunks: twins, else owner-computed

The old port's prompt history fetch (owner sends the history rows a chunk's selection names into a compact copy, the
receiver renumbers its token lists) is replaced:

- **Twins** (`TF_GLM_KV_SHARD_TWINS=1`, default): a prompt pass borrows, from the other part, as many pages as its
  prompt needs; on the non-owner the latent table points into them for the pass, so both ranks write and attend
  their own heads locally, exactly as unsharded. When the pass resumes a kept state, the latent rows up to the resume
  point are carried (below) into whichever rank's pages lack them. The twin is given back at the stream's first
  verify window (`Engine.verify`/`verifyShared` on every rank, before the forward), so no decode round reads it; a
  cancelled or ended stream gives it back with its pages.
- **Owner-computed** (no twin: `TWINS=0`, or the other part lacks the pages): the decode path above, on the prompt
  buffers (a chunk is one segment; a batched prompt pass's segments each). Such a chunk runs unpieced and unpaired
  (its exchanges on the main stream, never beside the side stream's), which costs speed only (same bits).

## DSA selection under sharding

The selection (`selectTokens`: scores over the pooled keys, top-k pools, the token lists) runs on both ranks over the
replicated keys, as unsharded; only the attention over the selected latent rows is the owner's. The old engine did
the same (its pooled keys were pool-wide on both ranks). A row under the dense limit attends every earlier position
(the dense pass), which is again the owner's latent rows only.

## Kept states (prompt cache)

`cuda_snapshots` copies a prefix of the live sequence into 2 MiB snapshot pages (device, then the pinned host tier
`TF_GLM_HOST_CACHE_GIB`). Sharded:
- the state's **home** is its stream's owner part, recorded in `Snap.home` on both ranks;
- the home rank's copy holds everything as unsharded; the other rank's copy holds the same parts **without the latent
  codes and scales** (fixed fields, KDA states, each DSA layer's rings and pooled keys, the MTP position, DFlash2's
  ring);
- both ranks lease the unsharded size, so rank 0's admission (which charges its own lease) can never pass a keep that
  the other rank's pages cannot hold (the old port's "safe over-count");
- a resume prefers the home part. Restoring, each rank copies its own parts back; then the latent rows go to every
  rank that holds the new stream's latent rows but is not the home (`carry`): the home rank copies them out of its
  snapshot pages into a staging buffer and sends them, the other receives them into its latent pages, 32 MiB a piece,
  the same pieces on both ranks. That covers both a stream placed outside its home (`TF_GLM_KV_SHARD_CARRY=1`) and a
  resumed pass's twin. With `CARRY=0` a resume whose stream could not go home starts from position 0. (The old
  `carry_min` of 2,048 tokens never binds here: a kept state sits on the 4,096-row chunk grid.)
- the host tier works unchanged (the staging copy reads either tier through the same piece walk as `move`).

## Kept states on disk (`--learn-conversations`, #1411)

The rows store keeps one image a state in rank 0's files: rank 0's image, then each other rank's prefix; every byte
past rank 0's prefix stands for every rank (`cuda_disk_image.route`). Sharded:
- **writing**, the image is the unsharded one: each DSA layer's latent codes and scales are read from the home rank
  alone, everything else from rank 0 as before. A rank off home reads and writes no latent byte (its image segments
  there are `absent`; asked anyway, `KvShardAbsent`), so the unfilled tail of its lease never reaches the disk. The
  image's parts and size are those of an unsharded state at the same position, so the store's rows by position stay
  shared whatever the home, and the disk costs what an unsharded state costs, no more;
- the **words** carry the home and the parts in the format word: format 3 = `3 | home << 8 | 2 << 16`. Unsharded
  states keep format 1, byte for byte. `decode` refuses a home outside two parts or any other format (2 included:
  the earlier image that carried DFlash2's drafter ring). The image carries no drafter ring either way, so a state
  read from disk restarts DFlash2 at its mark, sharded or not;
- **reading**, rank 0 picks the home before any byte arrives (the store sends the rows by position before the state's
  own bytes and words): the part a new stream would take now (`Parts.choose` with no home), sent in the blank step.
  The written home does not bind: the bytes are the same for either, and the stream that reads the state is placed
  now, not when it was written. The latent rows go to that rank alone; the seal sizes the other rank's copy without
  them. The sealed state is then a kept one, so the resume prefers its home, carries when placed elsewhere, and with
  `CARRY=0` starts from 0 outside it (`KvShardNotHome`), by the memory paths;
- **cross reads** read correctly. An unsharded image read sharded is placed as above; a sharded image read unsharded
  ignores the home and gives every rank the latent rows. The store identity facts are unchanged; the start-up probe's
  hash covers the image's bytes (latent rows from the probe's home), so if owner-computed rows ever differ from
  replicated ones the identities differ and the two never share a directory. The probe compares each other rank's
  held bytes alone with the image's.

## Switches and their agreement

| Switch | Default | Meaning |
| --- | --- | --- |
| `TF_GLM_KV_SHARD` | 0 | 1: shard the latent rows over the two ranks |
| `TF_GLM_KV_SHARD_CARRY` | 1 | a resumed state's rows carried to a stream outside its home part (0: resume from 0) |
| `TF_GLM_KV_SHARD_TWINS` | 1 | prompt passes borrow the other part's pages (0: owner-computed prompts); no decode twins |
| `TF_GLM_KV_SHARD_CHECK` | 1024 | decode rounds between the ranks' layout digests (0: none) |

Every rank reads its own (`cuda_kv_shard.Switches.read`, inside `cuda_ranks.Settings.read`); they ride the rank
fingerprint (`cuda_ranks.Fingerprint`, layout 15: `kv_shard` = on | carry << 1 | twins << 2, and `kv_shard_check`,
both 0 when off), so ranks that differ refuse with `RankMismatch` naming the switch. A value a rank cannot parse is
that rank's `BadSetting` with the switch's words, before NCCL, as every other GLM setting here is (the start-up's
liveness lines take the other rank down with it; the old port's `unreadable` sentinel belonged to its settings-text
agreement, which this tree does not have). The fingerprint grows by these two words whether or not sharding is on;
it is exchanged between the ranks at start only, so nothing served or computed changes. The refusals then run on
every rank alike from agreed inputs, before the weights load (`Engine.init`, after `ranks.agree`):

- TP other than 2 (`KvShardNeedsTwoRanks`);
- `--kv-cache` other than fp8 (`KvShardNeedsFp8`) and KV pages off (`KvShardNeedsPages`): only E4M3 pages are sharded;
- the target MTP head drafting (`KvShardWithMtp`: drafts without `--drafter`);
- and at the pool's attach on rank 0: `--parallel 1` (`KvShardNeedsParallel`: one stream would leave a part empty),
  a part too short (`KvShardPartTooSmall`).

`TF_GLM_MULTI_LONE` does not exist in this tree, so there is nothing to refuse. Disk spill of GLM CUDA kept states
(#1411, `--learn-conversations`) writes and reads sharded states as described under *Kept states on disk*.

## Placement and layout checks

The follower maps rank 0's ids, so a placement cannot drift by construction; the follower still checks each begin's
ids (all of one part, the twin's all of the other, `KvShardPlacement`), and every `TF_GLM_KV_SHARD_CHECK` decode
rounds rank 0 sends a `layout` step with a digest of every live stream's slot, owner part, page ids and twin ids; the
follower compares its own and a difference ends the pair (`KvShardLayout`). Off, no step is sent.

## Telemetry

`/health` `kv_shard` and `/metrics` `tensorfold:kv_shard_*`, from `core/lanes/kv_parts.zig` (the DeepSeek port's,
as the old GLM port lifted it): parts, pages a part, each part's free, live, twin pages and live streams, and the
counters (owner rounds and bytes, twins made and given back, carries and their bytes, resumes refused, checks).
Absent when off.

## Old pieces to new files

| Old (`glm_cuda/`) | New (`glm/`) |
| --- | --- |
| `cuda_shard.zig` switches, `budget`/`plan`/`fit`, `refuse`/`why` | `cuda_kv_shard.zig` (`cuda_shard.zig` here is the TP weight share, untouched) |
| `cuda_pool.zig` parts, twins, `digest` | `cuda_kv_shard.zig` `Parts` (two `kv_pool.Pool`s), `cuda_pages.zig` sharded leases, `digest` |
| `cuda_state.zig` `initShard`, views without latent | `cuda_state.zig` `Paged.lat*`, `State.ltable`/`lpages`, `onPages` |
| `cuda_segments.zig` glue planes | none: latent tables instead of glue rows |
| `cuda_pool_capacity.zig` `sharded` | `cuda_pages.layout` (3,542 B a token) and `attach`'s trim |
| `cuda_lanes.zig` `Note.place`/`.layout` | `cuda_follow.zig` begin's twin ids, `Kind.layout` |
| `cuda_shard_decode.zig` + `glm_shard.cu` counts/select | `cuda_kv_shard_attn.zig` (exchange plans, owner attention), no kernel |
| `cuda_shard_prompt.zig` + `glm_shard.cu` fetch kernels | twins + owner-computed prompts (no fetch) |
| `cuda_shard_keep.zig` | `cuda_kv_shard_keep.zig` (home, carry pieces) + `cuda_snapshots.zig` |
| `tools/glm_shard_emulate` | not ported: no new kernel to emulate |
| `tests/cuda/glm_shard_check.zig` | `tests/cuda/glm_kv_shard_check.zig` (`tf-cuda-test glm-kv-shard-check`) |
| `core/lanes/kv_parts.zig`, `/health`, `/metrics` | the same, lifted again |

## What the host can and cannot verify

Host (no GPU, no nvcc): the switches and refusals; the fingerprint agreement; the part pool and placement against a
model; the latent tables (owner, twin, trash) and the page layout's addresses; byte-identical layouts and launches when
off (the existing recorded-launch tests, unchanged); the exchange plans (who sends which rows to whom, both ranks
pairing every send with a receive); a two-rank host model of owner-computed attention over a reference attention
(bit-equal); the snapshot plans' parts with and without latent rows and the carry's pieces on both ranks; the layout
digest; the telemetry rendering. Nothing here builds or runs a kernel: no kernel is added.

GPU only (the check program, below): that the NCCL send/recv groups pair and do not deadlock beside the gathers and the
side streams; that sharded replies are token-identical to unsharded at 1-8 streams with DFlash2, with twins and
owner-computed prompts, resumed states and carries; the memory actually freed; the exchange cost.

### The GPU check program

`zig/tests/cuda/glm_kv_shard_check.zig`, built into `tf-cuda-test` (`zig build compile-cuda-tests
-Dtarget=aarch64-linux-gnu` compiles it here; `-Dnvcc` or `-Dfatbins` on the Sparks makes a runnable one). Both ranks,
the same id file (rank 0 writes it, rank 1 reads it), the recipe's NCCL environment:

```
# Spark A (rank 0) and Spark B (rank 1); sharding, E4M3 and pages are set by the check itself
tf-cuda-test glm-kv-shard-check <ckpt> 0 2 <idfile>
tf-cuda-test glm-kv-shard-check <ckpt> 1 2 <idfile>
```

It serves one sharded paged stream (with a live stream in the other part) against an unsharded twin on a whole window
in the same engine pair, byte for byte (first token, prompt logits, then 8 verify windows of 1 to 16 rows with keeps),
over seven paths: a twin in part 0 and in part 1; owner-computed prompts in part 0 (2,049 tokens) and part 1 (9,000,
past the dense limit, index split from 2,048); a state kept at 4,096 resumed at home with a twin, resumed in the other
part without one (the carry), and kept in part 1 resumed in part 0 with a twin (a carry to both ranks). Pages are
NaN-filled before use (TF_KV_POISON's fill), a layout digest every 4 rounds; it fails if a path was never reached (owner
layers, owner prompts, twins, carries, checks all above 0) or a part holds pages at the end. Caveat: its reference is
an E4M3 whole window; if that differs from paged E4M3 for reasons of its own, `glm-paged-check` (unsharded) shows it.

Beyond it, on the served recipe: `TF_GLM_KV_SHARD=1` with `KV=fp8 KV_PAGES=on PARALLEL=4..8 DRAFTER=dflash2`
against the same requests unsharded (greedy, as the old port's g2: every reply identical), `/health` `kv_shard`
counters moving, `tensorfold plan`/the start-up line's pages about 1.8x, and the decode cost at 1, 2, 4 and 8 streams
(the old engine measured -1.4% to -3.7% with one row a stream; DFlash2 windows exchange up to 16 rows a stream).

### Host tests

`cuda_kv_shard.zig` (switches, refusals, geometry, latent tables, placement, parts, digest, 5 mutants),
`cuda_kv_shard_pages_test.zig` (sharded layout and pool on a recording driver: arenas, both tables, twins, follower
map, digest, poison, the kept plan's latent parts, home seating through the lanes), `cuda_kv_shard_attn_test.zig`
(the two-rank threaded model of owner-computed attention, 150 random windows, 2 mutants), `cuda_kv_shard_keep.zig`
and `cuda_kv_shard_keep_test.zig` (carry sides and pieces, `moveParts`/`copyOut` on a recording driver, 3,000 random
admissions on two ranks, 2 mutants), `cuda_disk_shard_test.zig` (kept states through their disk image on two
ranks of host memory over both tiers: homes 0 and 1, carries, the store image against an unsharded state's, cross
reads, the format words, `CARRY=0`, 4 mutants), `cuda_ranks_test.zig` (the switches' read and fingerprint),
`core/lane_host_status_test.zig`, `server/metrics.zig`, `server/status_routes.zig` (telemetry). All in
`test-cuda-host` (the GLM module's test block) and runnable alone with the module's `--test-filter "kv shard"`.
