# T10: a failed disk recall under `TF_KV_SHARE` costs a cold pass, not the lane

**Branch:** `task/T10-recall-fallback`, one code commit (`3f03ed6`) plus this report, on top of
`eccbe9b` (T08b's last). Nature: new design work against the merged tree — no upstream sha carries a
recall fallback (checked `tf-at-share` `389769733`, `tf-at-radix` `a3a53dbe3`, `tf-main`: no
`disk_recalls_failed`, no poisoned-id mechanism anywhere), so everything here follows the task's brief
against T08/T08b's disk tier.

## What was done

### Piece 1 — rank 0 stays consistent (`prompt_radix.Store.recall`, `rows_read`, `rows_store`)

A failure of the recall's read (`rows.read.readPages`: the family's `blank_pages`, any `scatter`, or
the seal that rides the first scatter of the state's own file) now costs what the task asks:

- **The pages go back** — the old catch did `pages.deinit` without releasing the pool's holds, so every
  failed recall leaked its pages into the pool or part for good. The catch now releases each id through
  the tree's `Pages` vtable (the parts' release under `TF_GLM_KV_SHARD`), then deinit.
- **The blank state is dropped on every rank** — the existing path already did this
  (`rows_read.readInto`'s `errdefer f.drop` → the family's drop → rank 0's `dropState`, whose
  `e.dropped` flush reaches the followers with the next step); unchanged, now asserted by the tests.
- **The tree is unchanged** — nothing was inserted yet; the catch returns `found` before any insert.
- **The disk state is marked unreadable for this process** — `rows.State` gains `unreadable`
  (`rows_store.markUnreadable`, set by name so a `bad()` that removed states during the failed read
  cannot mark the wrong one). `rows_store.best` and `rows_select.choose` skip such states, so the next
  request for the same prompt does not retry the read in a loop; the request runs as a cold pass,
  exactly as if the disk had no such state. The files stay on disk (a restart clears the mark);
  a refused read (`SnapshotSize`, `DiskWords`, ...) still goes through the existing `bad()` and is
  deleted outright, which is that path's stronger form of the same guarantee.
- **A counter and a warning** — `pc.Counts` (both stores' counters) gains `disk_recalls_failed`;
  a failed recall increments it once and logs one `prompt cache:` warning naming the error and the
  token count. The flat store's `rows_read.recall` marks and counts the same way on a failed read
  (its `select.choose` skips the mark, so `/v1/imprint`'s `where` does not promise it either).

Interpretation note (the one judgement call): "exported where the store's counters are" — the counters'
only export today is the `prompt cache:` report line and the `Counts` struct itself; I put the field
beside the others in `pc.Counts` and did **not** change the report line, because TASK.md requires
"with no failure every path byte-identical to this head" and an extra always-present field in the
per-pass line would break that. The counter is public beside the others and asserted by the tests; if
the recipe wants it in the per-pass line (or a stats route), that is a one-line follow-up.

### Piece 2 — a failing follower no longer ends the rank (`cuda_disk_ranks.step`)

Today any error out of a follower's disk op propagated out of `step` → `apply` → the `follow` loop →
the rank stopped, and rank 0's next link step said `rank 1 stopped (SnapshotSize); this rank stops too`
— the incident. Now:

- **A failing op poisons its id** (`poison`): the id goes into `Engine.disk_failed` (new field), the
  rank's half-made state is dropped (`dropState`, the existing path), one warning logs the error and
  the id, and `step` returns success. The step loop runs on.
- **Put-style ops for a poisoned id are skipped** (the leader keeps sending until the seal), so no
  error storm.
- **The seal answers with a status word**: the seal op's handler no longer errors out — it seals (or
  records the failure), then every rank's status word crosses the link rank order (a broadcast each
  rank roots in turn). Rank 0's `ranks.seal` collects them right after the seal step and fails with
  `error.DiskRankFailed` when any rank says its ops failed for the id or its seal refused the words.
  That propagates through `scatter` → `readInto` → `recall`'s catch → the cold pass above. The seal is
  the collection point the task named; in the read-back it rides the state's own file, the last piece,
  so put-failures during the row pieces are covered by it.
- **Take-style ops (`take`, `lat_take`, `slot_take`, `page_take`) carry the status past their bytes**:
  the answer is now `depth + 4` bytes (payload, then one status word). Rank 0 reads the status after
  each piece and fails the op with `DiskRankFailed` instead of gathering what a broken rank's copy
  sent; the follower completes its side of the broadcast either way, so the link never desyncs. This
  is what makes piece 4 hold: a save whose follower side fails is skipped on rank 0 (`imprint: saving
  N tokens skipped`, already non-fatal) instead of writing another rank's garbage rows into a block
  file. The link room per piece drops by 4 bytes (`ids.len * 4 - status_bytes`), both sides changed
  together — no compatibility surface (one binary serves both ranks).
- Only a step that breaks the protocol itself (a `BadStep`, a link failure) still ends the rank, as
  before.

### Piece 3 — the cold pass

`recall` returns the tree's own match (or none), `begin` plans and prefills from scratch: nothing
adopted, nothing kept from the failed state. Byte-identical to a store with no such disk state, once
the mark is set.

### Piece 4 — saves

Rank 0's side was already non-fatal (`imprint: saving N tokens skipped`, `counts.skipped`); the
follower side of `lat_take` / `slot_take` / `page_take` / `take` now goes through `answered`, so a
failing take poisons the id, answers with its status (the gather fails on rank 0 → the save is
skipped) and the step loop runs on. A follower put-failure during a save's gather (the sharded
`key_put`/`lat_put`/`slot_put`/`page_put` writes on a read-back are the puts; a save's writes are
rank 0's own plus the prefixes) is covered the same way.

Unsharded and sharded alike: the same ops run under both layouts, and the pages-release goes through
the `Pages` vtable (one pool's or the parts'). With no failure every path is byte-identical to the
previous head: the op bodies themselves are unchanged (moved into `*Body` helpers verbatim), the
link's bytes change only on a take's answer (+4 status bytes) and the new seal reply (4 bytes a rank,
at the seal only), and the store-side code differs only on error paths.

## Tests (all run on this head, Zig 0.17.0, `-j8`)

New host tests:

- `prompt_radix_disk_test.zig` (the T08 fake pool + tiny image, now 264 pages so a state has a block
  piece and its own file — two reads): a shared `failedRecall` harness asserts, for each injected
  failure — scatter fails midway (the block piece), a failing seal (the state's own file, where GLM's
  seal runs), a failing `blank_pages` (its pages held then given back, as the image's cleanup does) —
  that `begin` returns a cold plan (`from` 0, nothing adopted), every page is back
  (`pool.available()` whole), the tree unchanged (no snaps, no held pages), the blank dropped
  (`f.live` 0), `disk_recalls_failed` 1, the state marked `unreadable`, and a second `begin` for the
  same prompt does not read the disk again (no blank, no scatter, no read, the counter still 1).
  The fake's `blankPagesFn` was also fixed to append its page ids only once they are all held (the
  real `cuda_glm_disk.blankPages` already did; the old fake left uninitialised ids in the caller's
  list on failure, which the new release path would have released as garbage).
- `cuda_disk_shard_test.zig` (the two-rank rig, `ncclBroadcast` now mocked and the link buffers
  registered so a `follow.Link` runs on it; event mocks added for the link's async uploads):
  `GLM disk steps: a follower's failed op poisons the id, answers the seal with a failure and the loop
  runs on` — a blank stands, a write past the blank's front fails (`DiskOrder`), the half-made state
  is dropped and the id poisoned; the same write again is skipped, not an error; a take for the
  poisoned id still crosses the link and answers status 1; its seal answers 1; a healthy blank's seal
  answers 0 and stands sealed. Rank 0's side of the collection is exercised through the real
  `Link.from`/download path (the `Leader.disk` send needs the kernel module, not constructible on a
  host — the collection reads the same link buffer the test drives).

| command | result |
| --- | --- |
| `zig build test-rows-host -Drows-filter= --summary all` | 429/429 pass (the 3 new failure-recall tests included; the benign `lane host: a round failed (FinalsFailed)` stderr writer exits 0, as T08's report notes) |
| `zig build test --summary all` | 445/445 steps succeeded, 200/202 tests (2 skips, the same pre-existing skips) |
| `zig build test-cuda-host -j8 --summary all` | 274/274 steps succeeded, 0 failures (the shard-test run includes the new follower test; the stderr "failed command" lines are the benign writers, each binary exits 0) |
| `zig build test-glm-pages-host --summary all` | 4/4 steps succeeded |
| `zig build test-glm-snapshots-host --summary all` | 27/27 pass |
| `zig build test-glm-kv-pages-host --summary all` | 31/31 pass |
| `zig build test-glm-startup-host --summary all` | 3/3 steps succeeded |
| `zig build test-kv-pool --summary all` | 35/35 pass |
| `zig build test-cuda-disk-open-host --summary all` | 5/5 steps succeeded |
| `zig build test-rows-prefetch-host --summary all` | 35/35 pass |
| `zig build test-window-rows-host --summary all` | 4/4 steps succeeded |
| `zig build test-inc4-rows-host --summary all` | 4/4 pass |
| `zig build test-learn-recovery --summary all` | 441/441 pass |
| `zig build test-learn-warning --summary all` | 3/3 steps succeeded |
| `zig build test-nemotron-learning-host --summary all` | 20/20 pass |
| `zig build test-proposal-rows-host --summary all` | 5/5 pass |
| `zig build test-qwen27-prompt-rows-host --summary all` | 8/8 pass |
| `zig build test-server-cpu -j8 --summary all` | 5/5 steps succeeded |
| `zig fmt --check build.zig zig` | clean |
| `/home/user/invt/glm-kv-share/cuda-compile-check.sh` | **CUDA COMPILE OK** (576 s) |

No NOTES.md failure remains on this tree (T07 fixed the old ones); nothing was skipped or weakened.

## Adapted, and why

- **The one existing expectation changed**: `rows_refusal_test`'s "a link failing keeps it" cases
  (`ScatterFailed`, `OutOfMemory` with `mine = true`) asserted `best` still selected the state after a
  failed recall — the retry-loop the task removes. The states still exist and their files stay; the
  one line now asserts `best` is null (not read again in this process) for every failed read, refused
  ones forgotten, the rest marked. The `counts.failed`, state-count and file-existence expectations
  are untouched.
- **Marking is unconditional on a failed read** (not only follower-reported ones): a transient OOM or
  link hiccup also marks the state, trading one state's recall for the guarantee that no failure
  retries in a loop. A restart clears it. Refused reads keep their stronger existing behaviour
  (`bad()` deletes the files).
- **The status rides the take's answer** rather than a separate step: the ops that already cross the
  link for an answer get the status for free there, and a save's gather fails at the piece that broke
  instead of after the whole state was gathered.
- **`pc.Counts` for the counter**: both stores' counters live there and the paged store's `tree.counts`
  is that struct; the flat store counts the same field at its own recall site.
- **Fake rig fixes**: `page_count` 16 → 264 (a 256-token state is what has a block piece and its own
  file, so "midway" and "seal" are distinct reads), `pass` fills whole pages only, `blankPagesFn`
  holds into a local before appending (matching the real image's contract).

## Left / uncertain

- **The seal is the only collection point**: a follower op failing *after* its seal answered (the own
  file's fixed-part and prefix writes) is poisoned and logged but not reported to rank 0 — the recall
  completes with that rank's fixed parts unwritten. Collecting after every piece would add world-1
  broadcasts per piece (~2,000 for a 600k state); not worth it for a window that is reached only by
  device-level failures on the follower. The GPU checks below should watch for it.
- Rank 0's side of the seal collection (`Leader.disk` → the `put_words` kernel) is compile-checked
  (`CUDA COMPILE OK`) and read, not run on a host; the follower side is fully tested.
- `Engine.disk_failed` grows by one entry per failed id per process (u64 keys, dropped states' ids
  never removed) — bounded by failure count, freed at `deinit`.
- The `atc` CLI is not installed in this environment (as in T08's report): no board/claim/`atc
  overlap --json` could run here; run `atc overlap --json` before the pull request and link the issue.

## GPU checks needed (the DGX boxes)

1. The incident replayed as a fault drill: a recall whose rank 1 fails inside (a `TF_GLM_*` fault
   injected at a follower disk op, or a deliberately shape-mismatched store) — expect one
   `rank 1: a disk step for state ... failed (...)` warning on the follower, rank 0's
   `recalling N tokens from disk failed (DiskRankFailed)` warning, the request completing as a cold
   pass, both ranks serving the next request, and `disk_recalls_failed` 1.
2. The same prompt again after the failure: no second read attempt (the mark), a cold prefill, and
   the store's files still present on disk.
3. A save whose follower take fails: rank 0 logs `imprint: saving N tokens skipped (DiskRankFailed)`,
   the lane and both ranks live on, and no block file holds the half-gathered state.
4. A healthy restart after the faults: the marked state readable again (the mark is per-process),
   its recall succeeding and resuming with the cold reply's hash.
5. Sharded pair (`TF_GLM_KV_SHARD=1`): the same drill with the state's home on rank 1 — the latent
   rows' `lat_take` status path, and a recall into the other part than the save's.
