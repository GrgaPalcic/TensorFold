# T05d: a kept state's pages stay in one part

Branch `task/T05d-one-part-keeps`, one commit (`2b7a09b`) plus this report. Recipe-side (no upstream sha): the
bug is from the measured pair in the task (share-port `f1cd3b7`, `TF_KV_SHARE=32`, `TF_GLM_KV_SHARD=1`, TP=2).

## What was done

The gate's cold pass broke T05's "a leaf's pages are always in one part": a skipped resume seated wherever
`Parts.choose` put it, and its keep inserted its tail under the kept prefix's node of the other part — a kept
state whose path spans both parts. The next resume adopted the mixed path and the follower's placement check
refused it. Three pieces, on top of T05/T05c:

- **A skipped resume's cold pass seats at the kept prefix's part.** `cuda_share.gate` now returns
  `?u8` — the kept prefix's part (`Pages.partOf(adopted[0])`), remembered before the begin's holds go back —
  and `cuda_lanes.opening` hands it to the new `Lanes.seatCold`, which uses it as the seat's `home` exactly as
  a kept state's home is today ("has room" is the parts' `fits`, so reclaiming the part's own idle leaves
  counts). The twin is borrowed as usual. When the part lacks room for the whole pass, `choose` takes over and
  the pass lands wherever it fits. Unsharded share is untouched: the gate returns null without the parts, and
  `seat` forwards `seatCold` with null.
- **A keep never builds a path across the parts** (`prompt_radix.zig`, rank 0, every build mode). `Store`
  takes an optional `Parting` (`onePart` + a `refused` note; `cuda_prompt_pages.Adapter` wires it through
  `kp.Parts`' calls, as it already wired the tree's pages). In `keep`, after the tree's prefix pages and the
  pass's tail are joined in `path`: all in one part — insert as before; else, when the pass is longer than the
  matched prefix and nothing branches off that chain (`Tree.alone`), `Tree.dropChain` replaces it whole — the
  older states freed, the nodes pruned, their pages released — and the keep retries; else the keep is refused
  (`held = false`, the tree unchanged) and counted: `Parts.refused` → `cuda_native.partsRefused` →
  `kv_shard_keeps_refused_total` (new `Counts`/`Stats` field, exported with the other `kv_shard_*` metrics),
  one debug line. `Tree.evict` now factors through the new `dropEntry`.
- **`begin` verifies the invariant before handing pages out** (`cuda_share.begin`, rank 0, every build mode):
  sharded, the adopted ids must lie in one part — on a violation nothing is adopted (the begin's holds go back
  through the parts), `owner.pending` is cleared, the state is evicted (`kp.Prompts.forget`, a new vtable call
  over `Store.forget`, so a later pass does not ask for it again) and one warning logged. A cold pass is
  correct; a cross-part resume is not.

With the switches unset every existing path is byte-for-byte as before: the gate's skip/adopt decision, the
unsharded share path, and the store's keep when `parts` is null are unchanged.

## Tests

New in `cuda_kv_shard_pages_test.zig`: the `Stored` fixture — a real `radix.Store` over the sharded pages
fake's parts, wired as `cuda_prompt_pages.Adapter` wires it on rank 0 (the parts as the tree's pages, the
keep's `Parting` over `kp.Parts`, each part's own reclaim hook, the family's save/restore through
`native.snap_restore` into the share's owner) — and four wiring tests:

1. the GPU's exact sequence: 4,096 kept in part A (8 pages); the 126,407-token prompt adopts, the gate skips
   it and remembers part A; the cold pass seats there with a twin; kept at 122,880 — every page of the kept
   path in part A, `keeps_refused` 0; the next request adopts the 122,880 state and the follower maps the
   begin step, the digests agreeing;
2. part A full (the cold pass needs more pages than the prefix's idle ones free): the cold pass seats in part
   B, and the keep at 122,880 replaces the part-A prefix chain whole — the new path all part B, the old pages
   back, `keeps_refused` 0, the 4,096 state gone;
3. the prefix node has another child: the cross-part keep is refused and counted, the tree unchanged, both
   kept states still resume;
4. a hand-built cross-part kept state (the parts' check off for one keep, as the bug's tree had): `begin`
   adopts nothing, evicts it, and the next pass runs cold.

Also new in `native/cuda_prompt_pages.zig`: the store-level replace (nothing branched off the chain) and
refuse (the chain has another child) over the sharded pool stand-in, both counted.

| Command | Result |
| --- | --- |
| `zig build test-cuda-host --summary all -j8` | 274/274 steps succeeded; 621/622 tests pass (1 skipped, the pre-existing skip), 0 failed, no leaks, no error logs — includes the glm module unfiltered, so all four new wiring tests and the two adapter tests run there |
| `zig build test-glm-kv-pages-host --summary all -j8` | 4/4 steps, 30/30 tests pass |
| `zig build test-glm-pages-host --summary all -j8` | 13/13 tests pass |
| `zig build test-kv-pool --summary all -j8` | 5/5 steps succeed (`kv_pool` 29/29) |
| `zig build test-lane-api-host --summary all -j8` | engine_api module 417/417 (`prompt_radix`/`prompt_radix_tree` with the new `Parting`, `alone`, `dropChain`, `forget`) |
| `zig build test-shared-lane-regressions-host --summary all -j8` | 3/3 steps succeed (`kv_parts`' new `keeps_refused` stat) |
| `zig fmt --check build.zig zig` | pass |

First runs caught three of my test's own defects (the follower fake's table length, a part that `fits` via
reclaimable pages, a driver table too short for a 248-id lease) — fixed in the tests/driver, nothing in the
ported code needed a change for them.

## Adaptations, and why

- **The eviction logs a warning, not an error.** The design (and my first cut) used `std.log.err`; the host
  test runner fails a step on any err-level log, and this path must run in `test-cuda-host` (the task's fourth
  test). Demoted to `std.log.warn` — the same call T07 made for `TF_GLM_MM_STAGES`'s note. The behaviour is
  unchanged: nothing adopted, the state evicted, one line logged; the follower-side hard stop (the placement
  check, `cuda_follow`'s digest error) still logs an error.
- **The store's `keep` offers the mixed path to the `Parting`, not a per-node check.** The invariant is
  checked once on the whole path the keep would insert (prefix pages joined with the tail), because that is
  the only place both parts' pages are known together; `insert` itself stays part-blind (and is how test 4
  hand-builds the bug's tree).
- **`forget` rides `kp.Prompts.VTable`** (one new call, both implementors updated): the store's eviction is
  the family's business — `cuda_share` cannot reach the radix store directly, and the same hook serves any
  family's store.

## Left, and uncertain

- The replace/refuse choice compares only "the pass is longer than the matched prefix" (`got < mine.len`);
  an equal-length cross-part keep is always refused, never replaced. The task's rule ("the new state is longer
  and newer") reads the same way; a shorter pass keeping over a longer prefix of another part cannot happen
  through the gate (the resume would have adopted or skipped), but a direct `keep` at a shorter `at` over a
  longer other-part prefix is refused.
- `Tree.alone` walks to the root on every refused keep; the chains are short (kept states a few pages apart),
  and the path only runs on the mismatch, so this did not seem worth a per-node mark.
- Nothing else known. The gate returns the part only when it skips; an adopting resume's home is still its
  adopted pages' part, and a plain kept state's its snapshot's home.

## GPU checks needed later (the pair, both switches on, TP=2)

- The task's sequence at the default `TF_GLM_KV_SHARE_UNTWINNED_PCT`: the 126,407-token request after the
  skipped 4,096 resume keeps at 122,880 without the `a stream's page ids from rank 0 do not lie in one part`
  error, and the following 126,408-token request resumes (cold near 70 s for the 126k pass, near 330 s for the
  610k one, `kv_shard_resumes_skipped_total` 2 as before).
- `kv_shard_keeps_refused_total` stays 0 in that run (the seat-at-the-prefix's-part fix should make every keep
  one-part); a non-zero count with the debug line names the conversation to look at.
- `TF_GLM_KV_SHARE_UNTWINNED_PCT=100` (round 3b) unchanged: no skips, keeps one-part as before.
- A cross-part kept state, if it ever appears, logs the warning once and the request runs cold — the pair must
  still complete (no placement error, no digest difference).
