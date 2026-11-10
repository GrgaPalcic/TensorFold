# T05c: a sharded share resume adopts only when the untwinned tail pays

Branch `task/T05c-adopt-threshold`, one commit (`7e817e8`) plus this report. No upstream sha: recipe-side,
from the measured pair in the task (a resume at 4,096 of 126,407 tokens was slower cold; at 122,880 it paid).

## What was done

Under `TF_KV_SHARE` with `TF_GLM_KV_SHARD=1` a resumed pass runs owner-computed (T05's blocker: no twin for
the adopted prefix), so a resume that adopts a short prefix and prefills a long tail is slower than not
resuming. The gate, in the sharded share path only:

- `cuda_share.gate` runs in `cuda_lanes.opening` right after `share_mod.begin` hands the adopted pages. It
  adopts only when `new_tokens × (100 − pct) ≤ adopted_tokens × pct` (`adopted_tokens` the kept state's `at`,
  `new_tokens = prompt − at`), the untwinned rate as `TF_GLM_KV_SHARE_UNTWINNED_PCT` percent of the twinned
  rate. Default 75 (a tail of at most three times the kept prefix); 100 always adopts; values outside 1..100
  are refused at start by name, on every rank (`Settings.read`, like the other `TF_GLM_*` switches). The rule
  applies to the deepest match only — that is what `begin` returns, and a skip does not fall back to a
  shallower state.
- A skipped resume runs cold as the pass would without a kept state: the begin's holds on the adopted pages
  go back through the parts (`Parts.release`, as `openWith`'s errdefer does — `Pages.give` releases through
  the one pool, which holds no pages when sharded), the tree's holds keep the state in the store and
  resumable, `owner.pending` is cleared so `opening` returns no resume, and the seat places the pass by the
  parts' choice with its twin borrowed. Rank 0 counts `kv_shard_resumes_skipped_total` (new `Counts`/`Stats`
  field `resumes_skipped`, exported with the other `kv_shard_*` metrics and in `/health`'s `kv_shard`) and
  logs one debug line.
- The decision is rank 0's alone and reaches the follower through the begin step as today: with nothing
  adopted the begin step opens the follower's pass cold. No new step field, nothing in the fingerprint.
- The unsharded share path is untouched: the gate returns without the parts, so its resumes adopt always.

With `TF_GLM_KV_SHARE_UNTWINNED_PCT` unset the default 75 changes only the sharded share path, which is the
task's point; every other path is byte-for-byte as before.

## Tests

| Command | Result |
| --- | --- |
| `zig build test-cuda-host --summary all -j8` | 274/274 steps succeeded, 0 failed (1 skipped); first run caught a crash in my new wiring test (retained a freed pool page), fixed, re-run green — the glm module's binary 252/252, my six new tests verified by name |
| `zig build test-glm-kv-pages-host --summary all -j8` | 4/4 steps, 30/30 tests pass |
| `zig build test-glm-pages-host --summary all -j8` | 4/4 steps, 13/13 tests pass |
| `zig build test-shared-lane-regressions-host --summary all -j8` | 3/3 steps succeed (the `lanes` module: `kv_parts`) |
| `zig build test-lane-api-host --summary all -j8` | 3/3 steps succeed |
| `zig build test-server-cpu --summary all -j8` | 5/5 steps succeed (`kv_shard_*` metrics, status) |
| `zig fmt --check build.zig zig` | pass |

New tests: the setting's parse (75 the default, blank/unset the default, 1 and 100 taken, 0/101/negative/
non-numeric refused); the pay rule with the task's numbers (4,096 kept against 126,407: cold; 122,880 kept:
adopted; the 610k case: cold; empty tail always pays; 100 adopts however long the tail); `Settings.read`
reads it by name and refuses outside 1..100 naming the switch; the wiring on the sharded pages fake — a
4,096-token kept state against a 126,407-token prompt is skipped (pages given back to the tree's hold only,
still in the store, `resumes_skipped` counted, the cold seat borrows a twin), a 122,880-token one adopts
(no twin, its adopted pages' part), at 100 the first case adopts, and an unsharded resume adopts always
(the gate leaves the begin's pages alone). Pre-existing failures: none — per NOTES.md every host target has
been green since T07, and everything above passes.

## Adaptations, and why

- **The give-back releases through the parts, not `Pages.give`.** `give` releases from the one pool, which
  holds no pages when sharded (a latent hazard in T05's claim-failure path, untouched here); the gate uses
  `Parts.release` as `openWith`'s errdefer does.
- **The setting rides `Settings`, not the fingerprint.** Only rank 0's value decides (the store is rank 0's);
  the follower learns the outcome through the begin step. Every rank still refuses a bad value at start.

## Left, and uncertain

- Nothing known. The gate sits before the seat, so a skipped resume waits for admission exactly as a cold
  pass does today; the kept state's part is not consulted for the cold placement (the parts choose as for any
  fresh pass).
- `Pages.give`'s sharded hazard (above) is noted for whoever touches the claim-failure path next; fixing it
  was outside this task.

## GPU checks needed later (the pair, both switches on, TP=2)

- The gate's 126,976 case: cold time back near 70 s with `cached` 0 reported for that request, and
  `kv_shard_resumes_skipped_total` incremented by 1 (`/metrics`).
- The 600k case (new 487k vs adopted 122,880: 487k > 3 × 122,880, skipped too) back near 330 s, the skip
  counted, the kept state still resumable by a shorter-tailed prompt afterwards.
- Warm cases unchanged: a resume at 122,880 of the 126,407-token prompt still adopts (tail 3,527 ≤ 3 ×
  122,880), no twin, `cached` = 122,880, time to first token near the T05 resumed numbers.
- `TF_GLM_KV_SHARE_UNTWINNED_PCT=100` on both ranks returns the old always-adopt behaviour byte for byte;
  a value like 0 or 101 refused at start by name on every rank.
- The debug line appears once per skipped resume on rank 0 at debug level.
