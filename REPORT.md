# T01 report: `prompt_radix` learns a lookahead and a planned grid (upstream a2d27dac7, a3a53dbe3)

## What was done

Ported upstream PR #1659 (commits `a2d27dac7` and `a3a53dbe3`) onto the pin:

- `zig/src/core/prompt_radix.zig` — the store's rules gain `lookahead` and use the existing `planned`/`grid`:
  a store asserts nothing at init any more; rules it cannot keep (`page == 0`, or `planned` with a grid that is
  0 or not a multiple of `page`) return `error.BadPromptRadixRules` in every build (`a3a53dbe3`). The step a pass
  cuts and resumes on is the grid for a planned store, else the page; `past()` is the lookahead a resume leaves
  (at least one token left to prefill); `find` and the keep/held paths match a state together with the tokens past
  it, so a page whose next token differs is not shared.
- `zig/src/core/prompt_radix_tree.zig` — `deepest` takes the store's step and only stops on multiples of it;
  page matching and tree insertions carry the lookahead.
- `zig/src/core/prompt_radix_test.zig` — seven new tests: the golden test (lookahead 0 and no grid: resume and
  evict exactly as the store before the PR kept them), lookahead 1 resume and sharing, planned grid marks/keeps,
  the GLM shape (grid 16, page 4, lookahead 1) and the rules-refusal test.

The first commit of the task (an earlier run cut off) had applied both patches with `git apply --reject` and
resolved the rejects; I verified that state rather than starting over:

- No `*.rej` or `*.orig` files remain.
- All 20 patch hunks landed: `a2d27dac7` 9 hunks in `prompt_radix.zig`, 1 in `prompt_radix_test.zig`,
  8 in `prompt_radix_tree.zig`; `a3a53dbe3` 1 hunk in `prompt_radix.zig`, 1 in the test file.
- All three files are byte-identical to the reference tree `/home/user/invt/glm-kv-share/tf-at-radix`
  (upstream at exactly `a3a53dbe3`), as the task requires: `diff` clean on all three.
- The recipe's behaviour is untouched: the PR touches only the three core files; nothing else in our tree
  (recipe patches 0108/0109/0110, the families, `kv_pool`) differs from the pin because of this port.

With `lookahead` unset (0) and no grid, every existing path is unchanged; the PR's own golden test pins that
and passes. New switches stay off by default: no caller passes `lookahead` or `planned` yet (the two
`radix.Rules` literals in `nemotron/hip/prefix.zig` and `qwen3_5/backend/hip/engine/hip_prefix.zig` are
unchanged, matching `tf-at-radix`).

## Test commands run (host, Zig 0.17.0)

- `zig build test-kv-pool --summary all -j8` (the task's target; compiles `prompt_radix_test.zig` through
  `kv_pool_checks` → `kv_pool_store_test`): **29/29 tests pass** (23 in the pool module + 6 in `kv_pool_checks`).
- Direct run of the store's own tests (module build of `engine_api` with a root importing the three files):
  **417/417 pass**, including all 15 `prompt_radix_test` tests — the golden, both lookahead tests, both planned
  tests, the tree-step test, the GLM-shape test and the rules-refusal test — plus `prompt_radix`'s and the rest
  of engine_api's suite (`rows`, `prompt_cache`, lanes), which the change must not disturb.
- `zig build test --summary all -j8`: passes except the pre-existing failures listed below; the store-relevant
  GLM/KV targets pass on top of the port: `test-glm-kv-pages-host` 30/30, `test-glm-window-contract-host` 17/17,
  `test-glm-prompt-settings-host` 8/8, `test-glm-pages-host`, `test-glm-page-contract-host`,
  `test-glm-prompt-profile-host`, `test-glm-index-host` all green.
- `zig fmt --check build.zig zig`: **pass**.

## Pre-existing failures (present with the port stashed, i.e. without it; not touched by this task)

These come from the recipe patches applied on top of the pin (`git stash` → same failures), not from the port:

- `test-cuda-host`: `draft_sources_cuda_test` "CUDA GLM distinguishes its checkpoint head from loaded DFlash2"
  (1/112), `glm_fp8_pages_host` "GLM FP8 pages: the prompt cache's store keeps paged E4M3 states…" (1/15) and
  `cuda_mm` "TF_GLM_MM_STAGES" logs one error (the recipe's 0100 test calls `stageIndex("4")`, whose
  `std.log.err` the test runner records; 227 pass).
- `test-glm-snapshots-host`: 4/26 fail — recipe patch 0094 (4,096-row prompt chunks) changed
  `stateBytes`/`snap_grid` sizes; the snapshot-sizing tests still expect the pin's numbers
  (e.g. expected 222298112, found 260046848).
- `test-glm-prompts-host`: 1/7 fails — recipe patch 0095 raised `dflash.MAX_CTX` to 4,096, so
  `promptRun(0, 0, 0, 5, 5, 2044)` no longer returns `DrafterContextTooWide` (the test line at the pin was
  written for `MAX_CTX == 2048` and no recipe patch updated it).

Per the rules I did not weaken or change these; they belong to the recipe's owners. If they are known-stale on
the recipe side, the 0094/0095 follow-ups should update the affected expectations.

## What is left or uncertain

- Nothing in the port itself. All three files match `tf-at-radix` exactly; both upstream shas are in.
- Later tasks: T02 (GLM's family turns the rules on) builds on the `lookahead`/`planned`/`grid` rules and
  `error.BadPromptRadixRules`; the GLM-shape test here (grid 16 = 4,096-row `snap_grid` / 256-row pages at
  real sizes, lookahead 1 = the MTP head's row) is the template.

## GPU checks needed

None (core only; no family turns the rules on yet), as the task says. On the DGX boxes the existing GLM HIP/CUDA
runs keep the rules unset, so no new device check is introduced by this port.

## Commits

- `d524e6d` core prompt_radix: a lookahead in the tree and a planned grid (upstream a2d27dac7, a3a53dbe3)
