"""Fused-group semantics: a rank's share of a fused q|k|v tensor is three disjoint runs."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensorfold.families.glm5_next.cuda.split import rule       # noqa: E402
from tensorfold.families.glm5_next.cuda.split import fused_rank_runs  # noqa: E402


def test_fused_groups_replicate_whole():
    for name in ("model.language_model.layers.0.self_attn.qkv_proj.trellis",
                 "model.language_model.layers.0.self_attn.qkv_proj.svh",
                 "model.language_model.layers.0.self_attn.conv1d.weight"):
        assert rule(name) == "rep", name        # a byte cut cannot express the three disjoint runs


def test_fused_rank_runs_partition():
    per, world = 8192, 2
    runs = {r: fused_rank_runs(3 * per, 3, r, world) for r in range(world)}
    # each rank holds one run per block; the blocks' unions cover the tensor exactly once
    covered = []
    for r in range(world):
        assert len(runs[r]) == 3
        for j, s in enumerate(runs[r]):
            assert s.start == j * per + r * (per // world) and s.stop == j * per + (r + 1) * (per // world)
            covered.append(set(range(s.start, s.stop)))
    flat = [i for c in covered for i in c]
    assert len(flat) == len(set(flat)) == 3 * per


def test_fused_rank_runs_reassemble():
    """Slicing rank0 and rank1 runs and concatenating per block reproduces the original rows."""
    rows = np.arange(24, dtype=np.int64).reshape(24, 1)      # 3 blocks of 8, world 2
    out = {}
    for r in range(2):
        out[r] = np.concatenate([rows[s] for s in fused_rank_runs(24, 3, r, 2)])
    assert np.array_equal(out[0], np.array([0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19])[:, None])
    assert np.array_equal(out[1], np.array([4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23])[:, None])


def test_fused_rank_runs_reject_odd():
    with pytest.raises(ValueError):
        fused_rank_runs(3 * 7, 3, 0, 2)


def test_qkv_trellis_tile_runs_in_bounds():
    """A rank's trellis tile columns: [K/16, 3N/16, words] with N = 24576, world 2."""
    trellis_shape = (384, 1536, 80)
    for r in range(2):
        for s in fused_rank_runs(trellis_shape[1], 3, r, 2):
            assert s.start % 8 == 0 and (trellis_shape[1] - s.stop) % 8 == 0   # Hadamard 128-blocks: tiles %8
