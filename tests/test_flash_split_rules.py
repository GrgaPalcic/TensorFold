"""Split-rule tests for the RED-SNOW dense-EXL3 port.

Names tier: every tensor in the real index resolves to exactly one rule (needs
REDSNOW_INDEX pointing at the pack's model.safetensors.index.json; skipped if absent).
Shapes tier: Hadamard-block divisibility for every projection class with synthetic
shapes (runs anywhere).
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensorfold.families.glm5_next.cuda import split as sp  # noqa: E402

INDEX = os.environ.get("REDSNOW_INDEX", "")

def test_real_index_coverage():
    if not INDEX or not Path(INDEX).exists():
        pytest.skip("REDSNOW_INDEX not set")
    weight_map = json.loads(Path(INDEX).read_text())["weight_map"]
    kinds: dict[str, int] = {}
    for name in weight_map:
        kind = sp.rule(name)
        assert kind in ("row", "col", "rep", "dim1", "drop"), name
        kinds[kind] = kinds.get(kind, 0) + 1
    # the load-bearing expectations from the coverage pass
    assert kinds.get("col", 0) == 0                                 # no plain col tensors in this pack
    assert kinds["drop"] == 351                                     # pack head 4 + vision 347
    assert kinds["dim1"] == 24906                                   # out-linear trellis (gate/up/q_b)
    assert kinds["row"] == 50074                                    # down/o_proj trellis+suh, out svh, plain f_b/g_b/b/kv_b
    assert kinds["rep"] == 75527                                    # suh(out)/svh(in), q_a/kv_a/eh_proj, norms, router, indexer
    assert sum(kinds.values()) == 150858                            # the full index, nothing unaccounted


def test_real_index_no_unknown():
    if not INDEX or not Path(INDEX).exists():
        pytest.skip("REDSNOW_INDEX not set")
    weight_map = json.loads(Path(INDEX).read_text())["weight_map"]
    for name in weight_map:
        try:
            sp.rule(name)
        except ValueError as e:  # noqa: PERF203
            pytest.fail(f"rule() failed on {name}: {e}")


def test_hadamard_shapes_pass():
    k16, n16, w = 16, 16, 80                      # a 5-bit mul1 tile: [K/16, N/16, 16*bits]
    shapes = {
        # layer 0 fused KDA q|k|v: N = 3*64*128 = 24576
        "layers.0.self_attn.qkv_proj.trellis": [256, 1536, w],
        "layers.0.self_attn.qkv_proj.suh": [4096],
        "layers.0.self_attn.qkv_proj.svh": [24576],
        "layers.0.self_attn.qkv_proj.mul1": [],
        # dense MLP (layers 0-2): K = hidden, N = intermediate
        "layers.0.mlp.gate_proj.trellis": [256, 128, w],
        "layers.0.mlp.gate_proj.suh": [4096],
        "layers.0.mlp.gate_proj.svh": [2048],
        "layers.2.mlp.down_proj.trellis": [128, 256, w],
        "layers.2.mlp.down_proj.suh": [2048],
        "layers.2.mlp.down_proj.svh": [4096],
        # routed experts (mean 5-bit: per-expert widths differ; shapes same)
        "layers.3.mlp.experts.0.gate_proj.trellis": [128, 128, w],
        "layers.3.mlp.experts.0.gate_proj.suh": [4096],
        "layers.3.mlp.experts.0.gate_proj.svh": [2048],
        "layers.3.mlp.experts.287.down_proj.trellis": [128, 256, w],
        # DSA attention (11 layers)
        "layers.3.self_attn.q_a_proj.trellis": [32, 96, w],        # K 1536: rep anyway
        "layers.3.self_attn.kv_a_proj_with_mqa.trellis": [32, 32, w],
        "layers.3.self_attn.q_b_proj.trellis": [96, 1024, w],      # N = 64*256 = 16384
        "layers.3.self_attn.q_b_proj.svh": [16384],
        "layers.3.self_attn.kv_b_proj.weight": [16384, 512],
        "layers.3.self_attn.o_proj.trellis": [1024, 256, w],       # K = 16384
        "layers.3.self_attn.o_proj.suh": [16384],
        "layers.3.self_attn.indexer.wq_b.trellis": [16, 8, w],     # replicated, no cut
        "layers.3.self_attn.indexer.wk.weight": [2048, 1536],
        # MTP layer eh_proj: [D, 2D] replicated
        "layers.45.eh_proj.trellis": [256, 512, w],
        # head from the pack: dropped
        "lm_head.trellis": [9673, 16, 128],
        "lm_head.suh": [154880],
        # vision: dropped
        "model.visual.blocks.0.attn.qkv.weight": [3072, 3072],
    }
    assert sp._check_exl3_cut_shapes(shapes, world=2) == 12


def test_hadamard_shapes_refuse_bad_cut():
    shapes = {"layers.0.self_attn.qkv_proj.trellis": [256, 1535, 80],   # odd tile columns
              "layers.0.self_attn.qkv_proj.suh": [4096]}
    with pytest.raises(ValueError):
        sp._check_exl3_cut_shapes(shapes, world=2)
    shapes = {"layers.2.mlp.down_proj.trellis": [127, 256, 80]}          # odd tile rows
    with pytest.raises(ValueError):
        sp._check_exl3_cut_shapes(shapes, world=2)
    shapes = {"layers.0.self_attn.qkv_proj.svh": [24577]}                # scale rows off 128
    with pytest.raises(ValueError):
        sp._check_exl3_cut_shapes(shapes, world=2)
