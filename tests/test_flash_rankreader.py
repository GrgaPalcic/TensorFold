"""RankReader end-to-end slicing on a synthetic RED-SNOW-shaped mini-pack (CPU).

Builds two shards + index with the pack's tensor forms (fused KDA qkv trellis group,
plain bf16 KDA low-rank pair, replicated projections, fused conv1d), then checks each
rank's reads: shapes, and that rank0 + rank1 halves reconstruct the original tensor.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safetensors.numpy import save_file                                    # noqa: E402
from tensorfold.families.glm5_next.cuda.split import RankReader            # noqa: E402

L = "model.language_model.layers.0"

# name -> ("dtype", shape, filler value); shapes small but rule-divisible
TENSORS = {
    # fused KDA q|k|v trellis group: dim1 split, suh rep, svh row, codebook rep
    f"{L}.self_attn.qkv_proj.trellis": ("U32", [16, 16, 80]),
    f"{L}.self_attn.qkv_proj.suh": ("BF16", [16]),
    f"{L}.self_attn.qkv_proj.svh": ("BF16", [256]),
    f"{L}.self_attn.qkv_proj.mul1": ("U8", [16]),
    # plain KDA low-rank pair: f_a/g_a replicated, f_b/g_b/b row-split
    f"{L}.self_attn.f_a_proj.weight": ("BF16", [16, 16]),
    f"{L}.self_attn.g_a_proj.weight": ("BF16", [16, 16]),
    f"{L}.self_attn.f_b_proj.weight": ("BF16", [16, 16]),
    f"{L}.self_attn.g_b_proj.weight": ("BF16", [16, 16]),
    f"{L}.self_attn.b_proj.weight": ("BF16", [16, 16]),
    # replicated projections as EXL3 groups (q_a/kv_a in the real pack)
    f"{L}.self_attn.q_a_proj.trellis": ("U32", [16, 16, 80]),
    f"{L}.self_attn.q_a_proj.suh": ("BF16", [16]),
    f"{L}.self_attn.q_a_proj.svh": ("BF16", [16]),
    f"{L}.self_attn.q_a_proj.mul1": ("U8", [4]),
    # norms / router / fused conv1d
    f"{L}.input_layernorm.weight": ("BF16", [16]),
    f"{L}.mlp.gate.weight": ("BF16", [16, 16]),
    f"{L}.self_attn.conv1d.weight": ("BF16", [16, 16]),
    # down_proj: trellis row-split, suh row (%256), svh rep
    f"{L}.mlp.down_proj.trellis": ("U32", [16, 16, 80]),
    f"{L}.mlp.down_proj.suh": ("BF16", [256]),
    f"{L}.mlp.down_proj.svh": ("BF16", [16]),
    f"{L}.mlp.down_proj.mul1": ("U8", [4]),
    # shared tensors: lm_head trellis dropped, embed replicated
    "lm_head.trellis": ("U32", [16, 16, 80]),
    "model.embed_tokens.weight": ("BF16", [16, 16]),
}
DT_NP = {"U32": np.uint32, "BF16": np.uint16, "U8": np.uint8}

# rank0/rank1 shapes per split kind: dim1 halves dim1, row halves dim0, rep unchanged
SPLIT_AXIS = {"dim1": 1, "row": 0, "rep": None}


def build(dir_path: Path) -> dict[str, np.ndarray]:
    originals = {name: np.arange(np.prod(shape), dtype=DT_NP[dt]).reshape(shape)
                 for name, (dt, shape) in TENSORS.items()}
    by_file: dict[str, dict] = {}
    for i, (name, (dt, _)) in enumerate(TENSORS.items()):
        file = f"shard-{i % 2}.safetensors"
        by_file.setdefault(file, {})[name] = originals[name]
    for file, tensors in by_file.items():
        save_file({k: (v.astype(DT_NP[TENSORS[k][0]])) for k, v in tensors.items()}, dir_path / file)
    index = {name: file for file, tensors in by_file.items() for name in tensors}
    (dir_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {' + ",".join(f'"{k}": "{v}"' for k, v in index.items()) + "}}")
    return originals


@pytest.fixture()
def pack(tmp_path):
    return build(tmp_path)


def test_rank_slices_and_recombination(pack, tmp_path):
    originals = pack
    readers = [RankReader(tmp_path, rank) for rank in (0, 1)]
    for name, (dt, shape) in TENSORS.items():
        kind = __import__("tensorfold.families.glm5_next.cuda.split", fromlist=["rule"]).rule(name)
        if kind == "drop":                   # lm_head trellis: served from the BF16 source
            continue
        axis = SPLIT_AXIS["rep"] if kind == "rep" else SPLIT_AXIS[kind]
        if axis is None:                     # replicated: both ranks hold the whole tensor
            for reader in readers:
                t = reader.get(name)
                assert list(t.shape) == shape, name
                assert t.numpy().astype(DT_NP[dt]).tobytes() == originals[name].tobytes(), name
        else:
            half = [list(shape) for _ in range(2)]
            half[0][axis] = half[1][axis] = shape[axis] // 2
            for rank, reader in enumerate(readers):
                t = reader.get(name)
                assert list(t.shape) == half[rank], f"{name}: rank {rank} shape {list(t.shape)}"
                expect = np.split(originals[name], 2, axis=axis)[rank]
                assert t.numpy().astype(DT_NP[dt]).tobytes() == expect.tobytes(), name
    for reader in readers:
        reader.close()


def test_dropped_names_rejected(tmp_path):
    build(tmp_path)
    reader = RankReader(tmp_path, 0)
    for name in ("lm_head.trellis", "model.visual.blocks.0.attn.qkv.weight"):
        with pytest.raises(KeyError):
            reader.get(name)
    reader.close()
