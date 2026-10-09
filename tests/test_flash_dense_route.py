"""Dense-route format checks: bits, tile plans and layout math for RED-SNOW's projections.

Runs anywhere: the EXL3 format module is pure numpy.
"""
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensorfold.cuda.exl3 import format as fmt                          # noqa: E402
from tensorfold.cuda.exl3.linear import k2_of, plan, strips             # noqa: E402

# the pack's projection classes: (K, N) at world=1; trellis words [K/16, N/16, 16 * bits]
PROJECTIONS = [
    ("qkv_proj", 6144, 24576, 5),        # fused KDA q|k|v
    ("q_a_proj", 6144, 1536, 5),
    ("kv_a_proj_with_mqa", 6144, 512, 5),
    ("q_b_proj", 1536, 16384, 5),
    ("o_proj", 16384, 6144, 5),
    ("gate/up (dense)", 6144, 1536, 5),
    ("gate/up (routed)", 6144, 2048, 5),
    ("down (routed)", 2048, 6144, 5),
    ("eh_proj (MTP)", 6144, 12288, 5),
]


@pytest.mark.parametrize("name,k,n,bits", PROJECTIONS)
def test_tiles_divide(name, k, n, bits):
    kt, nt = k // 16, n // 16
    sk, wk = plan(k, n)
    assert (k // 16) % (sk * wk) == 0, f"{name}: trellis tile rows {kt} vs plan ({sk}, {wk})"
    assert k % 128 == 0 and n % 128 == 0, name
    # the TP2 rank slice: out-linears halve the tile columns, in-linears the rows; both stay whole tiles
    assert n % 32 == 0 and k % 32 == 0, name


def test_bits_of_widths():
    assert fmt.bits_of((256, 1536, 80)) == 5          # mul1 5-bit: RED-SNOW's mean
    assert fmt.bits_of((256, 1536, 64)) == 4
    assert fmt.bits_of((256, 1536, 96)) == 6
    with pytest.raises(ValueError):
        fmt.bits_of((256, 1536, 72))                  # 4.5: not in BITS yet (phase 5 open question)


def test_strips_layout_roundtrip():
    kt, nt, w = 8, 16, 80
    words = torch.arange(kt * nt * w, dtype=torch.int32).reshape(kt, nt, w)
    s = strips(words)                                  # [N/128, K/16, 8, W]
    assert tuple(s.shape) == (2, 8, 8, 80)
    for b in range(2):                                 # block b holds tiles [8b:8b+8] of the n axis, k order kept
        assert torch.equal(s[b], words[:, b * 8:(b + 1) * 8, :])


def test_k2_of_bits():
    assert k2_of(5) == 10
    assert k2_of(4) == 8
