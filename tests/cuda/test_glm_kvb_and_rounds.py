"""TF_GLM_KVB formats (fp8/q4 kv_b), the wider decode windows' knobs, and late token delivery."""

import pytest
import torch

from tests.test_cuda_geometry import allocations  # noqa: F401  (a fake triton, no GPU needed)


def test_the_kv_b_formats_and_their_dispatch(allocations):
    from tensorfold.families.glm5_next.cuda import latent, qmm

    heads, qk_dim, v_dim, lw = 2, 16, 32, 256
    k_rows = torch.randn(heads * qk_dim, lw, dtype=torch.float32)
    v_rows = torch.randn(heads * v_dim, lw, dtype=torch.float32)

    assert latent.from_bf16(k_rows, v_rows, heads, "bf16").__class__ is latent.AbsorbW
    f8 = latent.from_bf16(k_rows, v_rows, heads, "fp8")
    assert f8.__class__ is latent.AbsorbF8 and f8.heads == heads and f8.lw == lw
    q4 = latent.from_bf16(k_rows, v_rows, heads, "q4")
    assert q4.__class__ is latent.AbsorbQ4
    assert latent.from_bf16(k_rows, v_rows, heads).lw == lw      # default: bf16
    odd = latent.from_bf16(k_rows[:, :100], v_rows[:, :100], heads, "fp8")
    assert odd.__class__ is latent.AbsorbW                       # an untiled latent stays bf16
    with pytest.raises(ValueError):
        latent.from_bf16(k_rows, v_rows, heads, "int8")

    bf16 = latent.AbsorbW.from_rows(k_rows, v_rows, heads)
    assert f8.nbytes() < bf16.nbytes() < 1.1 * f8.nbytes() + 8 * bf16.nbytes() // 8
    assert f8.sk.shape == (heads, qk_dim, lw // latent.F8_BLOCK)


def test_make_f8_quantization_is_blockwise_and_close(allocations):
    from tensorfold.families.glm5_next.cuda.qmm import F8_BLOCK, make_f8

    w = 0.01 * torch.randn(6, F8_BLOCK * 3, dtype=torch.float32)
    f8 = make_f8(w.to(torch.bfloat16))
    assert f8.weight.dtype == torch.float8_e4m3fn and f8.scale.shape == (6, 3)
    back = f8.weight.float().view(6, 3, F8_BLOCK) * f8.scale.view(6, 3, 1)
    err = (back.reshape(6, F8_BLOCK * 3) - w).abs().max()
    assert err < 0.05 * w.abs().max()


def test_kvb_kind_and_the_weights_estimate_transform(allocations):
    from tensorfold.families.glm5_next.cuda.qmm import KVB_KINDS, kvb_kind, kvb_weights

    assert kvb_kind({}) == "bf16" and "fp8" in KVB_KINDS and "q4" in KVB_KINDS
    assert kvb_kind({"TF_GLM_KVB": "fp8"}) == "fp8"
    with pytest.raises(ValueError):
        kvb_kind({"TF_GLM_KVB": "int4"})

    def base(name, info):
        shape = info["shape"]
        return (shape[0] * shape[1] * 2, 0)      # bf16 bytes

    est = kvb_weights(base, "fp8")
    latent_width = 4096                          # 128 columns a scale: bytes halve plus fp32 scales
    rows = 512
    assert est("layers.3.self_attn.kv_b_proj.weight", {"shape": [rows, latent_width], "dtype": "BF16"}) == \
        (rows * latent_width + rows * (latent_width // 128) * 4, 0)
    assert est("layers.3.self_attn.kv_b_proj.weight", {"shape": [rows, 100], "dtype": "BF16"}) == (rows * 100 * 2, 0)
    assert est("layers.3.mlp.down_proj.weight", {"shape": [rows, latent_width], "dtype": "BF16"}) == (rows * latent_width * 2, 0)
    q4est = kvb_weights(base, "q4")
    assert q4est("layers.3.self_attn.kv_b_proj.weight", {"shape": [rows, latent_width], "dtype": "BF16"}) == \
        (rows * latent_width // 2 + rows * (latent_width // 64) * 4, 0)
    assert kvb_weights(base, "bf16")("layers.3.self_attn.kv_b_proj.weight",
                                     {"shape": [rows, latent_width], "dtype": "BF16"}) == (rows * latent_width * 2, 0)


def test_the_wider_decode_window_knobs():
    from tensorfold.families.glm5_next.cuda import engine

    assert engine.MAX_ROWS == 16 and engine.MTP_ROWS == 8
    assert engine.GRAPH_ROWS == (1, 2, 3, 4, 5, 6, 7, 8)


def test_the_kept_prompt_point_sits_on_a_kda_sub_chunk_boundary():
    from tensorfold.families.glm5_next.cuda.engine import kept_point

    # the last multiple of 32 before the prompt's final token; none for prompts of 32 tokens or fewer
    assert [kept_point(n) for n in (1, 32, 33, 64, 65, 221, 3011)] == [None, None, 32, 32, 64, 192, 3008]


def test_the_decode_clock(allocations):
    from tensorfold.families.glm5_next.cuda.decode import _Clock

    clock = _Clock()
    assert clock() >= 0.0
