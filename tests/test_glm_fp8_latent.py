"""The GLM-5.3-Flash latent cache's fp8 (e4m3fn, per-token scale) option: the quantize -> dequant roundtrip's
error bound on unit-normal rows, and the geometry accounting (families/glm5_next/cuda/latent.py)."""

import math

import pytest
import torch

from tensorfold.cuda.geometry import mla_geometry


def _quantize(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's scheme: s = amax(row) / 448 (1.0 for an all-zero row), store (row / s) as e4m3fn."""
    amax = rows.abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    return (rows / s).to(torch.float8_e4m3fn), s.squeeze(-1)


def _dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * s[:, None]


def test_fp8_latent_roundtrip_error_bound():
    gen = torch.Generator().manual_seed(7)
    rows = torch.randn((4096, 512), generator=gen)          # unit-normal rows, the latents' scale
    got = _dequant(*_quantize(rows))
    rel = (got - rows).norm() / rows.norm()
    assert rel < 0.1
    # the largest value of each row survives exactly (amax / 448 fills e4m3fn's range)
    amax = rows.abs().amax(1)
    assert torch.allclose(got.abs().amax(1), amax, rtol=0.02)
    # an all-zero row keeps zeros (scale 1, no 0 / 0)
    zero = _dequant(*_quantize(torch.zeros((1, 512))))
    assert zero.eq(0).all()


def test_mla_geometry_counts_fp8_latent_rows():
    t = {"num_hidden_layers": 4, "full_attention_interval": 1, "hidden_size": 512, "kv_lora_rank": 512,
         "num_attention_heads": 16, "qk_nope_head_dim": 128, "v_head_dim": 128,
         "moe_intermediate_size": 512, "vocab_size": 1000, "num_experts_per_tok": 8,
         "index_n_heads": 8, "index_head_dim": 128}
    layers = 4
    bf16 = mla_geometry(t, 1, 8, latent=True).bytes_at(1024)
    fp8 = mla_geometry(t, 1, 8, latent=True, fp8=True).bytes_at(1024)
    layers = 4
    assert bf16 - fp8 == layers * 1024 * (512 * 2 - (512 + 4))    # 2 B a latent element -> 1 B + 4 B a token
    assert mla_geometry(t, 1, 8, latent=False, fp8=True).bytes_at(1024) == \
        mla_geometry(t, 1, 8, latent=False).bytes_at(1024)        # the non-latent path ignores fp8


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


@cuda
def test_fp8_latent_kernels_match_the_bf16_cache():
    """The fp8 cache written and read by the Triton kernels tracks the bf16 cache within the roundtrip error."""
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator("cuda").manual_seed(3)
    H, R, P = 16, 4, 512
    cache = torch.zeros((1024, 512), dtype=torch.bfloat16, device="cuda")
    scales = torch.zeros((1024,), dtype=torch.float32, device="cuda")
    fp8_cache = torch.zeros((1024, 512), dtype=torch.float8_e4m3fn, device="cuda")
    fp8_scales = torch.zeros((1024,), dtype=torch.float32, device="cuda")
    lat = torch.randn((R, 512), generator=gen, device="cuda", dtype=torch.bfloat16)
    pos = torch.tensor([P], dtype=torch.int32, device="cuda")
    latent.latent_write(lat, cache, pos)
    latent.latent_write(lat, fp8_cache, pos, fp8_scales)
    qa = torch.randn((R, H, 512), generator=gen, device="cuda", dtype=torch.bfloat16)
    scratch = latent.LatentScratch(R, H, latent.chunks_for(P + R), "cuda")
    out = torch.empty((R, H, 512), dtype=torch.bfloat16, device="cuda")
    fp8_out = torch.empty_like(out)
    latent.attention(qa, cache, pos, scratch, scale=0.07, nch=latent.chunks_for(P + R), out=out)
    latent.attention(qa, fp8_cache, pos, scratch, scale=0.07, nch=latent.chunks_for(P + R), out=fp8_out,
                     scales=fp8_scales)
    rel = (fp8_out.float() - out.float()).norm() / out.float().norm()
    assert rel < 0.1
    # the sparse read dequantizes too
    tokens = torch.arange(P, P + R, device="cuda", dtype=torch.int32).unsqueeze(0).expand(R, R).contiguous()
    counts = torch.full((R,), R, dtype=torch.int32, device="cuda")
    sp = torch.empty_like(out)
    sp8 = torch.empty_like(out)
    latent.sparse_attention(qa, cache, tokens, counts, sp, 0.07)
    latent.sparse_attention(qa, fp8_cache, tokens, counts, sp8, 0.07, scales=fp8_scales)
    rel = (sp8.float() - sp.float()).norm() / sp.float().norm()
    assert rel < 0.1
    # the written rows dequantize back onto the bf16 rows within the bound
    roundtrip = fp8_cache[P:P + R].to(torch.float32) * fp8_scales[P:P + R, None]
    rel = (roundtrip - lat.float()).norm() / lat.float().norm()
    assert rel < 0.1
