"""The GLM-5.3-Flash latent cache's q4 option (alongside bf16 and fp8): packed 4-bit codes, two a byte, with a
per-token power-of-two scale — the roundtrip's error bound against the bf16 reference, the geometry accounting
and the dtype knob's plumbing (families/glm5_next/cuda/latent.py, forward.State, engine)."""

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.geometry import mla_geometry  # noqa: E402


def _quantize(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's scheme: s = 2^ceil(log2(amax / 7)) (1.0 for an all-zero row), codes floor(x / s + 0.5)
    clamped to [-8, 7], two codes a byte (element 2k low, 2k + 1 high, each nibble the code's two's complement)."""
    amax = rows.float().abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, torch.exp2(torch.ceil(torch.log2(amax / 7.0))), torch.ones_like(amax))
    codes = torch.clamp(torch.floor(rows.float() / s + 0.5), -8, 7).to(torch.int32)
    pairs = codes.view(-1, rows.shape[1] // 2, 2)
    packed = ((pairs[:, :, 0] & 0xF) | (pairs[:, :, 1] << 4)).to(torch.uint8)
    return packed, s.squeeze(-1)


def _dequant(packed: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    b = packed.view(packed.shape[0], -1).int()
    codes = torch.stack([((b & 0xF) ^ 8) - 8, (((b >> 4) & 0xF) ^ 8) - 8], dim=-1).view(packed.shape[0], -1)
    return codes.float() * s.float()[:, None]


def test_q4_latent_roundtrip_error_bound():
    gen = torch.Generator().manual_seed(7)
    rows = torch.randn((4096, 512), generator=gen)          # unit-normal rows, the latents' scale
    got = _dequant(*_quantize(rows))
    rel = (got - rows).norm() / rows.norm()
    assert rel < 0.25                                       # ~0.19: 16 codes over the row's range
    # the codes never clip the row's largest value: amax <= 7 s holds by the scale's construction
    amax = rows.float().abs().amax(1)
    s = torch.exp2(torch.ceil(torch.log2(amax / 7.0)))
    assert (amax <= 7.0 * s + 1e-5).all()
    # an all-zero row keeps zeros (scale 1, no 0 / 0)
    zero = _dequant(*_quantize(torch.zeros((1, 512))))
    assert zero.eq(0).all()


def test_mla_geometry_counts_q4_latent_rows():
    t = {"num_hidden_layers": 4, "full_attention_interval": 1, "hidden_size": 512, "kv_lora_rank": 512,
         "num_attention_heads": 16, "qk_nope_head_dim": 128, "v_head_dim": 128,
         "moe_intermediate_size": 512, "vocab_size": 1000, "num_experts_per_tok": 8,
         "index_n_heads": 8, "index_head_dim": 128}
    layers = 4
    bf16 = mla_geometry(t, 1, 8, latent=True).bytes_at(1024)
    q4 = mla_geometry(t, 1, 8, latent=True, kv_dtype="q4").bytes_at(1024)
    assert bf16 - q4 == layers * 1024 * (512 * 2 - (256 + 4))    # 2 B a latent element -> 1/2 B + 4 B a token
    fp8 = mla_geometry(t, 1, 8, latent=True, kv_dtype="fp8").bytes_at(1024)
    assert q4 < fp8 < bf16
    assert mla_geometry(t, 1, 8, latent=False, kv_dtype="q4").bytes_at(1024) == \
        mla_geometry(t, 1, 8, latent=False).bytes_at(1024)       # the non-latent path ignores q4


def test_the_dtype_knob_reaches_the_family_and_the_cli():
    from tensorfold import cli
    from tensorfold.families import glm5_next

    assert glm5_next.CUDA_KV_DTYPES == ("bf16", "fp8", "q4")
    assert cli.build_parser().parse_args(["serve", "owner/model", "--kv-dtype", "q4"]).kv_dtype == "q4"
    # an unknown cache is refused before anything loads
    with pytest.raises(ValueError, match="kv-dtype"):
        glm5_next.cuda_engine("/nonexistent", tp=2, master="h", kv_dtype="int8")


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


@cuda
def test_q4_latent_kernels_track_the_bf16_cache():
    """The q4 cache written and read by the Triton kernels tracks the bf16 cache within the roundtrip error."""
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator("cuda").manual_seed(3)
    H, R, P = 16, 4, 512
    cache = torch.zeros((1024, 512), dtype=torch.bfloat16, device="cuda")
    q4_cache = torch.zeros((1024, 256), dtype=torch.uint8, device="cuda")
    q4_scales = torch.zeros((1024,), dtype=torch.float32, device="cuda")
    lat = torch.randn((R, 512), generator=gen, device="cuda", dtype=torch.bfloat16)
    pos = torch.tensor([P], dtype=torch.int32, device="cuda")
    latent.latent_write(lat, cache, pos)
    latent.latent_write(lat, q4_cache, pos, q4_scales)
    qa = torch.randn((R, H, 512), generator=gen, device="cuda", dtype=torch.bfloat16)
    scratch = latent.LatentScratch(R, H, latent.chunks_for(P + R), "cuda")
    out = torch.empty((R, H, 512), dtype=torch.bfloat16, device="cuda")
    q4_out = torch.empty_like(out)
    latent.attention(qa, cache, pos, scratch, scale=0.07, nch=latent.chunks_for(P + R), out=out)
    latent.attention(qa, q4_cache, pos, scratch, scale=0.07, nch=latent.chunks_for(P + R), out=q4_out,
                     scales=q4_scales)
    rel = (q4_out.float() - out.float()).norm() / out.float().norm()
    assert rel < 0.25
    # the sparse read dequantizes too
    tokens = torch.arange(P, P + R, device="cuda", dtype=torch.int32).unsqueeze(0).expand(R, R).contiguous()
    counts = torch.full((R,), R, dtype=torch.int32, device="cuda")
    sp = torch.empty_like(out)
    sq4 = torch.empty_like(out)
    latent.sparse_attention(qa, cache, tokens, counts, sp, 0.07)
    latent.sparse_attention(qa, q4_cache, tokens, counts, sq4, 0.07, scales=q4_scales)
    rel = (sq4.float() - sp.float()).norm() / sp.float().norm()
    assert rel < 0.25
    # the written rows dequantize back onto the bf16 rows within the bound
    b = q4_cache[P:P + R].int()
    nibbles = torch.stack([(b & 0xF), (b >> 4)], dim=-1).view(R, 512)
    roundtrip = (((nibbles ^ 8) - 8).float() * q4_scales[P:P + R, None])
    rel = (roundtrip - lat.float()).norm() / lat.float().norm()
    assert rel < 0.25
    # the kernel's packing and the test's packing agree bit for bit
    packed, s = _quantize(lat)
    assert packed.eq(q4_cache[P:P + R]).all()
    assert torch.allclose(s, q4_scales[P:P + R])
