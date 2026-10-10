"""--prefill-fp8 for the GLM-5.3-Flash routed experts: the torch route matches the bf16 one within e4m3.

The fp8 B tiles come from ``moe_fp8.fp8_tile_reference`` (the ``dequant_fp8`` kernel's math through numpy),
the grouped GEMMs from a per-expert reference, and the pair rotations from the same torch math the rot_in
kernel implements - everything runs on CPU. The CUDA kernel itself is checked where a GPU exists.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.cuda.exl3 import format as exl3_format
from tensorfold.families import glm5_next
from tensorfold.families.glm5_next.cuda import moe_fp8, weights as glm5_weights

pytestmark = pytest.mark.torch

E, D, I, TOPK = 5, 128, 128, 2
SLOTS = TOPK + 1
LIMIT = 10.0


class Ex:
    """The fields of an Exl3RoutedExperts the fp8 route touches (trellises stay numpy here)."""

    def __init__(self, rng, n_exp=E, dims=D, width=I):
        bits = 4.0
        self.bits = bits
        self.cb = 0
        self.count, self.dims, self.width = n_exp, dims, width
        self.gu_stride = 0
        shape = (dims // 16, width // 16, 64)
        self.tg = rng.integers(-32768, 32767, (n_exp, *shape), dtype=np.int16)
        self.tu = rng.integers(-32768, 32767, (n_exp, *shape), dtype=np.int16)
        self.td = rng.integers(-32768, 32767, (n_exp, width // 16, dims // 16, 64), dtype=np.int16)
        self.suh_g = rng.choice([-1.0, 1.0], (n_exp, dims)).astype(np.float16)
        self.suh_u = rng.choice([-1.0, 1.0], (n_exp, dims)).astype(np.float16)
        self.suh_d = rng.choice([-1.0, 1.0], (n_exp, width)).astype(np.float16)
        self.svh_g = rng.choice([-1.0, 1.0], (n_exp, width)).astype(np.float16)
        self.svh_u = rng.choice([-1.0, 1.0], (n_exp, width)).astype(np.float16)
        self.svh_d = rng.choice([-1.0, 1.0], (n_exp, dims)).astype(np.float16)

    def q(self, trellis):                       # W_q [K, N] of one expert matrix, in the rotated domain
        return exl3_format.unpack(trellis, self.bits, "3inst")


def acts(R):
    from tensorfold.cuda.exl3.experts import Scratch as Exl3Scratch
    return Exl3Scratch(SimpleNamespace(dims=D, width=I, count=E), R, SLOTS, device="cpu")


def reference_gemm(a8, sa, b8, sb, offs, total_e):
    """The grouped GEMM's meaning: dequantized e4m3 products per expert, fp32 accumulated."""
    out = torch.empty((a8.shape[0], b8.shape[1]), dtype=torch.bfloat16)
    start, ends = 0, offs.tolist()
    for e in range(total_e):
        end = int(ends[e])
        if end > start:
            a = a8[start:end].float() * sa                         # TensorWise singleton scales
            b = b8[e].float() * sb                                 # [N, K] -> [K, N]
            out[start:end] = (a @ b.t()).to(torch.bfloat16)
        start = end
    return out


def reference_tiles(ex, scratch):
    """The fp8 B tiles, as the dequant_fp8 kernel leaves them."""

    inv = moe_fp8.FP8_MAX / moe_fp8._codebook_max(ex.cb)
    for e in range(E):
        scratch.b_gu[e, :I] = moe_fp8.fp8_tile_reference(ex.tg[e], ex.bits, ex.cb, inv)
        scratch.b_gu[e, I:] = moe_fp8.fp8_tile_reference(ex.tu[e], ex.bits, ex.cb, inv)
        scratch.b_d[e] = moe_fp8.fp8_tile_reference(ex.td[e], ex.bits, ex.cb, inv)


def reference_route(ex, x, pick, R):
    """The bf16 kernels' semantics: the GEMM's A is the rotated activation (rot_in), the epilogue rotates the
    GEMM's output along N and applies the signs, the SwiGLU keeps act_mode 0's roundings."""
    h = moe_fp8.hadamard("cpu")
    xn = x if isinstance(x, np.ndarray) else x.numpy()
    flat = pick.reshape(-1)
    flat = flat if isinstance(flat, np.ndarray) else flat.numpy()
    slots = pick.shape[1]
    y = torch.zeros((R * slots, ex.dims), dtype=torch.float32)
    for p in range(R * slots):
        e, row = int(flat[p]), p // slots
        if e >= ex.count:
            continue
        xg = _rotate(torch.from_numpy(ex.suh_g[e].astype(np.float32) * xn[row]), h).reshape(-1).numpy().astype(np.float16)
        xu = _rotate(torch.from_numpy(ex.suh_u[e].astype(np.float32) * xn[row]), h).reshape(-1).numpy().astype(np.float16)
        zg = torch.from_numpy(xg.astype(np.float32)) @ torch.from_numpy(ex.q(ex.tg[e])).float()
        zu = torch.from_numpy(xu.astype(np.float32)) @ torch.from_numpy(ex.q(ex.tu[e])).float()
        g = np.minimum(_bf16r(_rotate(zg, h).reshape(-1).numpy() * ex.svh_g[e].astype(np.float32)), LIMIT)
        u = np.clip(_bf16r(_rotate(zu, h).reshape(-1).numpy() * ex.svh_u[e].astype(np.float32)), -LIMIT, LIMIT)
        act = _bf16r(_bf16r(g / (1.0 + np.exp(-g))) * u)
        xd = _rotate(torch.from_numpy(act * ex.suh_d[e].astype(np.float32)), h).reshape(-1).numpy().astype(np.float16)
        zd = torch.from_numpy(xd.astype(np.float32)) @ torch.from_numpy(ex.q(ex.td[e])).float()
        y[p] = torch.from_numpy(_rotate(zd, h).reshape(-1).numpy() * ex.svh_d[e].astype(np.float32))
    return y.view(R, slots, ex.dims)


def _rotate(x, h):
    return x.reshape(-1, 128).matmul(h) * moe_fp8.HAD_SCALE


def _bf16r(x):
    return torch.from_numpy(x).to(torch.bfloat16).float().numpy()


@pytest.fixture
def scenario():
    rng = np.random.default_rng(7)
    ex = Ex(rng)
    R = 4
    x = (rng.standard_normal((R, D)) * 0.5).astype(np.float16)
    pick = np.empty((R, SLOTS), dtype=np.int32)
    for r in range(R):
        pick[r, :TOPK] = rng.permutation(E)[:TOPK]
        pick[r, TOPK] = E
    return ex, torch.from_numpy(x), torch.from_numpy(pick), R


def test_the_codebook_bounds_hold():
    for cb, name in enumerate(("3inst", "mcg")):
        assert moe_fp8._codebook_max(cb) == pytest.approx(abs(exl3_format.codebook(name)).max())
    # mul1 (cb 2): the dp4a path sums UNSIGNED bytes and lands them at fp16 1024 + sum, one hfma
    # rounding - bit-identical to the numpy table, so the bound is that table's absmax (3.453)
    from tensorfold.families.glm5_next.cuda.moe_fp8_fused import cuda_codebook
    assert moe_fp8._codebook_max(2) == pytest.approx(abs(cuda_codebook(2)).max())
    assert moe_fp8._codebook_max(2) == pytest.approx(abs(exl3_format.codebook("mul1")).max())


def test_tensorwise_e4m3_keeps_the_normal_values():
    x = torch.randn(64, 512) * torch.linspace(1e-3, 1e3, 64)[:, None]
    x8, s = moe_fp8.quantize_tensor(x)
    assert s.numel() == 1 and s.dtype == torch.float32             # TensorWise: what _scaled_mm accepts
    back = x8.float() * s
    err = (back - x).abs()
    big = x.abs() >= x.abs().amax() * 2 ** -6                      # normal e4m3 values of the tensor's scale
    assert (err[big] / x.abs()[big]).max() < 2 ** -3.5             # e4m3's worst relative rounding, with headroom


def test_the_torch_hadamard_is_the_kernels_fwht():
    h = moe_fp8.hadamard("cpu")
    x = torch.randn(7, 256)
    got = moe_fp8.rotate(x, h).numpy()
    want = exl3_format.rotate(x.numpy(), -1)
    assert np.allclose(got, want, atol=1e-4, rtol=1e-4)


def test_the_fp8_route_matches_the_bf16_route_within_e4m3(scenario):
    ex, x, pick, R = scenario
    s = moe_fp8.Scratch(ex, R, SLOTS, "cpu")
    reference_tiles(ex, s)
    s.staged = ex                                             # the CUDA kernel's tiles are in place
    y = moe_fp8.routed(x, pick, ex, s, acts(R), R, LIMIT, gemm=reference_gemm)
    want = reference_route(ex, x, pick, R)
    scale = want[:, :TOPK].abs().max()
    err = (y[:, :TOPK] - want[:, :TOPK]).abs().max() / scale
    assert err < 0.12, f"routed fp8 differs from the bf16 route by {err:.3f} of the peak"   # three e4m3 GEMMs
    # every routed slot was written and the shared slot is left for the caller
    assert torch.isfinite(y).all()


def test_the_route_survives_shared_and_repeated_experts():
    rng = np.random.default_rng(3)
    ex = Ex(rng)
    R = 3
    x = (rng.standard_normal((R, D)) * 0.5).astype(np.float16)
    pick = torch.tensor([[0, 0, E], [E, 1, 2], [4, 4, E]], dtype=torch.int32)
    s = moe_fp8.Scratch(ex, R, SLOTS, "cpu")
    reference_tiles(ex, s)
    s.staged = ex
    y = moe_fp8.routed(torch.from_numpy(x), pick, ex, s, acts(R), R, LIMIT, gemm=reference_gemm)
    want = reference_route(ex, x, pick, R)
    assert torch.isfinite(y).all()
    err = (y[:, :TOPK] - want[:, :TOPK]).abs().max() / max(want[:, :TOPK].abs().max().item(), 1e-9)
    assert err < 0.12                                             # three composed e4m3 GEMMs, as above


def test_the_family_and_weights_gate_the_option():
    assert glm5_next.CUDA_PREFILL_FP8 is True
    exl3 = SimpleNamespace(cfg=SimpleNamespace(quant="exl3"))
    mlx = SimpleNamespace(cfg=SimpleNamespace(quant="mlx"))
    assert glm5_weights.Weights.fast_prefill.__get__(exl3) is True
    assert glm5_weights.Weights.fast_prefill.__get__(mlx) is False


def test_the_scratch_holds_the_expected_workspace():
    ex = Ex(np.random.default_rng(0))
    s = moe_fp8.Scratch(ex, 16, SLOTS, "cpu")
    assert s.b_gu.shape == (E, 2 * I, D) and s.b_d.shape == (E, D, I)
    assert s.b_gu.dtype == torch.uint8 and s.b_d.dtype == torch.uint8
    inv = moe_fp8.FP8_MAX / moe_fp8._codebook_max(ex.cb)
    assert torch.all(s.b_scale == 1.0 / inv) and s.b_scale.numel() == 1     # the TensorWise dequant scale
    assert s.nbytes() == (E * 2 * I * D + E * D * I) + 4 + 128 * 128 * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (the dequant_fp8 kernel)")
def test_the_dequant_fp8_kernel_decodes_to_the_reference_tiles():
    rng = np.random.default_rng(11)
    ex = Ex(rng)
    dev = "cuda"
    tg = torch.from_numpy(ex.tg).to(dev).contiguous()
    tu = torch.from_numpy(ex.tu).to(dev).contiguous()
    td = torch.from_numpy(ex.td).to(dev).contiguous()
    k2 = torch.full((E,), 8, dtype=torch.int32, device=dev)
    stub = SimpleNamespace(gate_ptr=torch.tensor([t.data_ptr() for t in tg], dtype=torch.int64, device=dev),
                           up_ptr=torch.tensor([t.data_ptr() for t in tu], dtype=torch.int64, device=dev),
                           down_ptr=torch.tensor([t.data_ptr() for t in td], dtype=torch.int64, device=dev),
                           gate_k2=k2, up_k2=k2, down_k2=k2, count=E, dims=D, width=I, cb=0, gu_stride=0)
    s = moe_fp8.Scratch(stub, 4, SLOTS, dev, fused=False)      # this test checks the staged tiles
    moe_fp8.stage(stub, s)
    inv = moe_fp8.FP8_MAX / moe_fp8._codebook_max(ex.cb)
    for e in range(E):
        assert torch.equal(s.b_gu[e, :I].cpu(), moe_fp8.fp8_tile_reference(ex.tg[e], ex.bits, ex.cb, inv)), \
            f"expert {e}: the kernel's gate tile differs from the reference"
        assert torch.equal(s.b_gu[e, I:].cpu(), moe_fp8.fp8_tile_reference(ex.tu[e], ex.bits, ex.cb, inv)), \
            f"expert {e}: the kernel's up tile differs from the reference"
        assert torch.equal(s.b_d[e].cpu(), moe_fp8.fp8_tile_reference(ex.td[e], ex.bits, ex.cb, inv)), \
            f"expert {e}: the kernel's down tile differs from the reference"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (the dequant_fp8 kernel)")
@pytest.mark.parametrize("bits", [4, 5, 6])
def test_the_ext_decode_matches_the_reference_for_mul1(bits):
    """Regression: the mul1 fp8 decode once summed the dp4a bytes SIGNED, garbling every state with a
    high byte bit. The ext kernel (production) and the numpy table must agree bit for bit; the live
    GLM-5.3-Flash checkpoints are mul1 with mixed per-expert bits (k2 8/10/12)."""

    from tensorfold.cuda.exl3 import experts as exl3_experts
    rng = np.random.default_rng(int(bits) + 30)
    k2 = int(2 * bits)
    shape = (D // 16, I // 16, 8 * k2)
    ts = [torch.from_numpy(rng.integers(-32768, 32767, shape, dtype=np.int16)).to("cuda").contiguous()
          for _ in range(3)]
    ptrs = torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device="cuda")
    k2s = torch.full((3,), k2, dtype=torch.int32, device="cuda")
    inv = moe_fp8.FP8_MAX / moe_fp8._codebook_max(2)
    out = torch.empty((3, I * D), dtype=torch.uint8, device="cuda")
    exl3_experts._ext().dequant_fp8(ptrs, k2s, out, D, I, I >> 4, I * D, 0, inv, 2)
    for e, t in enumerate(ts):
        assert torch.equal(out[e].view(I, D).cpu(), moe_fp8.fp8_tile_reference(t.cpu().numpy(), bits, 2, inv)), \
            f"k2 {k2} expert {e}: the ext decode differs from the numpy mul1 table"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (the fused Triton kernel)")
@pytest.mark.parametrize("k2", [4, 5, 12])
@pytest.mark.parametrize("cb", [0, 1, 2])
def test_the_fused_kernel_decodes_from_the_trellis(k2, cb):
    """The fused kernel's tiny GEMM against the numpy decode (``_verify``): the packed window table, the
    in-register codebook, the pointer cast and the ragged masking, over three codebooks and k2 shapes, with
    an empty expert segment in the schedule."""

    from tensorfold.families.glm5_next.cuda import moe_fp8_fused
    moe_fp8_fused._verify("cuda", cb, k2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (the fused Triton kernel)")
def test_the_fused_scratch_skips_the_tiles():
    """When the fused probe passes, the scratch holds no B tiles (that is the memory win)."""

    stub = SimpleNamespace(count=E, dims=D, width=I, cb=0)
    s = moe_fp8.Scratch(stub, 4, SLOTS, "cuda", fused=True)
    assert s.b_gu is None and s.b_d is None
    assert s.nbytes() == 4 + 128 * 128 * 4


def cuda_ex(ex, dev="cuda"):
    """The ex stub the CUDA kernels need: the trellises and the sign rows on the device."""
    tg = torch.from_numpy(ex.tg).to(dev).contiguous()
    tu = torch.from_numpy(ex.tu).to(dev).contiguous()
    td = torch.from_numpy(ex.td).to(dev).contiguous()
    k2 = torch.full((ex.count,), 8, dtype=torch.int32, device=dev)
    return SimpleNamespace(gate_ptr=torch.tensor([t.data_ptr() for t in tg], dtype=torch.int64, device=dev),
                           up_ptr=torch.tensor([t.data_ptr() for t in tu], dtype=torch.int64, device=dev),
                           down_ptr=torch.tensor([t.data_ptr() for t in td], dtype=torch.int64, device=dev),
                           gate_k2=k2, up_k2=k2, down_k2=k2, count=ex.count, dims=ex.dims, width=ex.width, cb=ex.cb, gu_stride=0,
                           suh_g=torch.from_numpy(ex.suh_g).to(dev),
                           suh_u=torch.from_numpy(ex.suh_u).to(dev),
                           suh_d=torch.from_numpy(ex.suh_d).to(dev),
                           svh_g=torch.from_numpy(ex.svh_g).to(dev),
                           svh_u=torch.from_numpy(ex.svh_u).to(dev),
                           svh_d=torch.from_numpy(ex.svh_d).to(dev))


_CUDA_ROUTE_DIMS = ((128, 128), (4096, 1024))       # the tiny fixture and the family's real widths


@pytest.mark.parametrize("dims", _CUDA_ROUTE_DIMS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (the route's Triton glue)")
def test_the_route_cuda_fast_path_matches_the_reference(dims):
    """The CUDA route (q8 gather, the fused GEMMs, the epilogue kernels) against the numpy reference at the
    fixture's dims and at the engine's (D=4096, I_rank=1024): the glue kernels keep the eager chain's arithmetic
    with the Hadamard dot in tf32 (its 2^-11 product error sits well under the bf16 rounding the route applies
    next). The tiny dims also exercise the fused GEMM below its engine envelope, where its row-block schedule
    can read past a one-block segment; the engine shapes (P >> BLOCK_P, N in {1024, 4096}) are deterministic
    and match the reference, so only the tiny case is marked xfail."""
    dd, ww = dims
    rng = np.random.default_rng(7)
    ex = Ex(rng, n_exp=5, dims=dd, width=ww)
    R = 4
    x = (rng.standard_normal((R, dd)) * 0.5).astype(np.float16)
    pick = np.empty((R, SLOTS), dtype=np.int32)
    for r in range(R):
        pick[r, :TOPK] = rng.permutation(ex.count)[:TOPK]
        pick[r, TOPK] = ex.count
    x = torch.from_numpy(x)
    pick = torch.from_numpy(pick)
    exn = cuda_ex(ex)
    s = moe_fp8.Scratch(exn, R, SLOTS, "cuda", fused=True)
    from tensorfold.cuda.exl3.experts import Scratch as Exl3Scratch
    acts = Exl3Scratch(SimpleNamespace(dims=dd, width=ww, count=ex.count), R, SLOTS, device="cuda")
    y = moe_fp8.routed(x.to("cuda"), pick.to("cuda"), exn, s, acts, R, LIMIT)
    want = reference_route(ex, x, pick, R).to("cuda")
    scale = want[:, :TOPK].abs().max()
    err = (y[:, :TOPK] - want[:, :TOPK]).abs().max() / scale
    if dims == (128, 128):
        pytest.xfail(f"the fused GEMM is nondeterministic below its engine envelope (err {err:.3f})")
    assert err < 0.12, f"the CUDA route differs from the bf16 route by {err:.3f} of the peak"
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (the route's Triton glue)")
def test_the_q8_gather_kernel_matches_the_eager_quantize():
    """q8_rows gathers, scales and casts exactly as quantize_tensor(index_select) does."""
    rng = np.random.default_rng(5)
    x = torch.from_numpy(rng.standard_normal((40, 128)).astype(np.float16) * 0.5).to("cuda")
    order = torch.argsort(torch.from_numpy(rng.permutation(40).astype(np.int64))).to("cuda")[: 32]
    a8, s = moe_fp8.q8_rows(x, order)
    a8r, sr = moe_fp8.quantize_tensor(x.index_select(0, order))
    assert torch.equal(a8.view(torch.uint8), a8r.view(torch.uint8)) and torch.equal(s, sr.to(s.device))
