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

    def __init__(self, rng):
        bits = 4.0
        self.bits = bits
        self.cb = 0
        self.count, self.dims, self.width = E, D, I
        self.gu_stride = 0
        shape = (D // 16, I // 16, 64)
        self.tg = rng.integers(-32768, 32767, (E, *shape), dtype=np.int16)
        self.tu = rng.integers(-32768, 32767, (E, *shape), dtype=np.int16)
        self.td = rng.integers(-32768, 32767, (E, I // 16, D // 16, 64), dtype=np.int16)
        self.suh_g = rng.choice([-1.0, 1.0], (E, D)).astype(np.float16)
        self.suh_u = rng.choice([-1.0, 1.0], (E, D)).astype(np.float16)
        self.suh_d = rng.choice([-1.0, 1.0], (E, I)).astype(np.float16)
        self.svh_g = rng.choice([-1.0, 1.0], (E, I)).astype(np.float16)
        self.svh_u = rng.choice([-1.0, 1.0], (E, I)).astype(np.float16)
        self.svh_d = rng.choice([-1.0, 1.0], (E, D)).astype(np.float16)

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
            a = a8[start:end].float() * sa[start:end, None]
            b = b8[e].float() * sb[e][None, :]                     # [N, K] -> [K, N]
            out[start:end] = (a @ b.t()).to(torch.bfloat16)
        start = end
    return out


def reference_tiles(ex, scratch):
    """The fp8 B tiles and the per-column scales, as the dequant_fp8 kernel leaves them."""

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
    y = torch.zeros((R * SLOTS, D), dtype=torch.float32)
    for p in range(R * SLOTS):
        e, row = int(flat[p]), p // SLOTS
        if e >= E:
            continue
        xg = _rotate(torch.from_numpy(ex.suh_g[e].astype(np.float32) * xn[row]), h).numpy().astype(np.float16)
        xu = _rotate(torch.from_numpy(ex.suh_u[e].astype(np.float32) * xn[row]), h).numpy().astype(np.float16)
        zg = torch.from_numpy(xg.astype(np.float32)) @ torch.from_numpy(ex.q(ex.tg[e])).float()
        zu = torch.from_numpy(xu.astype(np.float32)) @ torch.from_numpy(ex.q(ex.tu[e])).float()
        g = np.minimum(_bf16r(_rotate(zg, h).numpy() * ex.svh_g[e].astype(np.float32)), LIMIT)
        u = np.clip(_bf16r(_rotate(zu, h).numpy() * ex.svh_u[e].astype(np.float32)), -LIMIT, LIMIT)
        act = _bf16r(_bf16r(g / (1.0 + np.exp(-g))) * u)
        xd = _rotate(torch.from_numpy(act * ex.suh_d[e].astype(np.float32)), h).numpy().astype(np.float16)
        zd = torch.from_numpy(xd.astype(np.float32)) @ torch.from_numpy(ex.q(ex.td[e])).float()
        y[p] = torch.from_numpy(_rotate(zd, h).numpy() * ex.svh_d[e].astype(np.float32))
    return y.view(R, SLOTS, D)


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
    for cb, name in enumerate(("3inst", "mcg", "mul1")):
        assert moe_fp8._codebook_max(cb) == pytest.approx(abs(exl3_format.codebook(name)).max())


def test_per_row_e4m3_stays_within_the_relative_bound():
    x = torch.randn(64, 512) * torch.linspace(1e-3, 1e3, 64)[:, None]
    x8, s = moe_fp8.quantize_rows(x)
    back = x8.float() * s[:, None]
    err = (back - x).abs()
    big = x.abs() >= x.abs().amax(-1, keepdim=True) * 2 ** -6      # normal e4m3 values of the row's scale
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
    assert torch.all(s.s_g == 1.0 / inv) and torch.all(s.s_d == 1.0 / inv)
    assert s.nbytes() == (E * 2 * I * D + E * D * I + 2 * E * I + E * D) + 128 * 128 * 4


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
    s = moe_fp8.Scratch(stub, 4, SLOTS, dev)
    moe_fp8.stage(stub, s)
    inv = moe_fp8.FP8_MAX / moe_fp8._codebook_max(ex.cb)
    for e in range(E):
        assert torch.equal(s.b_gu[e, :I].cpu(), moe_fp8.fp8_tile_reference(ex.tg[e], ex.bits, ex.cb, inv)), \
            f"expert {e}: the kernel's gate tile differs from the reference"
        assert torch.equal(s.b_gu[e, I:].cpu(), moe_fp8.fp8_tile_reference(ex.tu[e], ex.bits, ex.cb, inv)), \
            f"expert {e}: the kernel's up tile differs from the reference"
        assert torch.equal(s.b_d[e].cpu(), moe_fp8.fp8_tile_reference(ex.td[e], ex.bits, ex.cb, inv)), \
            f"expert {e}: the kernel's down tile differs from the reference"
