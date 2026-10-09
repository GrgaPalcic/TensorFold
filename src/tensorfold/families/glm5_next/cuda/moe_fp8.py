"""--prefill-fp8 for GLM-5.3-Flash's routed EXL3 experts: the GEMMs run in e4m3 with TensorWise scales.

The EXL3 GEMV is bandwidth-bound on the trellises (about 5.5 GiB a layer across TP2), so prefill keeps the
bf16 GEMV while the GEMMs move to fp8: each layer's three expert matrices are decoded once into e4m3 B tiles
(``dequant_fp8``, one scalar scale per matrix - e4m3's relative rounding is scale invariant), the grouped
GEMMs run through ``torch._scaled_grouped_mm`` (per-expert ``_scaled_mm`` as the fallback), and the shared
expert stays a BF16 MLP. The routed path loses at most e4m3's ~2^-4 relative precision per matrix product;
decode keeps the exact bf16 trellis GEMV.

Per expert matrix (suh is Hadamard-folded into the pairs' activations by ``rot_in``, exactly as the bf16
route does; svh folds into the epilogue):

    B8 = e4m3(W_q / s)          s = codebook_max / 448      (the kernel's decode, K-major output)
    Z = gemm(e4m3(H x suh), B8)                              (grouped, one tensor scale a matrix)
    Y = rotate(Z) * HAD_SCALE * svh                          (the epilogue's Hadamard and signs)

The approximate workspace is the three e4m3 tile sets (2.4 + 1.2 GiB for 5.3-Flash on one rank) plus the
pairs' fp8 activations, on top of the bf16 route's scratch.
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache
import torch

from tensorfold.cuda import prompt_precision
from tensorfold.cuda.exl3 import experts as generic
from tensorfold.cuda.exl3 import format as exl3_format

HAD = 128
HAD_SCALE = 0.08838834764831845            # 1 / sqrt(128), the kernels' constant
FP8_MAX = 448.0                            # e4m3fn's largest finite value


def enabled() -> bool:
    """Whether --prefill-fp8 asked for the fp8 expert route (checked per call; tests flip the option)."""

    return prompt_precision.fp8()


@lru_cache(maxsize=3)
def _codebook_max(cb: int) -> float:
    """The largest absolute value a codebook state decodes to: the scale's bound (conservative is fine -
    e4m3 rounding is relative, so any scale that cannot overflow gives the same ~2^-4 relative error)."""

    return float(abs(exl3_format.codebook(("3inst", "mcg", "mul1")[cb])).max())


class Scratch:
    """The fp8 expert tiles of one layer at a time plus the GEMMs' tensor scale and Hadamard matrix."""

    def __init__(self, ex: generic.Exl3RoutedExperts, rows: int, slots: int, device) -> None:
        E, D, I = ex.count, ex.dims, ex.width
        self.rows, self.slots = rows, slots
        self.b_gu = torch.empty((E, 2 * I, D), dtype=torch.uint8, device=device)   # e4m3, gate | up stacked
        self.b_d = torch.empty((E, D, I), dtype=torch.uint8, device=device)
        # tiles hold e4m3(W * inv), so dequant needs 1 / inv; one scalar covers every expert because the
        # checkpoint's experts share one codebook, whose bound is a valid tensor scale (it never overflows)
        self.b_scale = torch.full((1,), 1.0 / (FP8_MAX / _codebook_max(ex.cb)), dtype=torch.float32, device=device)
        self.h = hadamard(device)
        self.staged = None               # the layer whose tiles are in place (identity check)

    def nbytes(self) -> int:
        """The fp8 workspace's own bytes (the bf16 route's scratch is separate)."""

        return self.b_gu.numel() + self.b_d.numel() + self.b_scale.numel() * 4 + self.h.numel() * 4


@lru_cache(maxsize=1)
def _h128_cpu() -> torch.Tensor:
    h = np.ones((1, 1))
    while h.shape[0] < HAD:
        h = np.block([[h, h], [h, -h]])            # H[i, j] = (-1)^popcount(i & j)
    return torch.from_numpy(h).float()


def hadamard(device) -> torch.Tensor:
    """The 128-point Hadamard matrix as fp32 (symmetric, so ``x @ h`` is the kernels' fwht128)."""

    return _h128_cpu().to(device)


def rotate(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """The epilogue's Hadamard along the last dim in fp32, with HAD_SCALE folded in: fwht128 then * HAD_SCALE."""

    shape = x.shape
    y = x.reshape(-1, HAD).float().matmul(h) * HAD_SCALE
    return y.reshape(shape)


def quantize_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 with one scale for the whole matrix (SM121's _scaled_mm only takes TensorWise scales): s = amax / 448
    over every element (0-matrices at 1). e4m3's wide dynamic range keeps outliers from drowning small values;
    the epilogues stay in bf16/fp32 either way."""

    amax = x.float().abs().amax()
    s = torch.where(amax > 0, amax * (1.0 / FP8_MAX), torch.ones_like(amax)).reshape(1)
    return (x.float() / s).to(torch.float8_e4m3fn).contiguous(), s


def fp8_tile_reference(trellis, bits: float, cb: int, inv: float) -> torch.Tensor:
    """The dequant_fp8 kernel's output as uint8 bytes [N, K] (e4m3 bits, column-major per expert; tests)."""

    q = torch.from_numpy(exl3_format.unpack(trellis, bits, ("3inst", "mcg", "mul1")[cb]))
    return (q.float() * inv).to(torch.float8_e4m3fn).t().contiguous().view(torch.uint8)


def stage(ex: generic.Exl3RoutedExperts, s: Scratch) -> None:
    """This layer's three expert matrices as e4m3 B tiles (column-major [N, K] per expert; gate and up share
    one [E, 2I, D] buffer: gate at proj_off 0, up at proj_off I)."""

    if s.staged is ex:
        return
    ext = generic._ext()
    D, I, inv = ex.dims, ex.width, FP8_MAX / _codebook_max(ex.cb)
    rows = ex.gu_stride if ex.gu_stride else I >> 4
    ext.dequant_fp8(ex.gate_ptr, ex.gate_k2, s.b_gu, D, I, rows, 2 * I * D, 0, inv, ex.cb)
    ext.dequant_fp8(ex.up_ptr, ex.up_k2, s.b_gu, D, I, rows, 2 * I * D, I, inv, ex.cb)
    ext.dequant_fp8(ex.down_ptr, ex.down_k2, s.b_d, I, D, D >> 4, D * I, 0, inv, ex.cb)
    s.staged = ex


def grouped_mm(a8: torch.Tensor, sa: torch.Tensor, b8: torch.Tensor, sb: torch.Tensor, offs: torch.Tensor,
               E: int) -> torch.Tensor:
    """The grouped fp8 GEMM: a8 [P, K] e4m3, b8 [E, N, K] e4m3 (column-major per expert), offs [E] int32
    cumulative row ends -> [P, N] bf16. Both scales are TensorWise singletons: what SM121's _scaled_mm
    accepts (its row-wise scaling is rejected, and _scaled_grouped_mm is unsupported below SM100)."""

    ok = _grouped_ok()
    if ok:
        try:
            return torch._scaled_grouped_mm(a8, b8.transpose(-2, -1), sa, sb, offs=offs, out_dtype=torch.bfloat16)
        except (RuntimeError, ValueError):
            globals()["_GROUPED"] = False
    # per-expert fallback (one host sync to read the group sizes)
    out = torch.empty((a8.shape[0], b8.shape[1]), dtype=torch.bfloat16, device=a8.device)
    ends = offs.tolist()
    start = 0
    for e in range(E):
        end = int(ends[e])
        if end > start:
            out[start:end] = torch._scaled_mm(a8[start:end], b8[e].t(), sa, sb, out_dtype=torch.bfloat16)
        start = end
    return out


_GROUPED: bool | None = None


def _grouped_ok() -> bool:
    """Whether ``torch._scaled_grouped_mm`` runs on this build (SM90+ kernels; probed on first use)."""

    global _GROUPED
    if _GROUPED is None:
        _GROUPED = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
    return _GROUPED


def routed(x: torch.Tensor, pick: torch.Tensor, ex: generic.Exl3RoutedExperts, s: Scratch, acts, R: int,
           limit: float, gemm=grouped_mm) -> torch.Tensor:
    """The routed slots' fp8 prefill: acts.y[:R * slots] as fp32 (the caller supplies the shared slot) -> [R, slots, D].

    Same arithmetic as the bf16 route's rot_in / grouped GEMV / epilogue, with the three GEMMs in e4m3: gate
    and up read the pairs' rotated activations (suh folded, one GEMM each), the SwiGLU keeps the bf16 route's
    roundings (act_mode 0), and down folds suh_d into its pairs' activations. TensorWise scales throughout:
    one scale a matrix (the GEMM's A scale comes from the routed pairs' amax, B's from the codebook bound).
    """

    D, I, E = ex.dims, ex.width, ex.count
    slots, P = s.slots, R * s.slots
    if R > s.rows:
        raise ValueError(f"{R} rows but the scratch holds {s.rows}")
    rot_in(x, pick, ex, acts, R, D, slots, E)
    # group the routed pairs by expert with no host sync: the shared expert's id E sorts last, and each row
    # contributes exactly one shared slot, so the routed count is R * top_k
    flat = pick.reshape(-1)
    order = torch.argsort(flat)[: R * (slots - 1)]          # pair indices, sorted by expert
    counts = torch.bincount(flat.long(), minlength=E + 1)[:E]
    offs = counts.cumsum(0).to(torch.int32)
    a8_g, s_g = quantize_tensor(acts.xg[:P].index_select(0, order))   # the pairs' fp8 A tiles, expert order
    a8_u, s_u = quantize_tensor(acts.xu[:P].index_select(0, order))
    b_gu = s.b_gu.view(torch.float8_e4m3fn).view(E, 2 * I, D)      # column-major [N, K] per expert
    z_g = gemm(a8_g, s_g, b_gu[:, :I], s.b_scale, offs, E)
    z_u = gemm(a8_u, s_u, b_gu[:, I:], s.b_scale, offs, E)
    # the epilogue: Hadamard, signs, the SwiGLU with the bf16 route's roundings (act_mode 0)
    e_of = flat[order]
    gg = bf16r(rotate(z_g, s.h) * signs(ex.svh_g, e_of)).clamp_max(limit)
    uu = bf16r(rotate(z_u, s.h) * signs(ex.svh_u, e_of)).clamp(-limit, limit)
    act = bf16r(bf16r(gg * torch.sigmoid(gg)) * uu)
    xd = rotate(act * signs(ex.suh_d, e_of), s.h)
    a8_d, s_d = quantize_tensor(xd)                  # already in expert-sorted pair order
    z = gemm(a8_d, s_d, s.b_d.view(torch.float8_e4m3fn).view(E, D, I), s.b_scale, offs, E)
    # the down epilogue and the slot scatter (one fp32 write per routed pair)
    y = rotate(z, s.h) * signs(ex.svh_d, e_of)
    acts.y[:P].index_copy_(0, order, y)
    return acts.y[:P].view(R, slots, D)


def signs(packed: torch.Tensor, e_of: torch.Tensor) -> torch.Tensor:
    """The pairs' per-expert sign rows as fp32 [P, N] (suh_g / svh_g / suh_u / svh_u / suh_d / svh_d)."""

    return torch.as_tensor(packed).index_select(0, e_of.long()).float()



def rot_in(x: torch.Tensor, pick: torch.Tensor, ex, acts, R: int, D: int, slots: int, E: int) -> None:
    """The pairs' rotated activations into acts.xg / acts.xu: the rot_in kernel on CUDA, the same math in
    torch otherwise (tests): fp16((H (x * suh)) * HAD_SCALE) per pair; shared slots are left out."""

    if x.is_cuda:
        generic._ext().rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, acts.xg, acts.xu, R, D, slots, E)
        return
    pairs = pick.reshape(-1).long().clamp_max(E - 1)
    for suh, out in ((ex.suh_g, acts.xg), (ex.suh_u, acts.xu)):
        su = torch.as_tensor(suh)
        vals = x.repeat_interleave(slots, 0).float() * su[pairs].float()
        out[:R * slots] = rotate(vals, _h128_cpu()).half()


def bf16r(x: torch.Tensor) -> torch.Tensor:
    """The bf16 route's round-to-nearest (the kernel's bf16r)."""

    return x.to(torch.bfloat16).float()
