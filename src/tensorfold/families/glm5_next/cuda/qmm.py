"""GLM-5.3-Flash's 4-bit and BF16 dense matmuls; the same groups and K slices at any row count keep each row's bits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels import qmm as shared

BN = 64                   # columns per stored tile
GS = 64                   # inputs per quantization group


@dataclass
class Q4:
    """A 4-bit group-64 matrix [n, k] packed for the shared lane matmul (``shared.pack``)."""

    weight: torch.Tensor
    scales: torch.Tensor
    biases: torch.Tensor
    n: int
    k: int
    gs: int = GS

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scales, self.biases))


def as_i32(w: torch.Tensor) -> torch.Tensor:
    return w.view(torch.int32) if w.dtype != torch.int32 else w


def make_q4(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> Q4:
    """From MLX arrays: weight (N, K/8) uint32/int32, scales/biases (N, K/64) bf16."""

    w = as_i32(weight)
    n, k8 = w.shape
    if scales.shape != (n, k8 // 8):
        raise ValueError(f"group-64 scales expected ({n}, {k8 // 8}), got {tuple(scales.shape)}")
    p = shared.pack(w, scales, biases, GS)
    return Q4(p.weight, p.scales, p.biases, n, k8 * 8)


def stack_q4(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Q4:
    """Rows of several MLX (weight, scales, biases) with the same K, stacked in order, then tiled."""

    return make_q4(torch.cat([as_i32(p[0]) for p in parts]), torch.cat([p[1] for p in parts]),
                   torch.cat([p[2] for p in parts]))


def to_mlx(q: Q4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return shared.unpack(q)


def dequantize(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Reference: MLX (..., N, K/8) words -> (..., N, K) fp32 values s * q + b."""

    k8 = words.shape[-1]
    w = as_i32(words).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=words.device, dtype=torch.int64) * 4
    q = ((w[..., None] >> shifts) & 0xF).reshape(*words.shape[:-1], k8 * 8).to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(GS, dim=-1)
    b = biases.to(torch.float32).repeat_interleave(GS, dim=-1)
    return q * s + b


def dequantize_q4(q: Q4) -> torch.Tensor:
    w, s, b = to_mlx(q)
    return dequantize(w.contiguous(), s.contiguous(), b.contiguous())


# K split rule for shapes without a table entry: split until the column tiles times slices reach this many programs
SPLIT_TARGET = 192


# Per-shape K slices determine arithmetic equally for every row; group-step, warp, and stage settings preserve bits.
SHAPE_SK: dict[str, int] = {"12576x4096": 4, "4096x4096": 2, "2048x4096": 4, "8192x1536": 4, "8192x512": 4,
                            "4096x8192": 8}


def split_k(n: int, k: int) -> int:
    """Choose K slices from shape alone until column tiles times slices reach SPLIT_TARGET; changing the target changes every row equally."""

    forced = SHAPE_SK.get(f"{n}x{k}")
    if forced:
        return forced
    tiles = -(-n // BN)
    groups = k // GS
    sk = 1
    while sk < 8 and tiles * sk < SPLIT_TARGET and groups % (sk * 2) == 0 and groups // (sk * 2) >= 8:
        sk *= 2
    return sk


def bucket(m: int) -> int:
    for b in (16, 32, 64, 128):
        if m <= b:
            return b
    raise ValueError(f"at most 128 rows, got {m}")


@triton.jit
def _group_sums(X, XS, x_stride, K: tl.constexpr, GB: tl.constexpr):
    m = tl.program_id(0)
    gb = tl.program_id(1)
    KG: tl.constexpr = K // 64
    g = gb * GB + tl.arange(0, GB)
    k = tl.arange(0, 64)
    ok = g < KG
    x = tl.load(X + m * x_stride + g[:, None] * 64 + k[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    tl.store(XS + m * KG + g, tl.sum(x, axis=1), mask=ok)


def group_sums(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """(M, K) bf16 (rows may be strided) -> (M, K/64) fp32 sums of each 64-input group."""

    m, k = x.shape
    kg = k // GS
    if out is None:
        out = torch.empty((m, kg), dtype=torch.float32, device=x.device)
    _group_sums[(m, triton.cdiv(kg, 16))](x, out, x.stride(0), K=k, GB=16, num_warps=2)
    return out


@triton.jit
def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr, F32: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < total
    acc = tl.load(PART + offs, mask=ok, other=0.0)
    for s in tl.static_range(1, SK):
        acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
    if F32:
        tl.store(OUT + offs, acc, mask=ok)
    else:
        tl.store(OUT + offs, acc.to(tl.bfloat16), mask=ok)


@dataclass
class B16:
    """A BF16 matrix [n, k] as the checkpoint stores it (EXL3 checkpoints keep every non-expert weight in BF16)."""

    weight: torch.Tensor      # [n, k] bf16, contiguous
    n: int
    k: int

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()



F8_BLOCK = 128           # fp8 columns a scale covers (the latent path's kv_b rows)


@dataclass
class F8:
    """An FP8 e4m3 matrix [n, k] with an fp32 scale per ``F8_BLOCK`` columns (the latent path's kv_b rows)."""

    weight: torch.Tensor      # [n, k] float8_e4m3fn
    scale: torch.Tensor       # [n, k // F8_BLOCK] fp32
    n: int
    k: int

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size() + self.scale.numel() * self.scale.element_size()


def make_f8(weight: torch.Tensor) -> F8:
    """BF16 rows as e4m3, a scale (the 128-column block's amax / 448) a block; lossy, W ~ q * scale."""

    n, k = weight.shape
    blocks = weight.float().reshape(n, k // F8_BLOCK, F8_BLOCK)
    scale = (blocks.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    q = (blocks / scale).to(torch.float8_e4m3fn).reshape(n, k)
    return F8(q.contiguous(), scale.reshape(n, k // F8_BLOCK).contiguous(), n, k)


def make_b16(weight: torch.Tensor) -> B16:
    w = weight.to(torch.bfloat16).contiguous()
    return B16(w, int(w.shape[0]), int(w.shape[1]))


def quantize4(w: torch.Tensor, chunk: int = 8192) -> Q4:
    """Quantize bf16 weights to tiled affine 4-bit groups of 64 for drafting only, never verification."""

    n, k = w.shape
    words = torch.empty((n, k // 8), dtype=torch.int32, device=w.device)
    scales = torch.empty((n, k // 64), dtype=torch.bfloat16, device=w.device)
    biases = torch.empty_like(scales)
    for r in range(0, n, chunk):
        g = w[r:r + chunk].float().view(-1, k // 64, 64)
        lo, hi = g.amin(-1), g.amax(-1)
        scale = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
        bias = lo.to(torch.bfloat16)
        q = torch.round((g - bias.float()[..., None]) / scale.float()[..., None]).clamp(0, 15).to(torch.int32)
        q = q.view(-1, k // 8, 8)
        part = torch.zeros(q.shape[:2], dtype=torch.int32, device=w.device)
        for j in range(8):
            part |= q[..., j] << (4 * j)
        words[r:r + chunk], scales[r:r + chunk], biases[r:r + chunk] = part, scale, bias
    return make_q4(words, scales.contiguous(), biases.contiguous())


def quantize4_mse(w: torch.Tensor, chunk: int = 4096, grid: int = 16) -> Q4:
    """bf16 weights as affine 4-bit groups of 64, each group's range the one of ``grid`` clippings (1.0 down to 0.55
    of its min / max) with the least squared error, scale and bias as stored (bf16). Lossy."""

    return make_q4(*quantize4_mse_raw(w, chunk, grid))


def quantize4_mse_raw(w: torch.Tensor, chunk: int = 4096,
                      grid: int = 16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``quantize4_mse``'s groups in MLX's layout, untiled: words [n, k / 8] int32 (8 nibbles low to high), scales and
    biases [n, k / 64] bf16 (what ``latent.AbsorbQ4`` reads)."""

    n, k = w.shape
    words = torch.empty((n, k // 8), dtype=torch.int32, device=w.device)
    scales = torch.empty((n, k // 64), dtype=torch.bfloat16, device=w.device)
    biases = torch.empty_like(scales)
    for r in range(0, n, chunk):
        g = w[r:r + chunk].float().view(-1, k // 64, 64)
        lo, hi = g.amin(-1), g.amax(-1)
        best_s, best_b, best_q, best_err = None, None, None, None
        for t in range(grid):                                   # each group's range 1.0 .. 0.55 of its min / max
            f = 1.0 - 0.45 * t / (grid - 1)
            s = ((hi - lo) * f / 15).clamp_min(1e-8).to(torch.bfloat16).float()
            b = lo.to(torch.bfloat16).float()
            q = torch.round((g - b[..., None]) / s[..., None]).clamp(0, 15)
            err = ((q * s[..., None] + b[..., None] - g) ** 2).sum(-1)
            if best_err is None:
                best_s, best_b, best_q, best_err = s, b, q, err
                continue
            better = err < best_err
            best_s = torch.where(better, s, best_s)
            best_b = torch.where(better, b, best_b)
            best_q = torch.where(better[..., None], q, best_q)
            best_err = torch.where(better, err, best_err)
        q = best_q.view(-1, k // 8, 8).to(torch.int32)
        part = torch.zeros(q.shape[:2], dtype=torch.int32, device=w.device)
        for j in range(8):
            part |= q[..., j] << (4 * j)
        words[r:r + chunk], scales[r:r + chunk], biases[r:r + chunk] = part, best_s, best_b
    return words, scales.contiguous(), biases.contiguous()


def stack_b16(parts: list[torch.Tensor]) -> B16:
    """Rows of several BF16 matrices with the same K, stacked in order."""

    return make_b16(torch.cat([p.to(torch.bfloat16) for p in parts]))


# BF16 matmuls: columns and K per step; the K slices come from ``split_k`` as for Q4 (fixed by the shape)
B16_BN, B16_BK = 64, 64


@triton.jit
def _bmm(X, W, OUT, PART, M, x_stride, N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
         BLOCK_N: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
    """Multiply bf16 rows over a fixed K slice in order with fp32 sums, independently of other rows."""

    PER: tl.constexpr = K // SK
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_s * PER, pid_s * PER + PER, BK):
        x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
        acc = acc + tl.dot(x, tl.trans(w))
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        if F32:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


# (warps, stages) for the BF16 matmul by row bucket: no choice changes bits
B16_CONFIG = {16: (4, 3), 32: (4, 3), 64: (4, 2), 128: (8, 2)}


def matmul(x: torch.Tensor, q: Q4 | B16, xs: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
           f32: bool = False, part: torch.Tensor | None = None) -> torch.Tensor:
    """Multiply strided bf16 x by Q4 or BF16 q.T, returning bf16 or unrounded fp32 with f32; BF16 weights ignore xs."""

    if isinstance(q, B16):
        return _matmul_b16(x, q, out=out, f32=f32, part=part)
    if x.shape[1] != q.k or x.stride(1) != 1 or x.dtype != torch.bfloat16:
        raise ValueError(f"matmul: x {tuple(x.shape)} {x.dtype} does not match K={q.k}")
    if out is not None and (out.shape != (x.shape[0], q.n) or not out.is_contiguous()):
        raise ValueError(f"matmul: out {tuple(out.shape)} must be a contiguous ({x.shape[0]}, {q.n})")
    return shared.matmul(x, q, group_sums(x) if xs is None else xs, sk=split_k(q.n, q.k), f32=f32, out=out)


def b16_split_k(n: int, k: int) -> int:
    """K slices of a BF16 matmul: like ``split_k``, fixed by the shape, in units of B16_BK."""

    tiles = -(-n // B16_BN)
    steps = k // B16_BK
    sk = 1
    while sk < 8 and tiles * sk < SPLIT_TARGET and steps % (sk * 2) == 0 and steps // (sk * 2) >= 4:
        sk *= 2
    return sk


def _matmul_b16(x: torch.Tensor, q: B16, *, out: torch.Tensor | None, f32: bool,
                part: torch.Tensor | None) -> torch.Tensor:
    m, k = x.shape
    if k != q.k or x.stride(1) != 1 or x.dtype != torch.bfloat16 or k % B16_BK:
        raise ValueError(f"matmul: x {tuple(x.shape)} {x.dtype} does not match K={q.k}")
    bm = bucket(min(m, 128))              # a prompt chunk runs as 128-row blocks: no bucket changes a row's bits
    warps, stages = B16_CONFIG[bm]
    sk = b16_split_k(q.n, q.k)
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, q.n) or not out.is_contiguous():
        raise ValueError(f"matmul: out {tuple(out.shape)} must be a contiguous ({m}, {q.n})")
    if sk > 1:
        need = sk * m * q.n
        if part is None or part.numel() < need:
            part = torch.empty((need,), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm), triton.cdiv(q.n, B16_BN), sk)
    _bmm[grid](x, q.weight, out, part if sk > 1 else out, m, x.stride(0), N=q.n, K=k, SK=sk, BM=bm,
               BLOCK_N=B16_BN, BK=B16_BK, F32=f32, num_warps=warps, num_stages=stages)
    if sk > 1:
        total = m * q.n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out


KVB_KINDS = ("bf16", "fp8", "q4")


def kvb_kind(env=None) -> str:
    """TF_GLM_KVB: how an EXL3 checkpoint's kv_b_proj is held for the latent path's absorb and expand (the kernels
    every decode round runs once a DSA layer): bf16 (as stored, the default), fp8 (e4m3, an fp32 scale per row and
    128 latent columns, ``make_f8``'s layout) or q4 (MSE-searched affine groups of 64 along the latent, the MLX
    checkpoint's kernels). fp8 and q4 are lossy, like the rest of TF_GLM_DENSE; a row's bits still never depend on its
    window's other rows, so drafted replies equal serial ones under each."""

    import os

    kind = (os.environ if env is None else env).get("TF_GLM_KVB", "bf16").strip() or "bf16"
    if kind not in KVB_KINDS:
        raise ValueError(f"TF_GLM_KVB is bf16, fp8 or q4, not {kind!r}")
    return kind


def kvb_weights(base, kind: str):
    """A split_weights transform for TF_GLM_KVB: kv_b_proj at a byte a value plus fp32 scales a 128 block (fp8), or
    half a byte plus a bf16 scale and bias a group of 64 (q4); the latent path then holds no other copy of it."""

    def transform(name: str, info: dict) -> tuple[int, int]:
        total, extra = base(name, info)
        shape = info.get("shape") or []
        if kind == "bf16" or not total or "kv_b_proj" not in name or info.get("dtype") != "BF16" or len(shape) != 2:
            return total, extra
        n = total // 2
        if kind == "fp8" and shape[-1] % F8_BLOCK == 0:
            return n + n // F8_BLOCK * 4, extra
        if kind == "q4" and shape[-1] % 64 == 0:
            return n // 2 + n // 64 * 4, extra
        return total, extra
    return transform
