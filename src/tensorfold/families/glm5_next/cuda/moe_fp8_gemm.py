"""The grouped fp8 GEMM's Triton kernel: one launch per expert GEMM (three a layer), no host sync.

Rows arrive sorted by expert (``routed``'s ``order``), so each row block can be pinned to one expert: the
host-side prep pads every segment to a BLOCK_P multiple (``padded_offs``, plain device ops, capture-legal),
and each program finds its expert by counting the padded segment ends at or below its first row. Blocks are
then a plain masked K loop (loads pipeline), segments' ragged tails masked out, and the launch grid is the
no-sync upper bound ``P + E * BLOCK_P`` rows - the programs past the padded total exit on one comparison.
The B tiles are the ``dequant_fp8`` kernel's e4m3 output ([N, K] K-major per expert, gate and up stacked in
one [E, 2I, D] buffer), loaded as uint8 and bitcast to e4m3 in registers. ``tl.dot`` runs on the fp8
operands where the Triton build supports it on this GPU (probed once, before capture via ``prime``);
otherwise both operands dequant to bf16 right after the load and the win is the one-launch grouping alone.
Scales are TensorWise singletons, folded into the fp32 accumulator after the K loop (slightly tighter than
dequantizing before the dot). The tiles (2.5 to 5 GiB a GEMM for 5.3-Flash's shapes) put this GEMM on the
memory-bandwidth wall - measured at the copy ceiling on GB10, at parity with the per-expert cuBLAS loop.
"""

from __future__ import annotations

import torch

BLOCK_P = 64
BLOCK_N = 256
BLOCK_K = 128
NUM_WARPS = 8
NUM_STAGES = 3

_FP8_DOT: bool | None = None            # whether tl.dot takes the e4m3 operands on this GPU (probed once)


def _jit():
    import triton
    import triton.language as tl

    @triton.jit
    def _grouped_fp8_gemm(a_ptr, sa_ptr, b_ptr, sb_ptr, offs_ptr, pad_ptr, out_ptr,
                          E, P, K, N, s_a, s_be, s_bn,
                          BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          FP8_DOT: tl.constexpr):
        b0 = tl.program_id(0) * BLOCK_P
        col0 = tl.program_id(1) * BLOCK_N
        ei = tl.arange(0, 512)                          # E <= 512: the padded ends, one block each
        pcv = tl.load(pad_ptr + ei, mask=ei < E, other=1 << 30)
        e = tl.sum((pcv <= b0).to(tl.int32))            # the segment holding row b0 (E means past the end)
        if e >= E:
            return
        start = tl.load(offs_ptr + e - 1, mask=e > 0, other=0)
        end = tl.load(offs_ptr + e)
        seg0 = tl.load(pad_ptr + e - 1, mask=e > 0, other=0)     # the segment's padded start
        rp = tl.arange(0, BLOCK_P)
        rn = tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        rows = start + (b0 - seg0) + rp
        cols = col0 + rn
        rmask = rows < end
        acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
        base = b_ptr + e.to(tl.int64) * s_be     # E * 2I * D passes 2^31 for the real shapes
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + rk
            a = tl.load(a_ptr + rows[:, None] * s_a + ks[None, :], mask=rmask[:, None], other=0)
            b = tl.load(base + (cols[None, :] * s_bn) + ks[:, None], mask=cols[None, :] < N, other=0)
            if FP8_DOT:
                acc = tl.dot(a.to(tl.float8e4nv, bitcast=True), b.to(tl.float8e4nv, bitcast=True), acc)
            else:
                acc = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
        scale = tl.load(sa_ptr) * tl.load(sb_ptr)
        tl.store(out_ptr + rows[:, None] * N + cols[None, :], (acc * scale).to(tl.bfloat16),
                 mask=rmask[:, None] & (cols < N)[None, :])

    return _grouped_fp8_gemm


def _errors():
    import triton
    return (RuntimeError, ValueError, triton.errors.CompilationError)


def grouped(a8: torch.Tensor, sa: torch.Tensor, b8: torch.Tensor, sb: torch.Tensor, offs: torch.Tensor,
            E: int) -> torch.Tensor:
    """The grouped GEMM: a8 [P, K] e4m3 (as uint8), b8 [E, N, K] e4m3 uint8 (K-major rows per expert),
    offs [E] int32 cumulative row ends -> [P, N] bf16. Scales are fp32 singletons, folded post-accumulate."""

    import triton

    global _FP8_DOT
    assert E <= 512, E                       # the kernel's expert table is a fixed 512-wide load
    P, K = a8.shape
    N = b8.shape[1]
    sizes = offs - torch.cat((offs.new_zeros(1), offs[:-1]))
    padded = ((sizes + (BLOCK_P - 1)) // BLOCK_P * BLOCK_P).cumsum(0).to(torch.int32)
    out = torch.empty((P, N), dtype=torch.bfloat16, device=a8.device)
    kern = _jit()
    grid = (triton.cdiv(P + E * BLOCK_P, BLOCK_P), triton.cdiv(N, BLOCK_N))
    args = (a8, sa, b8, sb, offs, padded, out, E, P, K, N, a8.stride(0), b8.stride(0), b8.stride(1))
    if _FP8_DOT is None:                    # one probe on tiny dummy tiles: keeps the choice off the hot path
        try:
            _probe(kern, a8.device, True)
            _FP8_DOT = True
        except _errors():
            _FP8_DOT = False
    kern[grid](*args, BLOCK_P, BLOCK_N, BLOCK_K, _FP8_DOT, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return out


def _probe(kern, device, fp8_dot: bool) -> None:
    """A tiny launch that just has to compile and run: decides fp8-dot support without touching real data."""

    import triton

    P, K, N, E = BLOCK_P, 4 * BLOCK_K, BLOCK_N, 1
    a = torch.zeros((P, K), dtype=torch.uint8, device=device)
    b = torch.zeros((E, N, K), dtype=torch.uint8, device=device)
    offs = torch.full((E,), P, dtype=torch.int32, device=device)
    out = torch.empty((P, N), dtype=torch.bfloat16, device=device)
    sa = torch.ones(1, dtype=torch.float32, device=device)
    kern[(1, 1)](a, sa, b, sa, offs, offs, out, E, P, K, N, a.stride(0), b.stride(0), b.stride(1),
                 BLOCK_P, BLOCK_N, BLOCK_K, fp8_dot, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    torch.cuda.synchronize()


def prime(device) -> None:
    """Compile the kernel and settle the fp8-dot choice before any CUDA graph capture (engine init)."""

    grouped(torch.zeros((BLOCK_P, 4 * BLOCK_K), dtype=torch.uint8, device=device),
            torch.ones(1, dtype=torch.float32, device=device),
            torch.zeros((1, BLOCK_N, 4 * BLOCK_K), dtype=torch.uint8, device=device),
            torch.ones(1, dtype=torch.float32, device=device),
            torch.full((1,), BLOCK_P, dtype=torch.int32, device=device), 1)
