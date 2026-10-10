"""The fused EXL3-decode + grouped fp8 GEMM: one launch per expert GEMM that reads the trellises directly.

The staged route (``moe_fp8`` + ``moe_fp8_gemm``) decodes each layer's three expert matrices into e4m3 B
tiles (writing ~4.8 GB a layer per rank) and then reads them back in the GEMM - the tile traffic, not the
trellis read, is why that route cannot beat the bf16 GEMV. This kernel drops the staging step: each program
decodes the B tile it needs straight from the expert's trellis (ExLlamaV3's 16x16 tile bitstream) and feeds
it to the same ``tl.dot`` schedule, so the only B traffic is the trellis itself (~2.75 GiB a layer per rank),
read once per row block instead of once per staging plus once per GEMM.

Per B element (row k, column n), with (r, c) = (k % 16, n % 16) inside its 16x16 tile:

    p    = 32*(c & 7) + 8*((r & 7) >> 1) + 4*(c >> 3) + 2*(r >> 3) + (r & 1)   (the value's stream index)
    st   = 16 bits ending at E(p) in the tile's ringed bitstream (``exl3_format.states``' u32 formula)
    W    = codebook(st)                                                        (pure ALU, no LUT)
    B8   = e4m3(W * inv)                                                       (inv = 448 / codebook_max)

The 16-bit extraction runs on uint32 words (a tile is 4 * k2 of them, int16 little-endian pairs) exactly as
``states()`` does: ``((words[i0] << 32 | words[i1]) >> (48 - off)) & 0xFFFF``, which collapses to a single
word shift whenever off <= 16 (the other half-load is masked off). The (i0, off, i1) windows depend only on
the value's stream index p and the expert's K2 (half-bits), so they live in one packed [15, 256] uint32
table (i0 | off << 6 | i1 << 11); each program loads its expert's row once. The row schedule, ragged-tail
masking and TensorWise scale folding are the staged kernel's (``moe_fp8_gemm``), unchanged.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import torch

BLOCK_P = 128
BLOCK_N = 128                # event-timed best on GB10 (cc 12.1); 256 costs ~2x despite fewer A reloads
BLOCK_K = 64                # four 16x16 k tiles per iteration; the B decode is per element, no staging
NUM_WARPS = 8
NUM_STAGES = 1              # double-buffering the stage measured slower in the config sweep

FP8_MAX = 448.0

_FP8_DOT: bool | None = None


def _jit():
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_fp8_gemm(a_ptr, sa_ptr, sb_ptr, tp_ptr, k2_ptr, tab_ptr, offs_ptr, pad_ptr, out_ptr,
                        E, K, N, s_a, rtiles, inv,
                        BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, WW: tl.constexpr,
                        CB: tl.constexpr, FP8_DOT: tl.constexpr):
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
        # this expert's trellis: the int64 byte base as uint32 words (a tile is 4 * k2 of them, 4-byte aligned)
        k2 = tl.load(k2_ptr + e)
        tw = tl.load(tp_ptr + e).to(tl.pointer_type(tl.uint32))
        # the value's stream index p from the position in the 16x16 tile, then the packed window table:
        # i0 | off << 6 | i1 << 11, row k2 - 2 of [15, 256] (K2 half-bits 2..16)
        r16 = (rk % 16)[:, None]
        c16 = ((col0 + rn) % 16)[None, :]
        pv = 32 * (c16 & 7) + 8 * ((r16 & 7) >> 1) + 4 * (c16 >> 3) + 2 * (r16 >> 3) + (r16 & 1)
        pk = tl.load(tab_ptr + (k2 - 2) * 256 + pv)
        i0 = pk & 63
        off = (pk >> 6) & 31
        # a value is the 16-bit window ending (16 - off) bits into word i0, four-bit aligned, so it always
        # sits inside the sliding u64 (word i0, word i0 + 1 mod 4 * k2): stage each k iteration's tiles with
        # two coalesced u32 loads (the second shifted by a word, wrapped) and gather one sliding u64 per
        # element -- one shared-memory gather instead of two scattered global loads per element
        NKT: tl.constexpr = BLOCK_K // 16
        NNT: tl.constexpr = BLOCK_N // 16
        NT: tl.constexpr = NKT * NNT
        tilef = tl.reshape((rk // 16)[:, None] * NNT + (rn // 16)[None, :], (BLOCK_K * BLOCK_N,))
        acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + rk
            a = tl.load(a_ptr + rows[:, None] * s_a + ks[None, :], mask=rmask[:, None])
            e1 = tl.arange(0, NT * WW)                       # the stage: NT tiles x WW sliding-u64 slots
            tt = e1 // WW
            pp = e1 % WW
            wbase = ((k0 // 16 + tt // NNT) * rtiles + col0 // 16 + (tt % NNT)) * (4 * k2) + pp
            wa = tl.load(tw + wbase, mask=pp < 4 * k2, other=0)
            wb = tl.load(tw + wbase + 1 - (pp == 4 * k2 - 1).to(tl.int32) * 4 * k2, mask=pp < 4 * k2, other=0)
            flat = wa.to(tl.uint64) | (wb.to(tl.uint64) << 32)
            wg = tl.reshape(tl.gather(flat, (tilef * WW + tl.reshape(i0, (BLOCK_K * BLOCK_N,))).to(tl.int32), 0),
                            (BLOCK_K, BLOCK_N))
            st = tl.where(off <= 16, (wg >> (16 - off)) & 0xFFFF,
                          (((wg & 0xFFFF) << tl.maximum(off - 16, 0)) | (wg >> tl.minimum(80 - off, 63)))
                          & 0xFFFF).to(tl.uint32)
            if CB == 2:                                 # mul1: the dp4a byte sum, the hfma constants
                x = st * 0x83DCD12D
                sm = ((x & 0xFF).to(tl.int32) + ((x >> 8) & 0xFF).to(tl.int32)
                      + ((x >> 16) & 0xFF).to(tl.int32) + ((x >> 24) & 0xFF).to(tl.int32)
                      + 0x6400)                         # unsigned bytes: 0x6400 lands the sum at fp16 1024 + sum
                hv = (sm & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
                sc = tl.full((1,), 0x1EEE, tl.uint16).to(tl.float16, bitcast=True)
                bi = tl.full((1,), 0xC931, tl.uint16).to(tl.float16, bitcast=True)
                v = (hv * sc + bi).to(tl.float16)
            else:                                       # 3inst / mcg: the mask-flip pair sum
                if CB == 0:
                    x = st * 89226354 + 64248484
                else:
                    x = st * 0xCBAC1FED
                x = (x & 0x8FFF8FFF) ^ 0x3B603B60
                lo = (x & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
                hi = (x >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
                v = (lo.to(tl.float32) + hi.to(tl.float32)).to(tl.float16)
            b = (v.to(tl.float32) * inv).to(tl.float8e4nv)
            af = a.to(tl.float8e4nv, bitcast=True)      # A arrives as fp8 or raw uint8 bytes; never a value cast
            if FP8_DOT:
                acc = tl.dot(af, b, acc)
            else:
                acc = tl.dot(af.to(tl.bfloat16), b.to(tl.bfloat16), acc)
        scale = tl.load(sa_ptr) * tl.load(sb_ptr)
        tl.store(out_ptr + rows[:, None] * N + cols[None, :], (acc * scale).to(tl.bfloat16),
                 mask=rmask[:, None] & (cols < N)[None, :])

    return _fused_fp8_gemm


def _errors():
    import triton
    return (RuntimeError, ValueError, triton.errors.CompilationError)


def _packed_rows() -> np.ndarray:
    """The [15, 256] uint32 window table, row k2 - 2: i0 | off << 6 | i1 << 11 (``states()``' indices)."""

    rows = np.zeros((15, 256), dtype=np.uint32)
    for k2 in range(2, 17):
        p1 = np.arange(1, 257, dtype=np.int64)
        ends = p1 * (k2 // 2) if k2 % 2 == 0 else (p1 * k2 - (p1 % 2)) // 2   # stream_ends, in half-bit units
        first = ends - 16 + 128 * k2                    # the window's first bit, made non-negative
        nw = 4 * k2
        i0 = (first // 32) % nw
        off = first % 32
        i1 = (i0 + 1) % nw
        rows[k2 - 2] = i0 | (off << 6) | (i1 << 11)
    return rows


_TABLES: dict = {}


def tables(device) -> torch.Tensor:
    """The packed window table on ``device`` (one [256] row per K2 in 2..16)."""

    t = _TABLES.get(device)
    if t is None:
        t = torch.from_numpy(_packed_rows()).to(device)
        _TABLES[device] = t
    return t


def grouped(a8: torch.Tensor, sa: torch.Tensor, ptrs: torch.Tensor, k2s: torch.Tensor, rtiles: int, inv: float,
            cb: int, sb: torch.Tensor, offs: torch.Tensor, E: int, N: int) -> torch.Tensor:
    """The fused grouped GEMM: a8 [P, K] e4m3 (fp8 or uint8), ptrs [E] int64 trellis byte bases, k2s [E] int32
    half-bits, rtiles the trellis' row stride in 16-column tiles (``gu_stride`` or N >> 4), offs [E] int32
    cumulative row ends -> [P, N] bf16. Scales are fp32 singletons, folded post-accumulate. N must be a
    BLOCK_N multiple and K a BLOCK_K multiple (the family's shapes are)."""

    import triton

    global _FP8_DOT
    assert E <= 512, E                       # the kernel's expert table is a fixed 512-wide load
    P, K = a8.shape
    assert N % BLOCK_N == 0 and K % BLOCK_K == 0, (N, K)
    sizes = offs - torch.cat((offs.new_zeros(1), offs[:-1]))
    padded = ((sizes + (BLOCK_P - 1)) // BLOCK_P * BLOCK_P).cumsum(0).to(torch.int32)
    out = torch.empty((P, N), dtype=torch.bfloat16, device=a8.device)
    kern = _jit()
    grid = (triton.cdiv(P + E * BLOCK_P, BLOCK_P), triton.cdiv(N, BLOCK_N))
    assert N % BLOCK_N == 0 and K % BLOCK_K == 0, (N, K)
    kern[grid](a8, sa, sb, ptrs, k2s, tables(a8.device), offs, padded, out,
               E, K, N, a8.stride(0), rtiles, inv, BLOCK_P, BLOCK_N, BLOCK_K,
               32 if int(k2s.max()) <= 8 else 64, cb, _FP8_DOT,
               num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return out


def cuda_codebook(cb: int) -> np.ndarray:
    """The [65536] fp16 state table per the CUDA kernels' arithmetic (``cb_pair`` in experts_grouped.cuh).

    Bit-identical to ``exl3_format.codebook`` for every codebook: mul1's dp4a sums UNSIGNED bytes and the
    +0x6400 lands the sum at fp16 1024 + sum, and the hfma is one fp16 rounding — the same single rounding
    the float64 numpy formula makes."""

    from tensorfold.cuda.exl3 import format as exl3_format
    if cb == 2:
        return exl3_format.codebook("mul1")
    return exl3_format.codebook(("3inst", "mcg")[cb])


def tile_reference(trellis, k2: int, cb: int, inv: float) -> torch.Tensor:
    """The fused kernel's B tile as uint8 bytes [N, K] (e4m3 bits, column-major; tests and ``prime``)."""

    from tensorfold.cuda.exl3 import format as exl3_format
    st = exl3_format.states(np.ascontiguousarray(trellis), k2 / 2)      # [.., 256], the kernels' formula
    q = cuda_codebook(cb)[st.reshape(-1)].reshape(st.shape)
    kt, nt = st.shape[0], st.shape[1]
    r, c = exl3_format.tile_positions()
    w = np.zeros((kt, 16, nt, 16), dtype=np.float16)
    w[np.arange(kt)[:, None, None], r, np.arange(nt)[None, :, None], c] = q.reshape(kt, nt, 256)
    q8 = torch.from_numpy(w.reshape(kt * 16, nt * 16)).float() * np.float32(inv)  # e4m3(fp32(h) * inv), as the kernels
    return q8.to(torch.float8_e4m3fn).t().contiguous().view(torch.uint8)


def _verify(device, cb: int, k2: int = 8) -> None:
    """One tiny GEMM against the numpy decode: the whole kernel path (pointer cast, windows, codebook, dot)."""

    D, I, E = BLOCK_N, BLOCK_N, 3
    rng = np.random.default_rng(cb + 7)
    inv = FP8_MAX / float(abs(cuda_codebook(cb)).max())
    tgs = [torch.from_numpy(rng.integers(0, 1 << 16, size=(D // 16, I // 16, 8 * k2), dtype=np.uint16)
                            .astype(np.int16)).to(device).contiguous() for _ in range(E)]
    ptrs = torch.tensor([t.data_ptr() for t in tgs], dtype=torch.int64, device=device)
    k2s = torch.full((E,), k2, dtype=torch.int32, device=device)
    a = (torch.randn(BLOCK_P, D, device=device) / 8).to(torch.float8_e4m3fn)
    sa = torch.full((1,), 0.05, dtype=torch.float32, device=device)
    sb = torch.full((1,), 1.0 / inv, dtype=torch.float32, device=device)
    offs = torch.tensor([64, 64, BLOCK_P], dtype=torch.int32, device=device)    # the middle expert is empty
    out = grouped(a, sa, ptrs, k2s, I >> 4, inv, cb, sb, offs, E, I)
    bref = torch.stack([tile_reference(t.cpu().numpy(), k2, cb, inv) for t in tgs]).to(device)  # [E, N, K] uint8
    z = torch.zeros(BLOCK_P, I, dtype=torch.float32, device=device)
    st = 0
    for e in range(E):
        en = int(offs[e])
        if en > st:
            z[st:en] = (a[st:en].float() @ bref[e].view(torch.float8_e4m3fn).t().float()) * (sa * sb).item()
        st = en
    n = int(offs[-1])
    err = (out[:n].float() - z[:n]).abs().max().item() / max(z[:n].abs().max().item(), 1e-9)
    assert err < 0.02, f"cb {cb} k2 {k2}: the fused kernel differs from the reference by {err:.4f}"


def _probe_fp8_dot(device) -> None:
    """Compile and run the fp8-dot variant once on a dummy launch (decides the dot dtype before capture)."""

    import triton
    k2 = 8
    P, K, N, E = BLOCK_P, BLOCK_K, BLOCK_N, 1
    trellis = torch.zeros((1, (N // 16) * (K // 16) * 8 * k2), dtype=torch.int16, device=device)
    ptrs = torch.tensor([trellis.data_ptr()], dtype=torch.int64, device=device)
    k2s = torch.full((E,), k2, dtype=torch.int32, device=device)
    a = torch.zeros((P, K), dtype=torch.uint8, device=device)
    offs = torch.full((E,), P, dtype=torch.int32, device=device)
    out = torch.empty((P, N), dtype=torch.bfloat16, device=device)
    kern = _jit()
    kern[(1, 1)](a, torch.ones(1, dtype=torch.float32, device=device),
                 torch.ones(1, dtype=torch.float32, device=device), ptrs, k2s, tables(device), offs, offs,
                 out, E, K, N, a.stride(0), N >> 4, 1.0, BLOCK_P, BLOCK_N, BLOCK_K, 32, 0, True,
                 num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    torch.cuda.synchronize()


def prime(device) -> bool:
    """Compile the kernel (all three codebooks), settle the fp8-dot choice and verify one tiny GEMM per
    codebook against the numpy decode. False means the fused path is unusable and the staged tiles take
    over; the engine calls this at Buffers creation, so first use is never inside a CUDA graph capture."""

    global _FP8_DOT
    if (not torch.cuda.is_available() or torch.device(device).type != "cuda"
            or importlib.util.find_spec("triton") is None):
        return False
    try:
        _jit()
        if _FP8_DOT is None:
            try:
                _probe_fp8_dot(device)
                _FP8_DOT = True
            except _errors():
                _FP8_DOT = False
        for cb in (0, 1, 2):
            _verify(device, cb)
        return True
    except Exception:
        if _FP8_DOT is True:            # a broken fp8 dot can also fail the verify: retry bf16 dots
            _FP8_DOT = False
            try:
                for cb in (0, 1, 2):
                    _verify(device, cb)
                return True
            except Exception:
                pass
        return False
