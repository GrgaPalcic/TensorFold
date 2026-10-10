"""GLM's MLA attention on the 512-wide latent cache; kernels compute each row alone, so window rows keep serial bits (docs/recipes/glm-5.3-flash.md).

The cache stores one row a token and DSA layer as bf16, or quantized: fp8 (e4m3fn codes with a per-token f32
scale, amax / 448) or q4 — 256 uint8 bytes a row, two signed 4-bit codes a byte (element 2k in the low nibble,
2k + 1 in the high, each nibble the code's two's complement), plus a per-token f32 power-of-two scale in
``lat_scales``: s = 2^ceil(log2(amax / 7)) (1.0 for an all-zero row), codes floor(x / s + 0.5) clamped to
[-8, 7]. The row's largest value never clips (amax <= 7 s; the fp32 log2's rounding aside, where the clamp
bounds the loss at s), so decode is x ~= s * code, read as bf16 into the attention's dots with the scale folded
into the scores like the fp8 cache's."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import LATENT as ENABLED     # off (TF_GLM_LATENT=0): per-head keys and values, 0.3.5's path, for A/B

L = 512            # GLM-5.3-Flash's latent width (kv_lora_rank); the kernels take the width from the tensors
CHUNK = 512        # keys per chunk program, merged in absolute order
KT = 32            # keys per tensor-core tile
HB = 16            # heads per attention tile (all from one row)
HB_WIDE = 32       # prefill chunks: a rank's 32 heads in one tile read each key once (same bits as 16, tested)


def head_block(R: int) -> int:
    """Heads per attention program: a rank's 32 in prompt chunks (keys read once), 16 in decode windows; 16- and 32-row tiles give the same bits."""
    return HB_WIDE if R >= 64 else HB


# ------------------------------------------------------------------------------------------------ weights ---

def dequant_mlx4(w: torch.Tensor, s: torch.Tensor, b: torch.Tensor, group: int = 64) -> torch.Tensor:
    """MLX affine 4-bit rows -> fp32 [out, in]: w uint32 [out, in / 8], 8 nibbles low to high; s, b [out, in / group]."""

    out, words = w.shape
    shifts = torch.arange(0, 32, 4, device=w.device, dtype=torch.int32)
    q = (w.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF                        # [out, words, 8]
    q = q.reshape(out, words * 8).to(torch.float32)
    s = s.to(torch.float32).repeat_interleave(group, dim=1)
    b = b.to(torch.float32).repeat_interleave(group, dim=1)
    return q * s + b


class AbsorbW:
    """One DSA layer's kv_b_proj split per head for the latent path: wk [H, 256, 512], wv [H, 256, 512] bf16."""

    def __init__(self, wk: torch.Tensor, wv: torch.Tensor) -> None:
        if wk.dim() != 3 or wv.dim() != 3 or wk.shape[2] != wv.shape[2]:
            raise ValueError("AbsorbW: expected [heads, dim, latent] blocks")
        self.wk = wk.to(torch.bfloat16).contiguous()
        self.wv = wv.to(torch.bfloat16).contiguous()
        self.heads, self.qk_dim, self.lw = self.wk.shape
        self.v_dim = self.wv.shape[1]

    @classmethod
    def from_rows(cls, k_rows: torch.Tensor, v_rows: torch.Tensor, heads: int) -> "AbsorbW":
        """k_rows [heads * qk_dim, latent], v_rows [heads * v_dim, latent], float, in head order."""
        lw = k_rows.shape[1]
        return cls(k_rows.reshape(heads, -1, lw), v_rows.reshape(heads, -1, lw))

    def nbytes(self) -> int:
        return self.wk.numel() * 2 + self.wv.numel() * 2


class AbsorbQ4:
    """kv_b split per head in MLX's affine 4-bit layout: words [H, dim, latent / 8], scales and biases [H, dim, latent / 64], key rows then value rows."""

    def __init__(self, k: tuple, v: tuple, heads: int) -> None:
        def per_head(t):
            w, sc, b = t
            return (w.reshape(heads, -1, w.shape[1]).contiguous(), sc.reshape(heads, -1, sc.shape[1]).contiguous(),
                    b.reshape(heads, -1, b.shape[1]).contiguous())

        self.wkw, self.wks, self.wkb = per_head(k)
        self.wvw, self.wvs, self.wvb = per_head(v)
        self.heads, self.qk_dim = self.wkw.shape[0], self.wkw.shape[1]
        self.v_dim = self.wvw.shape[1]
        self.lw = self.wkw.shape[2] * 8
        if self.wks.shape[2] * 64 != self.lw or self.wvw.shape[2] * 8 != self.lw:
            raise ValueError("AbsorbQ4: expected groups of 64 along the latent")

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.wkw, self.wks, self.wkb, self.wvw, self.wvs, self.wvb))


F8_BLOCK = 128     # latent columns a scale covers (qmm.make_f8's block)


class AbsorbF8:
    """kv_b split per head as FP8 e4m3 (TF_GLM_KVB=fp8): wk [H, qk_dim, latent], wv [H, v_dim, latent] and fp32
    scales [H, dim, latent / 128], qmm.make_f8's rows as they are (W ~ q * scale). Lossy; half AbsorbW's bytes."""

    def __init__(self, k, v, heads: int) -> None:
        def per_head(q):
            return (q.weight.reshape(heads, -1, q.k).contiguous(),
                    q.scale.reshape(heads, -1, q.k // F8_BLOCK).contiguous())

        self.wk, self.sk = per_head(k)
        self.wv, self.sv = per_head(v)
        self.heads, self.qk_dim, self.lw = self.wk.shape
        self.v_dim = self.wv.shape[1]
        if self.lw % F8_BLOCK or self.wv.shape[2] != self.lw:
            raise ValueError("AbsorbF8: expected 128-column blocks along the latent")

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.wk, self.sk, self.wv, self.sv))


def from_bf16(k_rows: torch.Tensor, v_rows: torch.Tensor, heads: int, kind: str = "bf16"):
    """An EXL3 checkpoint's kv_b rows (bf16, head order: k_rows [heads * qk_dim, latent], v_rows alike) as the latent
    path holds them for TF_GLM_KVB ``kind``: AbsorbW (bf16, as stored), AbsorbF8 (fp8) or AbsorbQ4 (q4, MSE-searched
    groups of 64 along the latent). A latent width the format does not tile stays bf16."""

    from . import qmm

    if kind not in ("bf16", "fp8", "q4"):
        raise ValueError(f"kv_b as bf16, fp8 or q4, not {kind!r}")
    lw = k_rows.shape[1]
    if kind == "fp8" and lw % F8_BLOCK == 0:
        return AbsorbF8(qmm.make_f8(k_rows), qmm.make_f8(v_rows), heads)
    if kind == "q4" and lw % 64 == 0:
        return AbsorbQ4(qmm.quantize4_mse_raw(k_rows), qmm.quantize4_mse_raw(v_rows), heads)
    return AbsorbW.from_rows(k_rows, v_rows, heads)


# ----------------------------------------------------------------------------------------- absorb, expand ---

F8_ABSORB_BN = 64          # FP8 absorb columns per decode program: 64-byte weight rows, inside one scale block
if F8_BLOCK % F8_ABSORB_BN:
    raise ValueError("the FP8 absorb tiles must sit inside 128-column scale blocks")


@triton.jit
def _absorb_q8(Q, WK, SK, QA, R, H: tl.constexpr, D: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """_absorb_q over FP8 weights: the program's [D, BN] block (inside one 128-column scale block) dequantized to
    fp32 once, then each row's one sum over k, whatever R is."""

    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    KB: tl.constexpr = LW // 128
    k = tl.arange(0, D)
    n = n0 + tl.arange(0, BN)
    w = tl.load(WK + (h * D + k[:, None]) * LW + n[None, :]).to(tl.bfloat16).to(tl.float32)     # [D, BN], exact
    s = tl.load(SK + (h * D + k) * KB + n0 // 128)                                              # [D]
    w = w * s[:, None]
    for r in range(R):
        q = tl.load(Q + (r * H + h) * D + k).to(tl.float32)
        acc = tl.sum(q[:, None] * w, axis=0)
        tl.store(QA + (r * H + h) * LW + n, acc.to(tl.bfloat16))


@triton.jit
def _expand_v8(OL, WV, SV, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """_expand_v over FP8 weights: the program's [BN, LW] rows dequantized to fp32 once (a scale a 128 block), then
    each row's output as one fp32 sum over the latent, whatever R is."""

    h = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    KB: tl.constexpr = LW // 128
    k = tl.arange(0, LW)
    w = tl.load(WV + (h * DV + n[:, None]) * LW + k[None, :]).to(tl.bfloat16).to(tl.float32)   # [BN, LW], exact
    s = tl.load(SV + (h * DV + n[:, None]) * KB + (k // 128)[None, :])                          # [BN, LW]
    w = w * s
    for r in range(R):
        x = tl.load(OL + (r * H + h) * LW + k).to(tl.float32)
        acc = tl.sum(w * x[None, :], axis=1)
        tl.store(OUT + (r * H + h) * DV + n, acc.to(tl.bfloat16))



@triton.jit
def _absorb_q(Q, WK, QA, R, H: tl.constexpr, D: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """Program (head, column block): QA[r, h, n] = sum_k Q[r, h, k] WK[h, k, n] for every row r, k in one sum."""

    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    k = tl.arange(0, D)
    n = n0 + tl.arange(0, BN)
    w = tl.load(WK + (h * D + k[:, None]) * LW + n[None, :]).to(tl.float32)            # [D, BN]
    for r in range(R):
        q = tl.load(Q + (r * H + h) * D + k).to(tl.float32)
        acc = tl.sum(q[:, None] * w, axis=0)
        tl.store(QA + (r * H + h) * LW + n, acc.to(tl.bfloat16))


@triton.jit
def _expand_v(OL, WV, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """Program (head, output block): OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k] for every row r."""

    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    k = tl.arange(0, LW)
    n = n0 + tl.arange(0, BN)
    w = tl.load(WV + (h * DV + n[:, None]) * LW + k[None, :]).to(tl.float32)          # [BN, LW]
    for r in range(R):
        o = tl.load(OL + (r * H + h) * LW + k).to(tl.float32)
        acc = tl.sum(w * o[None, :], axis=1)
        tl.store(OUT + (r * H + h) * DV + n, acc.to(tl.bfloat16))


@triton.jit
def _absorb_q4(Q, WW, WS, WB, QA, R, H: tl.constexpr, D: tl.constexpr, LW: tl.constexpr, RB: tl.constexpr):
    """Program (head, 64-latent group, RB rows): QA = Q @ W with W = q * s + b per group, weights loaded once for the block and each row summed alike."""
    h = tl.program_id(0)
    g = tl.program_id(1)
    rb = tl.program_id(2)
    d = tl.arange(0, D)
    j = tl.arange(0, 8)
    shifts = tl.arange(0, 8) * 4
    KW: tl.constexpr = LW // 8
    KG: tl.constexpr = LW // 64
    words = tl.load(WW + (h * D + d[:, None]) * KW + g * 8 + j[None, :])                   # [D, 8]
    qint = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (D, 64)).to(tl.float32)
    sc = tl.load(WS + (h * D + d) * KG + g).to(tl.float32)
    bi = tl.load(WB + (h * D + d) * KG + g).to(tl.float32)
    n = g * 64 + tl.arange(0, 64)
    for i in tl.static_range(RB):
        r = rb * RB + i
        ok = r < R
        qv = tl.load(Q + (r * H + h) * D + d, mask=ok & (d >= 0), other=0).to(tl.float32)
        acc = tl.sum((qv * sc)[:, None] * qint, axis=0) + tl.sum(qv * bi, axis=0)
        tl.store(QA + (r * H + h) * LW + n, acc.to(tl.bfloat16), mask=ok & (n >= 0))


@triton.jit
def _expand_v4(OL, WW, WS, WB, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """Program (head, BN outputs): the 4-bit rows unpacked once to fp32, then each row's output as one fp32 sum over the latent, whatever R is."""
    h = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    KW: tl.constexpr = LW // 8
    KG: tl.constexpr = LW // 64
    kw = tl.arange(0, KW)
    shifts = tl.arange(0, 8) * 4
    words = tl.load(WW + (h * DV + n[:, None]) * KW + kw[None, :])                          # [BN, KW]
    qint = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (BN, KG, 64)).to(tl.float32)
    gi = tl.arange(0, KG)
    sc = tl.load(WS + (h * DV + n[:, None]) * KG + gi[None, :]).to(tl.float32)            # [BN, KG]
    bi = tl.load(WB + (h * DV + n[:, None]) * KG + gi[None, :]).to(tl.float32)
    w = tl.reshape(qint * sc[:, :, None] + bi[:, :, None], (BN, LW))
    k = tl.arange(0, LW)
    for r in range(R):
        x = tl.load(OL + (r * H + h) * LW + k).to(tl.float32)
        acc = tl.sum(w * x[None, :], axis=1)
        tl.store(OUT + (r * H + h) * DV + n, acc.to(tl.bfloat16))


def row_block(R: int) -> int:
    """Rows per program: 1 for decode windows (most parallel), 16 for prefill chunks (weights reused)."""
    return 1 if R <= 16 else 16


def absorb_q(q: torch.Tensor, a, out: torch.Tensor) -> torch.Tensor:
    """q [R, H, qk_dim] bf16 -> out [R, H, latent] bf16."""
    R, H, D = q.shape
    if isinstance(a, AbsorbF8):
        _absorb_q8[(H, a.lw // F8_ABSORB_BN)](q, a.wk, a.sk, out, R, H=H, D=D, LW=a.lw, BN=F8_ABSORB_BN, num_warps=4)
        return out
    if isinstance(a, AbsorbQ4):
        rb = row_block(R)
        _absorb_q4[(H, a.lw // 64, triton.cdiv(R, rb))](q, a.wkw, a.wks, a.wkb, out, R, H=H, D=D, LW=a.lw, RB=rb,
                                                        num_warps=4)
        return out
    BN = 32
    _absorb_q[(H, a.lw // BN)](q, a.wk, out, R, H=H, D=D, LW=a.lw, BN=BN, num_warps=4)
    return out


def expand_v(o_lat: torch.Tensor, a, out: torch.Tensor) -> torch.Tensor:
    """o_lat [R, H, latent] bf16 -> out [R, H, v_dim] bf16."""
    R, H, _ = o_lat.shape
    if isinstance(a, AbsorbF8):
        BN = 16
        _expand_v8[(H, a.v_dim // BN)](o_lat, a.wv, a.sv, out, R, H=H, DV=a.v_dim, LW=a.lw, BN=BN, num_warps=4)
        return out
    if isinstance(a, AbsorbQ4):
        BN = 16
        _expand_v4[(H, a.v_dim // BN)](o_lat, a.wvw, a.wvs, a.wvb, out, R, H=H, DV=a.v_dim, LW=a.lw, BN=BN,
                                        num_warps=4)
        return out
    BN = 16
    _expand_v[(H, a.v_dim // BN)](o_lat, a.wv, out, R, H=H, DV=a.v_dim, LW=a.lw, BN=BN, num_warps=4)
    return out


# ------------------------------------------------------------------------------------------------ caches ---

@triton.jit
def _lat_write(LAT, lat_stride, LC, POS, SC, LW: tl.constexpr, FP8: tl.constexpr, Q4: tl.constexpr):
    r = tl.program_id(0)
    P = tl.load(POS).to(tl.int64)
    if Q4:
        k = tl.arange(0, LW)
        row = tl.load(LAT + r * lat_stride + k).to(tl.float32)
        amax = tl.max(tl.abs(row), 0)
        s = tl.where(amax > 0, tl.exp2(tl.ceil(tl.log2(amax / 7.0))), 1.0)
        tl.store(SC + P + r, s)
        codes = tl.minimum(tl.maximum(tl.floor(row / s + 0.5), -8.0), 7.0).to(tl.int32)
        pairs = tl.reshape(codes, (LW // 2, 2))
        packed = (pairs[:, 0] & 0xF) | (pairs[:, 1] << 4)     # element 2k low, 2k + 1 high, two's complement
        tl.store(LC + (P + r) * (LW // 2) + tl.arange(0, LW // 2), packed.to(tl.uint8))
    else:
        k = tl.arange(0, LW)
        if FP8:
            row = tl.load(LAT + r * lat_stride + k).to(tl.float32)
            amax = tl.max(tl.abs(row), 0)
            s = tl.where(amax > 0, amax / 448.0, 1.0)       # e4m3fn's 448: the row's largest value fills the range
            tl.store(SC + P + r, s)
            tl.store(LC + (P + r) * LW + k, (row / s).to(tl.float8e4nv))
        else:
            tl.store(LC + (P + r) * LW + k, tl.load(LAT + r * lat_stride + k))


def latent_write(lat: torch.Tensor, cache: torch.Tensor, pos: torch.Tensor,
                 scales: torch.Tensor | None = None) -> None:
    """lat [R, latent] bf16 rows into cache slots pos .. pos + R - 1 (pos read on the device); with ``scales``
    (fp32 [capacity]) the cache holds quantized rows: e4m3fn codes with a per-token f32 scale (amax / 448), or
    a uint8 cache: two packed 4-bit codes a byte and a per-token power-of-two scale (the module docstring's q4)."""
    q4 = cache.dtype == torch.uint8
    _lat_write[(lat.shape[0],)](lat, lat.stride(0), cache, pos, scales if scales is not None else cache,
                                LW=lat.shape[1], FP8=scales is not None and not q4, Q4=q4, num_warps=4)

# --------------------------------------------------------------------------------------------- attention ---

@triton.jit
def _tile(q, kv, m, l, o, valid, SCALE: tl.constexpr):
    """One key tile for HB heads of one row: kv [KT, 512] is both key and value (the latent)."""
    scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * SCALE
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _dense_chunks(QA, LC, SC, POS, PO, PM, PL, R, H: tl.constexpr, LW: tl.constexpr, CH: tl.constexpr,
                  SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr, FP8: tl.constexpr,
                  Q4: tl.constexpr):
    """Program (row, head block, chunk): causal attention of HB heads of row r over keys [c CH, (c + 1) CH)."""
    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    P = tl.load(POS)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H                                                      # tile rows past the last head are padding
    k = tl.arange(0, LW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    start = c * CH
    limit = P + r                                                     # keys 0 .. P + r are visible to row r
    if start <= limit:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        for t in range(CH // KTT):
            ki = start + t * KTT + tl.arange(0, KTT)
            ok = ki <= limit
            if FP8:
                # triton 3.8 cannot take an int ``other`` on an fp8 load: mask after the float conversion
                kv = tl.load(LC + ki[:, None].to(tl.int64) * LW + k[None, :], mask=ok[:, None], other=0.0)
                kv = (kv.to(tl.float32) * tl.load(SC + ki, mask=ok, other=1.0)[:, None])
                kv = tl.where(ok[:, None], kv, 0.0).to(tl.bfloat16)
            elif Q4:
                kb = tl.arange(0, LW // 2)
                codes = tl.load(LC + ki[:, None].to(tl.int64) * (LW // 2) + kb[None, :], mask=ok[:, None], other=0)
                n = codes.to(tl.int32)
                lo = ((n & 0xF) ^ 8) - 8                     # a nibble's two's complement: 0..15 -> -8..7
                hi = ((n >> 4) ^ 8) - 8
                kv = (tl.reshape(tl.join(lo, hi), (KTT, LW)).to(tl.float32)
                      * tl.load(SC + ki, mask=ok, other=1.0)[:, None])
                kv = tl.where(ok[:, None], kv, 0.0).to(tl.bfloat16)
            else:
                kv = tl.load(LC + ki[:, None].to(tl.int64) * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            m, l, o = _tile(q, kv, m, l, o, ok, SCALE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _sparse_chunks(QA, LC, SC, TOK, CNT, PO, PM, PL, R, W: tl.constexpr, H: tl.constexpr, LW: tl.constexpr,
                   CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr, FP8: tl.constexpr,
                   Q4: tl.constexpr):
    """Program (row, head block, chunk): HB heads of row r over its selected tokens [c CH, (c + 1) CH) in list order."""
    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(CNT + r)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H
    k = tl.arange(0, LW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    if c * CH < n:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        for t in range(CH // KTT):
            idx = c * CH + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            tok = tl.load(TOK + r * W + idx, mask=ok, other=0).to(tl.int64)
            if FP8:
                kv = tl.load(LC + tok[:, None] * LW + k[None, :], mask=ok[:, None], other=0.0)
                kv = (kv.to(tl.float32) * tl.load(SC + tok, mask=ok, other=1.0)[:, None])
                kv = tl.where(ok[:, None], kv, 0.0).to(tl.bfloat16)
            elif Q4:
                kb = tl.arange(0, LW // 2)
                codes = tl.load(LC + tok[:, None] * (LW // 2) + kb[None, :], mask=ok[:, None], other=0)
                w = codes.to(tl.int32)
                lo = ((w & 0xF) ^ 8) - 8                     # a nibble's two's complement: 0..15 -> -8..7
                hi = ((w >> 4) ^ 8) - 8
                kv = (tl.reshape(tl.join(lo, hi), (KTT, LW)).to(tl.float32)
                      * tl.load(SC + tok, mask=ok, other=1.0)[:, None])
                kv = tl.where(ok[:, None], kv, 0.0).to(tl.bfloat16)
            else:
                kv = tl.load(LC + tok[:, None] * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            m, l, o = _tile(q, kv, m, l, o, ok, SCALE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _merge(PO, PM, PL, OUT, CNT, R, H: tl.constexpr, LW: tl.constexpr, NCH: tl.constexpr, SPARSE: tl.constexpr):
    """Program (row, head): the row's chunk partials in chunk order -> OUT[r, h] bf16. Sparse: rows with CNT 0 skip."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    if SPARSE:
        if tl.load(CNT + r) == 0:
            return
    k = tl.arange(0, LW)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((LW,), tl.float32)
    for c in range(NCH):
        base = (c * R + r) * H + h
        cm = tl.load(PM + base)
        cl = tl.load(PL + base)
        co = tl.load(PO + base * LW + k)
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a + co * b
        l = l * a + cl * b
        m = next_m
    tl.store(OUT + (r * H + h) * LW + k, (o / l).to(tl.bfloat16))


class LatentScratch:
    """Chunk partials for ``part_rows`` (one ``attention`` call) x heads x chunks; queries and latents for ``rows``."""

    def __init__(self, rows: int, heads: int, chunks: int, device, lw: int = L, part_rows: int | None = None) -> None:
        part_rows = rows if part_rows is None else min(rows, part_rows)
        self.rows, self.part_rows, self.heads, self.nch, self.lw = rows, part_rows, heads, chunks, lw
        self.po = torch.empty((chunks * part_rows * heads * lw,), dtype=torch.float32, device=device)
        self.pm = torch.empty((chunks * part_rows * heads,), dtype=torch.float32, device=device)
        self.pl = torch.empty((chunks * part_rows * heads,), dtype=torch.float32, device=device)
        self.qa = torch.empty((rows, heads, lw), dtype=torch.bfloat16, device=device)
        self.ol = torch.empty((rows, heads, lw), dtype=torch.bfloat16, device=device)
        self.dummy = torch.zeros((1,), dtype=torch.int32, device=device)


def attention(qa: torch.Tensor, cache: torch.Tensor, pos: torch.Tensor, s: LatentScratch, *, scale: float,
              nch: int, out: torch.Tensor, hb: int | None = None,
              scales: torch.Tensor | None = None) -> torch.Tensor:
    """Dense causal attention of qa [R, H, 512] through pos + R - 1 in nch 512-key chunks; a row ignores the others.
    ``scales``: fp32 [capacity] per-token scales for an fp8 or q4 cache (None: a bf16 cache)."""
    R, H, LW = qa.shape
    if nch > s.nch or R > s.part_rows or LW != s.lw:
        raise ValueError(f"latent attention: {R} rows, {nch} chunks, width {LW} past the scratch's "
                         f"{s.part_rows}, {s.nch}, {s.lw}")
    n = nch * R * H
    hb = head_block(R) if hb is None else hb
    if hb not in (HB, HB_WIDE):
        raise ValueError(f"latent attention: {hb} heads a program, not {HB} or {HB_WIDE}")
    fp8 = scales is not None and cache.dtype != torch.uint8
    _dense_chunks[(R, triton.cdiv(H, hb), nch)](qa, cache, scales if fp8 else cache, pos, s.po[:n * LW], s.pm[:n],
                                                s.pl[:n], R, H=H, LW=LW, CH=CHUNK, SCALE=scale, HBT=hb, KTT=KT,
                                                FP8=fp8, Q4=scales is not None, num_warps=8, num_stages=1)
    _merge[(R, H)](s.po, s.pm, s.pl, out, s.dummy, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
    return out


def sparse_attention(qa: torch.Tensor, cache: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor,
                     out: torch.Tensor, scale: float, scales: torch.Tensor | None = None) -> None:
    """Attention of rows with counts > 0 over their selected tokens (ascending, -1 padded), written into out; other rows are left alone.
    ``scales``: fp32 [capacity] per-token scales for an fp8 or q4 cache (None: a bf16 cache)."""
    R, H, LW = qa.shape
    W = tokens.shape[1]
    nch = triton.cdiv(W, CHUNK)
    n = nch * R * H
    po = torch.empty((n * LW,), dtype=torch.float32, device=qa.device)
    pm = torch.empty((n,), dtype=torch.float32, device=qa.device)
    pl = torch.empty((n,), dtype=torch.float32, device=qa.device)
    hb = head_block(R)
    fp8 = scales is not None and cache.dtype != torch.uint8
    _sparse_chunks[(R, triton.cdiv(H, hb), nch)](qa, cache, scales if fp8 else cache, tokens, counts, po, pm, pl, R,
                                                 W=W, H=H, LW=LW, CH=CHUNK, SCALE=scale, HBT=hb, KTT=KT,
                                                 FP8=fp8, Q4=scales is not None, num_warps=8, num_stages=1)
    _merge[(R, H)](po, pm, pl, out, counts, R, H=H, LW=LW, NCH=nch, SPARSE=True, num_warps=4)


def chunks_for(length: int) -> int:
    return triton.cdiv(length, CHUNK)
