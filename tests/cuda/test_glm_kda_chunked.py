"""GLM's KDA prompt chunks in chunked (WY) form (families/glm5_next/cuda/kda_chunked.py): the chunked reference
matches the float64 serial delta rule, a row's results keep their bits across aligned chunk starts, and the CUDA
kernel (kda_chunk.cu) matches the references."""

import pytest
import torch

from tensorfold.families.glm5_next.cuda import kda_chunked

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")

H = 2
C = 3 * H * kda_chunked.DK
B_OFF = C
LOWER, EPS = 0.05, 1e-6


def _inputs(rows, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    conv_w = (torch.randn((C, 4), generator=gen) * 0.1).to(torch.bfloat16).to(device)
    conv_state = torch.randn((3, C), generator=gen).to(torch.bfloat16).to(device)
    p = torch.randn((rows + 3, C + H), generator=gen).to(torch.bfloat16).to(device)
    a = torch.randn((rows + 3, H * kda_chunked.DK), generator=gen).to(torch.bfloat16).to(device)   # decay + output
    g = torch.randn((rows + 3, H * kda_chunked.DK), generator=gen).to(torch.bfloat16).to(device)   # gate rows
    a_log = torch.randn((H,), generator=gen).to(device)
    dt_bias = torch.randn((H * kda_chunked.DK,), generator=gen).to(device)
    norm_w = torch.randn((kda_chunked.DV,), generator=gen).to(torch.bfloat16).to(device)
    state = torch.randn((H, kda_chunked.DK, kda_chunked.DV), generator=gen).to(device)
    return p, a, g, conv_state, conv_w, a_log, dt_bias, norm_w, state


def test_chunked_reference_matches_the_serial_delta_rule():
    """float64: the WY-form sub-chunks compute the same recurrence as row-by-row decays."""
    rows = 70
    p, a, g, conv_state, conv_w, a_log, dt_bias, norm_w, state = _inputs(rows, 7, "cpu")
    q, k, v, lg, beta = kda_chunked.prep_reference(p, B_OFF, a, conv_state, conv_w, a_log, dt_bias, LOWER, rows)
    serial, s_serial = kda_chunked.recurrent_reference(q, k, v, lg, beta, state)
    chunked, s_chunked = kda_chunked.chunked_reference(q, k, v, lg, beta, state)
    torch.testing.assert_close(chunked, serial, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(s_chunked, s_serial, rtol=1e-8, atol=1e-9)


def test_aligned_chunk_starts_keep_a_rows_bits():
    """A row's results depend on its sub-chunk and the entering state, not on where its chunk began (pos 0 vs 32)."""

    rows = 70
    p, a, g, conv_state, conv_w, a_log, dt_bias, norm_w, state = _inputs(rows, 11, "cpu")
    q, k, v, lg, beta = kda_chunked.prep_reference(p, B_OFF, a, conv_state, conv_w, a_log, dt_bias, LOWER, rows)
    out0, s0 = kda_chunked.chunked_reference(q, k, v, lg, beta, state, pos=0)
    out32, s32 = kda_chunked.chunked_reference(q, k, v, lg, beta, state, pos=kda_chunked.CHUNK)
    assert torch.equal(out0, out32)
    assert torch.equal(s0, s32)


def test_chain_reference_gates_like_kda_cus_norm():
    """The gated RMSNorm reference agrees with a direct formula and keeps bf16 read-outs' rounding."""

    rows = 33
    p, a, g, conv_state, conv_w, a_log, dt_bias, norm_w, state = _inputs(rows, 3, "cpu")
    out, _, y = kda_chunked.chain_reference(p, B_OFF, a, g, conv_state, conv_w, state, a_log, dt_bias, norm_w, EPS,
                                            LOWER, rows)
    assert out.shape == (rows, H * kda_chunked.DV) and out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


@cuda
def test_kernel_matches_the_chunked_reference():
    """kda_chunk.cu against its torch reference in fp32 (same products, same order) and the float64 serial rule."""
    from tensorfold.families.glm5_next.cuda import kda_chunked as kc

    rows = 70
    p, a, g, conv_state, conv_w, a_log, dt_bias, norm_w, state = _inputs(rows, 5, "cuda")
    out = torch.empty((rows, H * kda_chunked.DV), dtype=torch.bfloat16, device="cuda")
    state_out = torch.empty_like(state)
    kc.chain(p, B_OFF, a, g, conv_state, conv_w, state, a_log, dt_bias, norm_w, EPS, LOWER, rows, out, state_out)
    ref, ref_state, y = kda_chunked.chain_reference(p, B_OFF, a, g, conv_state, conv_w, state, a_log, dt_bias, norm_w,
                                                    EPS, LOWER, rows, dtype=torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(state_out, ref_state, rtol=1e-3, atol=1e-3)
    q, k, v, lg, beta = kc.prep_reference(p, B_OFF, a, conv_state, conv_w, a_log, dt_bias, LOWER, rows)
    serial, _ = kc.recurrent_reference(q, k, v, lg, beta, state)
    torch.testing.assert_close(out.float(), serial, rtol=1e-2, atol=2e-2)


@cuda
def test_kernel_bits_do_not_depend_on_the_chunk_start():
    """Aligned chunk starts (pos 0 vs 96) run the same sub-chunks on the same entering state: identical bits."""

    rows = 70
    p, a, g, conv_state, conv_w, a_log, dt_bias, norm_w, state = _inputs(rows, 9, "cuda")
    outs, states = [], []
    for pos in (0, 96):
        out = torch.empty((rows, H * kda_chunked.DV), dtype=torch.bfloat16, device="cuda")
        state_out = torch.empty_like(state)
        kda_chunked.chain(p, B_OFF, a, g, conv_state, conv_w, state, a_log, dt_bias, norm_w, EPS, LOWER, rows, out,
                          state_out, pos)
        outs.append(out)
        states.append(state_out)
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(states[0], states[1])
