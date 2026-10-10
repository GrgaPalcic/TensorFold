"""TF_GLM_HC_SPLIT (families/glm5_next/cuda/hcsplit.py): the settings parse and partition rows, and the split's
building blocks keep the unsplit path's bits — hc_post_pair sums two held-apart partials exactly like hc_post sums
the gathered stack, and the per-row glue kernels give a row the same bits whichever rows run with it."""

import os
import sys
import types

import pytest
import torch

try:
    import triton  # noqa: F401  (the glue kernels run on a GPU; CPU collection only needs the module to import)
except ImportError:
    _tl = types.SimpleNamespace(constexpr=object)
    sys.modules["triton"] = types.SimpleNamespace(jit=lambda f=None, **k: f if f is not None else (lambda g: g),
                                                  language=_tl)
    sys.modules["triton.language"] = _tl

from tensorfold.families.glm5_next.cuda import glue, hcsplit
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


def test_settings_parse_and_code():
    s = hcsplit.SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1", "TF_GLM_PREFILL_OVERLAP": "1",
                                        "TF_GLM_OVERLAP_PIECES": "3", "TF_GLM_HC_SPLIT_MIN_ROWS": "128",
                                        "TF_GLM_HC_EXCHANGE": "gather"})
    assert (s.split, s.overlap, s.pieces, s.min_rows, s.exchange) == (True, True, 3, 128, "gather")
    assert s.piece_count() == 1                      # overlap pieces are p2p only
    assert s.code() == [1, 1, 1, 128, 1]
    two = hcsplit.SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1", "TF_GLM_PREFILL_OVERLAP": "2"})
    assert two.fronts and two.code() == [1, 2, 2, 64, 0]      # the default 2 pieces, p2p
    with pytest.raises(ValueError):
        hcsplit.SplitSettings.from_env({"TF_GLM_PREFILL_OVERLAP": "2"})          # still needs the split
    with pytest.raises(ValueError):
        hcsplit.SplitSettings.from_env({"TF_GLM_PREFILL_OVERLAP": "3"})
    with pytest.raises(ValueError):
        hcsplit.SplitSettings(split=True, fronts=True)                           # fronts need the overlap

    off = hcsplit.SplitSettings.from_env({})
    assert off.code() == [0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        hcsplit.SplitSettings.from_env({"TF_GLM_PREFILL_OVERLAP": "1"})          # needs the split
    with pytest.raises(ValueError):
        hcsplit.SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1", "TF_GLM_HC_EXCHANGE": "ring"})


def _fake_split(w, b, settings):
    return hcsplit.HcSplit(w, b, settings)


def test_begin_partitions_rows():
    """Pieces cover the rank's half; an odd chunk pads to Rp = R + 1 and zeroes the pad row."""

    class FakeComm:
        def all_gather(self, send, recv):
            pass

    class W:
        world, rank, comm, cfg = 2, 0, FakeComm(), type("C", (), {"hc_eps": 1e-6, "hc_iters": 1})()

    rows, d = 131, 8
    b = type("B", (), {"rows": rows, "part": torch.zeros((rows + 1, d)), "gath": torch.zeros((2 * (rows + 1) * d,)),
                       "x": torch.ones((rows + 1, 4 * d)) * 7.0})()
    sp = _fake_split(W(), b, hcsplit.SplitSettings(split=True, exchange="gather"))
    sp.begin(rows)
    assert sp.Rp == 132 and sp.H == 66
    assert [off for off, n in sp.pieces] == list(range(0, 66, 66)) or sum(n for _, n in sp.pieces) == 66
    assert b.x[131].abs().sum() == 0 and b.part[131].abs().sum() == 0
    sp.finish()


@cuda
def test_hc_post_pair_sums_like_hc_post():
    """Two held-apart partials (any element offset): the same bits as hc_post over the gathered [2, R, D] stack."""

    gen = torch.Generator().manual_seed(4)
    rows, d = 97, 256
    x = torch.randn((rows, 4 * d), generator=gen).to(torch.bfloat16).cuda()
    post = torch.randn((rows, 4), generator=gen).float().cuda()
    comb = torch.randn((rows, 16), generator=gen).float().cuda()
    g0 = torch.randn((rows, d), generator=gen).float().cuda()
    g1 = torch.randn((rows, d), generator=gen).float().cuda()
    want = x.clone()
    glue.hc_post(x, want, torch.stack((g0, g1)), post, comb)
    got = x.clone()
    glue.hc_post_pair(got, got, g0, g1, post, comb)
    assert torch.equal(want, got)
    # rank 1's view: g0 and g1 swapped must not change the sum's order (rank 0's partial first)
    rev = x.clone()
    glue.hc_post_pair(rev, rev, g1, g0, post, comb)
    assert torch.equal(rev, want)


@cuda
def test_glue_pieces_match_the_whole_rows():
    fn = torch.randn((24, 4 * d), generator=gen).to(torch.bfloat16).cuda()          # [24, S*D]
    norm_w = torch.randn((d,), generator=gen).to(torch.bfloat16).cuda()
    base = torch.randn((24,), generator=gen).float().cuda()
    scale = torch.randn((3,), generator=gen).float().cuda()

    out_full = torch.empty((rows, d), dtype=torch.bfloat16, device="cuda")
    xs_full = torch.empty((rows, d // 64), dtype=torch.float32, device="cuda")
    post = torch.empty((rows, 4), dtype=torch.float32, device="cuda")
    comb = torch.empty((rows, 16), dtype=torch.float32, device="cuda")
    part = torch.empty((rows, glue.HC_BLOCKS, 32), dtype=torch.float32, device="cuda")
    glue.hc_pre(x, fn, base, scale, norm_w, out_full, xs_full, post, comb, part, 1e-6, 1e-6, 1)

    out_p = torch.empty_like(out_full)
    xs_p = torch.empty_like(xs_full)
    post_p = torch.empty_like(post)
    comb_p = torch.empty_like(comb)
    part_p = torch.empty_like(part)
    for lo, hi in ((0, 33), (33, 70)):
        glue.hc_pre(x[lo:hi], fn, base, scale, norm_w, out_p[lo:hi], xs_p[lo:hi], post_p[lo:hi], comb_p[lo:hi],
                    part_p[lo:hi], 1e-6, 1e-6, 1)
    assert torch.equal(out_full, out_p)
    assert torch.equal(xs_full, xs_p)
    assert torch.equal(post, post_p)
    assert torch.equal(comb, comb_p)

    hidden_full = glue.stream_mean(x, torch.empty((rows, d), dtype=torch.bfloat16, device="cuda"))
    hidden_p = torch.empty_like(hidden_full)
    for lo, hi in ((0, 33), (33, 70)):
        glue.stream_mean(x[lo:hi], hidden_p[lo:hi])
    assert torch.equal(hidden_full, hidden_p)
