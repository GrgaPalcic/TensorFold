"""GLM's DFlash2 tree drafting: the proposal flattens to rows (the greedy chain first), the round commits the
longest accepted root path, and replies equal serial. Runs without a GPU: the fake triton of the ``allocations``
fixture lets the CUDA modules import, and the round runs on a fake engine with a scripted sampler."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch

EOS = frozenset({77})
PENDING = 5


def decode_mod() -> object:
    return importlib.import_module("tensorfold.families.glm5_next.cuda.decode")


def candidates(seed=7, vocab=50, k=4, depth=4, rank=4):
    """Random candidate rows (tokens, logits, selector projections) with random rank-r codebooks."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, vocab, size=(depth, k)).astype(np.int64), rng.normal(size=(depth, k)),
            rng.normal(size=depth), rng.normal(size=(vocab, rank)), rng.normal(size=(vocab, rank)))


class FakeEngine:
    """A serial engine whose sampler reads a per-forward table of position -> token."""

    def __init__(self, samples: list[dict[int, int]]):
        self.w = SimpleNamespace(cfg=SimpleNamespace(eos=EOS))
        self.st = SimpleNamespace(pos=100)
        self.rows = 8
        self.buf = SimpleNamespace()
        self.constraint = None
        self.samples = samples
        self.windows: list[list[int]] = []
        self.followed: list[list[int]] = []
        self._call = 0

    def verify_window(self, tokens: list[int]) -> list[int]:
        return list(tokens)

    def forward(self, tokens: list[int]) -> torch.Tensor:
        self.windows.append(list(tokens))
        self._call += 1
        return torch.zeros((len(tokens), 4))

    def sample(self, logits, positions, sampling, **_):
        table = self.samples[self._call - 1]
        return [table[p] for p in positions]

    def follow(self, tokens: list[int]) -> None:
        self.followed.append(list(tokens))

    def tap_rows(self, keep: int):
        return ("taps", keep)


class FakeDrafter:
    def __init__(self, proposals: list[tuple[list[int], list[int]]]):
        self.proposals = list(proposals)
        self.calls: list[tuple] = []
        self.taps: list = []

    def propose_tree(self, pending, depth, fanout, sampling, confidence=0.0, budget=7):
        self.calls.append((pending, depth, fanout, confidence))
        return self.proposals.pop(0)

    def add_taps(self, rows) -> None:
        self.taps.append(rows)


@pytest.fixture
def rounds(monkeypatch, allocations):  # noqa: F811
    """The fake triton import, decode.commit replaced with a recorder that advances st.pos, a silent synchronize."""
    dec = decode_mod()
    commits: list[tuple[int, int]] = []

    def commit(w, st, b, R, keep):
        commits.append((R, keep))
        st.pos += keep

    monkeypatch.setattr(dec, "commit", commit)
    monkeypatch.setattr(dec.torch.cuda, "synchronize", lambda: None)
    return dec, commits


def test_tree_policy_specs_round_trip(allocations):  # noqa: F811
    eng = importlib.import_module("tensorfold.families.glm5_next.cuda.engine")
    assert eng.encode_policy("t2d3") == [6, 3, 2, 0]
    assert eng.encode_policy("ft2d3") == [16, 3, 2, 0]
    assert (eng.decode_policy([6, 3, 2, 0]).fanout, eng.decode_policy([6, 3, 2, 0]).depth) == (2, 3)
    assert (eng.decode_policy([16, 3, 2, 0]).fanout, eng.decode_policy([16, 3, 2, 0]).depth) == (2, 3)
    for bad in ("t2d4", "t0d3", "t5d1", "t2", "t2d", "t2d0"):
        with pytest.raises(ValueError):
            eng.encode_policy(bad)


def test_fanout_one_tree_is_the_greedy_chain(allocations):  # noqa: F811
    """One chain of the tree is byte-identical to chain(): the same edge scores, noise and confidence rule."""
    from tensorfold.engine.exact_sampling import Sampling

    d2 = importlib.import_module("tensorfold.families.glm5_next.cuda.dflash2")
    tokens, values, proj, succ, pred = candidates()
    fake = SimpleNamespace(succ=succ, pred=pred)
    for sampling in (None, Sampling(seed=11, temperature=0.7)):
        chain = d2.Drafter.chain(fake, tokens, values, proj, PENDING, 100, sampling, 0.0)
        out, parents = d2.tree_chains(tokens, values, proj, succ, pred, PENDING, 100, sampling,
                                      fanout=1, depth=4, budget=7)
        assert out == chain
        assert parents == list(range(-1, len(out) - 1))


def test_tree_flattens_chain_first_with_connected_parents(allocations):  # noqa: F811
    """Two chains a depth 3: the greedy chain is the flat prefix, every later node's parent an earlier row."""
    d2 = importlib.import_module("tensorfold.families.glm5_next.cuda.dflash2")
    tokens, values, proj, succ, pred = candidates()
    out, parents = d2.tree_chains(tokens, values, proj, succ, pred, PENDING, 100, None, fanout=2, depth=3, budget=7)
    assert len(out) == 6
    chain = d2.Drafter.chain(SimpleNamespace(succ=succ, pred=pred), tokens, values, proj, PENDING, 100, None, 0.0)
    assert out[:3] == chain[:3] and parents[:3] == [-1, 0, 1]
    assert all(0 <= p < i for i, p in enumerate(parents) if p >= 0)
    assert parents.count(-1) == 2                       # the chain's seed and the second chain's seed


def test_budget_and_confidence_bound_the_tree(allocations):  # noqa: F811
    d2 = importlib.import_module("tensorfold.families.glm5_next.cuda.dflash2")
    tokens, values, proj, succ, pred = candidates()
    out, parents = d2.tree_chains(tokens, values, proj, succ, pred, PENDING, 100, None, fanout=2, depth=3, budget=4)
    assert len(out) == 4 and len(parents) == 4
    # a floor no depth-1 branch meets: only the anchor's seeds survive
    out, parents = d2.tree_chains(tokens, values, proj, succ, pred, PENDING, 100, None,
                                  fanout=2, depth=3, confidence=0.99, budget=7)
    assert len(out) == 2 and parents == [-1, -1]


def test_accepted_path_prefers_the_chain_and_stops_at_eos(rounds):  # noqa: F811
    dec, _ = rounds
    drafts, parents = [10, 11, 12, 13], [-1, 0, 0, 2]
    # a later branch tying the chain's depth loses to the earlier rows
    assert dec.accepted_path(drafts, parents, [10, 11, 12, 99], EOS, True) == [0, 1]
    # an accepted node sampled EOS ends its path: children do not extend it (unless stop_eos is off)
    drafts, parents = [10, 77, 12, 13], [-1, 0, 1, 2]
    assert dec.accepted_path(drafts, parents, [10, 77, 12, 13], EOS, True) == [0, 1]
    assert dec.accepted_path(drafts, parents, [10, 77, 12, 13], EOS, False) == [0, 1, 2, 3]


def test_round_commits_the_longest_accepted_root_path(rounds):  # noqa: F811
    """The chain breaks at depth 2 while a beam branch verifies whole: the round re-verifies it in its own
    window and commits 4 tokens, not the chain's 2."""
    dec, commits = rounds
    e = FakeEngine([{101: 41, 102: 42, 103: 91, 104: 42, 105: 43, 106: 44},
                    {101: 41, 102: 42, 103: 43, 104: 44}])
    drafter = FakeDrafter([([41, 91, 92, 42, 43], [-1, 0, 1, 0, 3])])
    res = dec.tree_decode(e, drafter, PENDING, 5, None, policy=dec.TreePolicy(2, 3))
    assert res.tokens == [PENDING, 41, 42, 43, 44]
    assert (res.rounds, res.drafted, res.accepted) == (1, 5, 3)
    assert commits == [(4, 4)]                          # one commit, of the clean re-verified path
    assert e.windows == [[PENDING, 41, 91, 92, 42, 43], [PENDING, 41, 42, 43]]
    assert e.st.pos == 104 and e.followed == [[41, 42, 43, 44]]
    assert drafter.taps == [("taps", 4)]                # the drafter advances from the kept rows only


def test_round_falls_back_to_the_chain_when_the_path_does_not_survive_cleanly(rounds):  # noqa: F811
    """A path the contaminated window accepted but the clean window rejects: the round re-verifies the chain
    prefix and commits it, still 3 tokens."""
    dec, commits = rounds
    e = FakeEngine([{101: 41, 102: 42, 103: 7, 104: 44, 105: 45, 106: 46},
                    {101: 41, 102: 42, 103: 8, 104: 9},
                    {101: 41, 102: 42, 103: 43}])
    drafter = FakeDrafter([([41, 42, 91, 44, 45], [-1, 0, 1, 0, 3])])
    res = dec.tree_decode(e, drafter, PENDING, 4, None, policy=dec.TreePolicy(2, 3))
    assert res.tokens == [PENDING, 41, 42, 43]
    assert (res.rounds, res.drafted, res.accepted) == (1, 5, 2)
    assert commits == [(3, 3)]                          # the fallback's clean chain prefix, not the path's window
    assert e.windows == [[PENDING, 41, 42, 91, 44, 45], [PENDING, 41, 44, 45], [PENDING, 41, 42]]
    assert e.st.pos == 103 and drafter.taps == [("taps", 3)]


def test_round_keeps_the_chain_prefix_when_it_is_the_best_path(rounds):  # noqa: F811
    """The chain is the longest accepted path: one forward, one commit, no re-verification."""
    dec, commits = rounds
    e = FakeEngine([{101: 41, 102: 42, 103: 7, 104: 8}])
    drafter = FakeDrafter([([41, 42, 91], [-1, 0, 1])])
    res = dec.tree_decode(e, drafter, PENDING, 4, None, policy=dec.TreePolicy(2, 3))
    assert res.tokens == [PENDING, 41, 42, 7]
    assert (res.rounds, res.drafted, res.accepted) == (1, 3, 2)
    assert commits == [(4, 3)] and len(e.windows) == 1
    assert e.st.pos == 103 and drafter.taps == [("taps", 3)]


def test_auto_tree_round_feeds_both_backlogs_from_the_kept_tokens(rounds):  # noqa: F811
    """The auto loop's tree branch: the kept tokens join the MTP backlog and DFlash2's taps, the round is
    recorded on the f arm."""
    choice = importlib.import_module("tensorfold.families.glm5_next.cuda.drafter_choice")
    dec, commits = rounds
    e = FakeEngine([{101: 41, 102: 42, 103: 91, 104: 44}])
    e.w.cfg.hidden = 8
    e.w.device = "cpu"
    e.buf = SimpleNamespace(taps=(1, 2, 3))
    e.last_hidden = torch.zeros((1, 8), dtype=torch.bfloat16)
    e.main_hidden = lambda sl: torch.zeros((sl.stop, 8), dtype=torch.bfloat16)
    e.tap_rows = lambda keep: torch.zeros((keep, 24), dtype=torch.bfloat16)
    drafter = FakeDrafter([([41, 42, 91], [-1, 0, 1])])
    costs = {"verify": [28.4, 38.8, 43.8, 48.8, 53.8, 58.8, 63.8, 68.8], "mtp": 1.79, "mtp_step": 1.56,
             "mtp_row": 0.17, "block": 3.16, "taps_row": 0.049}
    res = choice.auto_decode(e, drafter, PENDING, 5, None, choice=choice.DrafterChoice(costs, first="f"),
                             m_policy=dec.DepthPolicy(3, fixed=True), f_policy=dec.TreePolicy(2, 3))
    assert res.tokens == [PENDING, 41, 42, 91, 44]
    assert res.arms == "f" and res.keeps == [4] and res.depths == [3]
    assert (res.rounds, res.drafted, res.accepted) == (1, 3, 3)
    assert commits == [(4, 4)] and e.st.pos == 104
    assert len(drafter.taps) == 1 and tuple(drafter.taps[0].shape) == (4, 24)   # fed once, by tree_round only


def test_acceptance_math_matches_the_docstring(allocations):  # noqa: F811
    """E[max of k geometric(p)] = sum_j (-1)^(j+1) C(k, j) p^j / (1 - p^j): a round of 2 chains commits about
    4.27 tokens where one chain commits 2.70, at p = 0.73 (the measured chain acceptance)."""
    from math import comb

    def emax(k, p):
        return sum((-1) ** (j + 1) * comb(k, j) * p**j / (1 - p**j) for j in range(1, k + 1))

    assert round(emax(2, 0.73), 2) == 4.27 and round(emax(1, 0.73), 2) == 2.70
