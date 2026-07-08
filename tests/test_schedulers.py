"""CPU unit tests for Family 3 concept schedulers."""

import torch

from frames.representations.schedulers import RoundRobin, SentenceBoundary, Stochastic


def channel_projections(
    n_rows: int = 4, t: int = 5, n_concepts: int = 3
) -> torch.Tensor:
    """Projections where concept c's channel is constant c+1 — makes the
    selected concept identifiable from the output values."""
    base = torch.arange(1, n_concepts + 1, dtype=torch.float)
    return base.expand(n_rows, t, 1, n_concepts).clone()


WEIGHTS = torch.ones(3)


class TestRoundRobin:
    def test_cycles_deterministically(self):
        scheduler = RoundRobin()
        scheduler.reset(n=2)
        proj = channel_projections()
        seen = [scheduler(proj, WEIGHTS)[0, 0, 0].item() for _ in range(7)]
        assert seen == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]

    def test_reset_restarts_cycle(self):
        scheduler = RoundRobin()
        proj = channel_projections()
        scheduler(proj, WEIGHTS)
        scheduler(proj, WEIGHTS)
        scheduler.reset(n=2)
        assert scheduler(proj, WEIGHTS)[0, 0, 0].item() == 1.0

    def test_output_is_exactly_the_active_channel(self):
        scheduler = RoundRobin()
        scheduler.reset(n=1)
        proj = torch.randn(2, 4, 1, 3)
        out = scheduler(proj, WEIGHTS)
        assert torch.equal(out, proj[..., 0])


class TestStochastic:
    def test_same_seed_reproduces_schedule(self):
        proj = channel_projections()
        runs = []
        for _ in range(2):
            scheduler = Stochastic([0.5, 0.3, 0.2], seed=42)
            scheduler.reset(n=2)
            runs.append([scheduler(proj, WEIGHTS)[0, 0, 0].item() for _ in range(20)])
        assert runs[0] == runs[1]

    def test_reset_reproduces_same_schedule(self):
        proj = channel_projections()
        scheduler = Stochastic([0.5, 0.3, 0.2], seed=7)
        scheduler.reset(n=2)
        first = [scheduler(proj, WEIGHTS)[0, 0, 0].item() for _ in range(10)]
        scheduler.reset(n=2)
        second = [scheduler(proj, WEIGHTS)[0, 0, 0].item() for _ in range(10)]
        assert first == second

    def test_distribution_roughly_matches_probabilities(self):
        proj = channel_projections(n_concepts=2)
        scheduler = Stochastic([0.8, 0.2], seed=0)
        scheduler.reset(n=1)
        draws = [
            scheduler(proj[..., :2], torch.ones(2))[0, 0, 0].item() for _ in range(500)
        ]
        share_first = draws.count(1.0) / len(draws)
        assert 0.7 < share_first < 0.9


class FakeTokenizer:
    """Maps token id -> string; enough for boundary detection."""

    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab

    def batch_decode(self, token_ids: torch.Tensor) -> list[str]:
        return [self.vocab.get(int(ids[0]), "x") for ids in token_ids]


class TestSentenceBoundary:
    def make(self, n: int = 2) -> SentenceBoundary:
        tokenizer = FakeTokenizer({0: " word", 1: ".", 2: "!", 3: " and"})
        scheduler = SentenceBoundary(tokenizer)
        scheduler.reset(n=n)
        return scheduler

    def tokens(self, newest_per_input: list[int], m: int = 2) -> torch.Tensor:
        """Token tensor (n*m, T) whose representative rows end in the given ids."""
        n = len(newest_per_input)
        t = torch.zeros(n, m, 3, dtype=torch.long)
        for i, tok in enumerate(newest_per_input):
            t[i, 0, -1] = tok
        return t.flatten(0, 1)

    def test_no_boundary_keeps_first_concept(self):
        scheduler = self.make()
        proj = channel_projections()
        scheduler.observe(self.tokens([0, 0]), n=2)
        out = scheduler(proj, WEIGHTS)
        assert out[0, 0, 0].item() == 1.0

    def test_boundary_advances_concept(self):
        scheduler = self.make()
        proj = channel_projections()
        scheduler.observe(self.tokens([1, 0]), n=2)  # input 0 ends a sentence
        out = scheduler(proj, WEIGHTS)
        # rows 0..1 belong to input 0 (advanced), rows 2..3 to input 1 (not)
        assert out[0, 0, 0].item() == 2.0
        assert out[1, 0, 0].item() == 2.0
        assert out[2, 0, 0].item() == 1.0

    def test_wraps_around_modulo_concepts(self):
        scheduler = self.make(n=1)
        proj = channel_projections(n_rows=2)
        for _ in range(3):  # three boundaries, three concepts -> back to first
            scheduler.observe(self.tokens([2], m=2), n=1)
        out = scheduler(proj, WEIGHTS)
        assert out[0, 0, 0].item() == 1.0

    def test_reset_clears_per_input_state(self):
        scheduler = self.make()
        scheduler.observe(self.tokens([1, 1]), n=2)
        scheduler.reset(n=2)
        proj = channel_projections()
        assert scheduler(proj, WEIGHTS)[0, 0, 0].item() == 1.0
