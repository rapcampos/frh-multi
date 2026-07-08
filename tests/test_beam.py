"""CPU unit tests for the beam-search selection logic (_topk_beam_selection)."""

import torch

from frames.representations import FrameUnembeddingRepresentation

select = FrameUnembeddingRepresentation._topk_beam_selection


def make_pool(n: int, m: int, k: int, t: int) -> tuple[torch.Tensor, torch.Tensor]:
    cum = torch.zeros(n, m, t)
    candidate_tokens = torch.arange(n * m * k * t).reshape(n, m, k, t)
    return cum, candidate_tokens


def test_selects_best_candidates_across_parents():
    # pool of 4 = 2 parents x 2 children; the two best come from different parents
    cum, cand = make_pool(1, 4, 2, 3)
    cum[0, :, -1] = torch.tensor([0.9, 0.1, 0.8, 0.2])

    new_tokens, best_probe = select(cum, cand, 2)

    assert new_tokens.shape == (1, 2, 2, 3)
    assert torch.equal(new_tokens[0, 0], cand[0, 0])  # best beam first
    assert torch.equal(new_tokens[0, 1], cand[0, 2])  # runner-up from other parent
    assert torch.equal(best_probe[0], cum[0, 0])  # full trajectory of the best


def test_selection_is_independent_per_input():
    cum, cand = make_pool(2, 3, 2, 4)
    cum[0, :, -1] = torch.tensor([0.1, 0.9, 0.5])
    cum[1, :, -1] = torch.tensor([0.7, 0.2, 0.3])

    new_tokens, best_probe = select(cum, cand, 2)

    assert torch.equal(new_tokens[0, 0], cand[0, 1])
    assert torch.equal(new_tokens[0, 1], cand[0, 2])
    assert torch.equal(new_tokens[1, 0], cand[1, 0])
    assert torch.equal(new_tokens[1, 1], cand[1, 2])
    assert torch.equal(best_probe[0], cum[0, 1])
    assert torch.equal(best_probe[1], cum[1, 0])


def test_selection_uses_final_cumulative_score_not_peak():
    # candidate 0 peaks mid-sequence but ends low; candidate 1 ends highest
    cum, cand = make_pool(1, 2, 2, 3)
    cum[0, 0] = torch.tensor([0.0, 5.0, 0.1])
    cum[0, 1] = torch.tensor([0.0, 0.2, 0.9])

    new_tokens, best_probe = select(cum, cand, 1)

    assert torch.equal(new_tokens[0, 0], cand[0, 1])
    assert torch.equal(best_probe[0], cum[0, 1])


def test_k_larger_than_pool_keeps_whole_pool():
    cum, cand = make_pool(1, 2, 3, 2)
    cum[0, :, -1] = torch.tensor([0.2, 0.5])

    new_tokens, _ = select(cum, cand, 3)

    assert new_tokens.shape == (1, 2, 3, 2)
    assert torch.equal(new_tokens[0, 0], cand[0, 1])


def test_k1_degenerates_to_argmax():
    cum, cand = make_pool(1, 3, 1, 2)
    cum[0, :, -1] = torch.tensor([0.3, 0.1, 0.8])

    new_tokens, best_probe = select(cum, cand, 1)

    assert new_tokens.shape == (1, 1, 1, 2)
    assert torch.equal(new_tokens[0, 0], cand[0, 2])
    assert torch.equal(best_probe[0], cum[0, 2])
