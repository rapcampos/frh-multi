"""Score-space aggregators (Family 2) for multi-concept guided generation.

All aggregators operate on a per-concept score stack whose last dimension
indexes concepts, matching the scorer interface of
`FrameUnembeddingRepresentation._generate_guided`:

    scorer(scores: (..., n_concepts), weights: (n_concepts,)) -> (...)

Per-step, per-concept normalization over the candidate pool is provided by
`normalize_scores`. It is what makes score-space composition a genuinely
distinct method family: frame correlation is linear in the concept frame, so
an UNNORMALIZED weighted sum of per-concept scores equals the score against a
single weighted-average frame (Family 1 without the Procrustes step).
"""

from functools import partial
from typing import Callable

import torch

EPS = 1e-8


def normalize_scores(
    scores: torch.Tensor, method: str | None = "zscore", dim: int = 0
) -> torch.Tensor:
    """Normalize scores per concept across the candidate pool.

    Frames differ in rank and norm across concepts; without normalization,
    aggregation weights are meaningless (one concept's score scale dominates).

    Args:
        scores: Score tensor; `dim` must index the competing candidates.
        method: "zscore" (population z-score; constant scores map to 0),
            "rank" (candidate rank scaled to [0, 1]; invariant under any
            monotone transform of scores), or None (no-op).
        dim: Candidate-pool dimension to normalize over.

    Returns:
        torch.Tensor: Normalized scores, same shape.
    """
    if method is None:
        return scores
    if method == "zscore":
        mean = scores.mean(dim=dim, keepdim=True)
        std = scores.std(dim=dim, keepdim=True, unbiased=False)
        return (scores - mean) / (std + EPS)
    if method == "rank":
        ranks = scores.argsort(dim=dim).argsort(dim=dim).to(scores.dtype)
        return ranks / max(scores.size(dim) - 1, 1)
    raise ValueError(f"Unknown normalization method: {method!r}")


def weighted_sum(scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """F2.a — weighted sum. OR-like: one strong concept can compensate others.

    Signed weights give negative steering (w < 0 repels).
    """
    return (scores * weights).sum(-1)


def softmin(
    scores: torch.Tensor, weights: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    """F2.b — soft minimum: -tau * logsumexp(-w*s / tau). AND semantics.

    A candidate is only as good as its worst (weighted) concept score.
    tau -> 0 recovers the hard minimum (brittle under greedy decoding);
    larger tau blends toward an average.
    """
    return -tau * torch.logsumexp(-(scores * weights) / tau, dim=-1)


def constrained(
    scores: torch.Tensor,
    weights: torch.Tensor,  # noqa: ARG001 - kept for the scorer interface
    thresholds: list[float] | torch.Tensor,
    penalty: float = float("-inf"),
) -> torch.Tensor:
    """F2.c — lexicographic: maximize concept 0 subject to the rest >= thresholds.

    `weights` is unused (interface compatibility). `thresholds` has one entry
    per secondary concept (scores[..., 1:]).

    Note on `penalty`: inside the decoding loop, scores are cumulatively
    summed over token positions, so a -inf would poison every later position.
    Use a finite penalty there (see `constrained_scorer`).
    """
    thresholds = torch.as_tensor(thresholds, device=scores.device, dtype=scores.dtype)
    primary = scores[..., 0]
    satisfied = (scores[..., 1:] >= thresholds).all(-1)
    return torch.where(satisfied, primary, torch.full_like(primary, penalty))


def softmin_scorer(tau: float = 1.0) -> Callable:
    """Scorer factory for `softmin` with a fixed temperature."""
    return partial(softmin, tau=tau)


def constrained_scorer(
    thresholds: list[float] | torch.Tensor, penalty: float = -1e4
) -> Callable:
    """Scorer factory for `constrained`; finite penalty suits the decoding loop."""
    return partial(constrained, thresholds=thresholds, penalty=penalty)
