"""Torch-only helpers for subspace-guided top-k generation.

These mirror the per-step scoring and candidate selection of the original FRH
top-k guided method (``FrameUnembeddingRepresentation._generate_with_topk_guide``):
greedy generation with k-way lookahead — at each step the single best
continuation is kept and its ``k`` candidate children are carried forward. This
is *not* beam search. The only change here is the per-position score, which is a
subspace projection magnitude instead of a frame correlation. Kept model-free so
the selection logic is unit testable on CPU.
"""

from __future__ import annotations

import torch


def subspace_alignment(
    last_hs: torch.Tensor,
    basis: torch.Tensor,
    repel_basis: torch.Tensor | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Signed per-position projection magnitude onto one or two subspaces.

    Args:
        last_hs: hidden states, shape ``(..., d)``.
        basis: attractive orthonormal subspace basis, shape ``(d, r)``.
        repel_basis: optional repelled subspace basis, shape ``(d, r_repel)``.
            Its projection magnitude is subtracted from the attractive score.
        normalize: divide each projection magnitude by ``‖h‖``, scoring the
            *fraction* of hidden-state energy inside the subspace rather than its
            absolute size. Bounded in ``[0, 1]`` (``[-1, 1]`` with a repelled
            basis), so the alignment term sits on the same scale as the frame
            pipeline's correlation and matches the cosine metric the comparison
            battery reports. The default ``False`` keeps the original
            scale-dependent score, which existing results depend on.

    Returns:
        ``(...)`` per-position score ``‖B_attractᵀh‖`` or, when
        ``repel_basis`` is supplied,
        ``‖B_attractᵀh‖ - ‖B_repelᵀh‖`` — each term divided by ``‖h‖`` when
        ``normalize`` is set.
    """
    hs = last_hs.float()
    # Padding positions can be exactly zero; clamp keeps the score finite.
    scale = hs.norm(dim=-1).clamp_min(torch.finfo(hs.dtype).tiny) if normalize else None

    attractive = (hs @ basis.to(last_hs.device).float()).norm(dim=-1)
    if normalize:
        attractive = attractive / scale
    if repel_basis is None:
        return attractive

    repelled = (hs @ repel_basis.to(last_hs.device).float()).norm(dim=-1)
    if normalize:
        repelled = repelled / scale
    return attractive - repelled


def select_best_candidate(
    scores: torch.Tensor, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick, per input, the candidate with the highest cumulative alignment.

    Mirrors the ``scores.cumsum(-2).reshape(n, k, -1).max(-2)`` selection of the
    original top-k guided method — greedy with k-way lookahead, not beam search.

    Args:
        scores: per-position alignment for each current candidate, shape
            ``(n*k, T)``. Rows are ordered as ``n`` blocks of ``k`` candidates.
        n: number of inputs.
        k: number of lookahead candidates per input.

    Returns:
        ``(col_idx, probe)`` where ``col_idx`` is ``(n,)`` the winning candidate
        index per input, and ``probe`` is ``(n, T)`` the cumulative-alignment
        trajectory of the winning candidate.
    """
    cum = scores.unsqueeze(-1).cumsum(-2).reshape(n, k, -1)  # (n, k, T)
    best = cum.max(-2)
    col_idx = best.indices[..., -1]  # (n,)
    probe = best.values  # (n, T)
    return col_idx, probe


def select_best_candidate_fluent(
    scores: torch.Tensor,
    cum_logprob: torch.Tensor,
    fluency_weight: float,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fluency-blended selection: cumulative alignment + ``λ``·cumulative logprob.

    Mirrors the ``fluency_weight`` branch of the frame pipeline's
    ``FrameUnembeddingRepresentation._generate_guided`` so ``λ`` means exactly the
    same thing across representations: the winning candidate maximizes
    ``cumulative_alignment(last) + λ · cumulative_token_logprob`` rather than
    alignment alone, trading concept steering against the model's own fluency.

    The returned ``probe`` stays the **concept** (alignment) trajectory of the
    selected candidate — not the blended value — so probes remain comparable to
    the plain :func:`select_best_candidate` path.

    Args:
        scores: per-position alignment for each candidate, shape ``(n*k, T)``
            (``n`` blocks of ``k`` candidates).
        cum_logprob: cumulative token log-probability of each candidate beam,
            shape ``(n*k,)``.
        fluency_weight: ``λ >= 0``. ``0`` reproduces
            :func:`select_best_candidate` exactly.
        n: number of inputs.
        k: number of lookahead candidates per input.

    Returns:
        ``(col_idx, probe)`` — the winning candidate index per input ``(n,)`` and
        its cumulative-alignment trajectory ``(n, T)``.
    """
    cum = scores.unsqueeze(-1).cumsum(-2).reshape(n, k, -1)  # (n, k, T)
    blended = cum[..., -1] + fluency_weight * cum_logprob.reshape(n, k).to(cum.device)
    col_idx = blended.argmax(-1)  # (n,)
    probe = cum[torch.arange(n, device=cum.device), col_idx]  # (n, T)
    return col_idx, probe
