"""Grassmann angle/distance metrics between subspaces (metric v3 building blocks).

Angles between the span of a concept frame and either single hidden-state
vectors or a text's dominant subspace. All quantities here are deliberately
**gauge-invariant** (subspace/Grassmann geometry): unchanged under column
reordering and sign flips of the input frames. This is the opposite design
choice from the frame correlation used by the steering objective, which is a
Stiefel quantity — evaluation asks "did the text land near the concept's
span", not "in the right token order". Bounded and scale-invariant, so
values are comparable across models, precisions, and concepts.

All functions accept batched inputs and compute in float64 (frames come from
fp16/bf16 models and are only approximately orthonormal); results are
returned in float64.
"""

from __future__ import annotations

import torch

EPS = 1e-8


def principal_angles(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    """Principal angles between the spans of two orthonormal frames.

    Args:
        frame1: Orthonormal frame(s) of shape (..., d, k1).
        frame2: Orthonormal frame(s) of shape (..., d, k2).

    Returns:
        torch.Tensor: Angles in radians, ascending, shape
            (..., min(k1, k2)); theta_i = arccos sigma_i for the singular
            values of frame1^T frame2, clamped to [0, 1].
    """
    overlap = frame1.double().mT @ frame2.double()
    sigma = torch.linalg.svdvals(overlap).clamp(0.0, 1.0)
    return torch.arccos(sigma)


def grassmann_distance(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on the Grassmann manifold: ||theta||_2.

    Range [0, pi/2 * sqrt(min(k1, k2))]; 0 iff the spans coincide.
    """
    return principal_angles(frame1, frame2).norm(dim=-1)


def mean_canonical_correlation(
    frame1: torch.Tensor, frame2: torch.Tensor
) -> torch.Tensor:
    """Mean cosine of the principal angles, in [0, 1]; 1 iff spans coincide."""
    overlap = frame1.double().mT @ frame2.double()
    return torch.linalg.svdvals(overlap).clamp(0.0, 1.0).mean(dim=-1)


def vector_subspace_cosine(
    vectors: torch.Tensor, frame: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Cosine of the angle between each vector and an orthonormal frame's span.

    ||F^T h|| / ||h|| — the fraction of the vector's norm lying in the span.

    Args:
        vectors: Tensor of shape (..., T, d).
        frame: Orthonormal frame of shape (d, k).
        eps: Zero-vector guard; vectors with norm <= eps score 0.

    Returns:
        torch.Tensor: Cosines in [0, 1], shape (..., T).
    """
    v = vectors.double()
    projected = (v @ frame.double()).norm(dim=-1)
    norms = v.norm(dim=-1)
    cosine = (projected / norms.clamp_min(eps)).clamp(0.0, 1.0)
    return torch.where(norms > eps, cosine, torch.zeros_like(cosine))


def text_frame(hidden_states: torch.Tensor, k: int) -> torch.Tensor:
    """Orthonormal basis of a text's dominant k-dimensional subspace.

    Top-k right singular directions of the stacked hidden states — the
    directions in R^d carrying the most variance across positions.

    Args:
        hidden_states: Tensor of shape (T, d).
        k: Target dimension; capped at min(T, d).

    Returns:
        torch.Tensor: Orthonormal frame of shape (d, min(k, T, d)).
    """
    _, _, vh = torch.linalg.svd(hidden_states.double(), full_matrices=False)
    return vh[: min(k, vh.size(-2))].mT
