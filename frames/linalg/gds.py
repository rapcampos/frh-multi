"""Generalized difference subspace (GDS) operations for concept frames.

Fukui & Maki's GDS: given class subspaces {F_i}, eigendecompose the sum of
their projectors G = sum_i F_i F_i^T and split the spectrum at eigenvalue 1.
Directions with eigenvalue > 1 form the principal/common subspace M (what the
classes share); the rest of the union span is the difference subspace, which
carries only discriminative structure. The original method projects DATA onto
the difference subspace before classifying; here the same split supports two
steering constructions:

- `remove_common` — the faithful use: project each concept frame off the
  common subspace (guides then compete only on what distinguishes them).
  A fixed projector acts column-by-column, so the token-ordered gauge that
  the Stiefel correlation scorer relies on is preserved.
- `gds_blend` — the FRH-native hybrid: project the raw frame SUM off the
  common subspace, then polar-project back onto the manifold. The sum's
  token-ordered gauge anchors the output (same resolution as the aligned
  mean); note that without per-frame re-orthonormalization this equals the
  blend of `remove_common` outputs, since the projector is linear.

Everything computes in float64 and casts back to the input dtype. Zero-padded
columns (the codebase's unequal-rank policy) are preserved as zeros: columns
that fall below `eps` after projection are zeroed rather than renormalized,
so a concept living mostly inside the common subspace loses rank instead of
being blown up out of it.
"""

from __future__ import annotations

import torch

EPS = 1e-6


def _polar_preserving_zeros(matrix: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Polar factor of (d, k) `matrix`, keeping (near-)zero columns as zeros.

    A plain polar decomposition invents arbitrary directions for rank-deficient
    columns; here they stay zero (rank counts nonzero columns downstream).
    """
    norms = matrix.norm(dim=-2)
    keep = norms > eps
    out = torch.zeros_like(matrix)
    if keep.any():
        u, _, vh = torch.linalg.svd(matrix[:, keep], full_matrices=False)
        out[:, keep] = u @ vh
    return out


def gds_split(
    frames: torch.Tensor, common_threshold: float = 1.05, eps: float = EPS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the union span of concept frames into common and difference parts.

    Args:
        frames: Stack of orthonormal frames, shape (n, d, k) (zero-padding to
            a common k is fine — zero columns contribute nothing).
        common_threshold: Eigenvalues of G = sum_i F_i F_i^T above this are
            common directions. 1 is the theoretical boundary (canonical pairs
            give 1 +- cos(theta)); the default keeps a small margin so nearly
            independent directions are not treated as shared.
        eps: Numerical floor below which directions are discarded entirely.

    Returns:
        (common, difference, eigenvalues): orthonormal bases (d, m) and
        (d, r) plus the eigenvalue spectrum of G, all float64.
    """
    n, d, k = frames.shape
    basis = frames.permute(1, 0, 2).reshape(d, n * k).double()
    u, s, _ = torch.linalg.svd(basis, full_matrices=False)
    eigenvalues = s.pow(2)
    common = eigenvalues > common_threshold
    difference = (eigenvalues > eps) & ~common
    return u[:, common], u[:, difference], eigenvalues


def remove_common(
    frames: torch.Tensor,
    common_threshold: float = 1.05,
    reorthonormalize: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Faithful GDS preprocessing: project each frame off the common subspace.

    Columns that lie (almost) entirely inside the common subspace are zeroed
    (rank reduction), not renormalized. With `reorthonormalize`, each frame is
    polar-projected back onto the Stiefel manifold (gauge-equivariant, so the
    column/token pairing survives).

    Args:
        frames: Stack of orthonormal frames, shape (n, d, k).

    Returns:
        torch.Tensor: Same shape and dtype as `frames`.
    """
    common, _, _ = gds_split(frames, common_threshold, eps)
    projected = frames.double() - common @ (common.mT @ frames.double())
    norms = projected.norm(dim=-2, keepdim=True)
    projected = torch.where(norms > eps, projected, torch.zeros_like(projected))
    if reorthonormalize:
        projected = torch.stack([_polar_preserving_zeros(f, eps) for f in projected])
    return projected.to(frames.dtype)


def gds_blend(
    frames: torch.Tensor,
    weights: list[float] | torch.Tensor | None = None,
    common_threshold: float = 1.05,
    eps: float = EPS,
) -> torch.Tensor:
    """GDS-anchored composite guide: polar( (I - MM^T) sum_i w_i F_i ).

    The raw frame sum carries the token-ordered gauge; the common projector
    removes shared structure column-by-column without disturbing it.

    Args:
        frames: Stack of orthonormal frames, shape (n, d, k).
        weights: Optional per-frame weights (default uniform).

    Returns:
        torch.Tensor: Composite frame of shape (d, k), dtype of `frames`.
    """
    common, _, _ = gds_split(frames, common_threshold, eps)
    if weights is None:
        weights = torch.ones(frames.size(0), device=frames.device)
    weights = torch.as_tensor(weights, dtype=torch.float64, device=frames.device)
    summed = (frames.double() * weights.view(-1, 1, 1)).sum(0)
    projected = summed - common @ (common.mT @ summed)
    return _polar_preserving_zeros(projected, eps).to(frames.dtype)


def common_blend(
    frames: torch.Tensor,
    weights: list[float] | torch.Tensor | None = None,
    common_threshold: float = 1.05,
    eps: float = EPS,
) -> torch.Tensor:
    """Inverse of `gds_blend`: polar( MM^T sum_i w_i F_i ) — keep ONLY the
    common component of the frame sum.

    Motivated by the notebook-33 result: for JOINT expression the common
    subspace carries the shared semantic neighborhood — the cheapest region
    for a text to express all concepts at once — so a common-only guide is
    the objective-matched composition operator this predicts. Same gauge
    anchoring as `gds_blend` (projector on the token-ordered sum, polar
    back to the manifold).

    Args:
        frames: Stack of orthonormal frames, shape (n, d, k).
        weights: Optional per-frame weights (default uniform).

    Raises:
        ValueError: If no direction exceeds `common_threshold` (the concepts
            share no common subspace at this threshold).

    Returns:
        torch.Tensor: Composite frame of shape (d, k), dtype of `frames`.
    """
    common, _, _ = gds_split(frames, common_threshold, eps)
    if common.shape[1] == 0:
        raise ValueError("no common subspace above threshold")
    if weights is None:
        weights = torch.ones(frames.size(0), device=frames.device)
    weights = torch.as_tensor(weights, dtype=torch.float64, device=frames.device)
    summed = (frames.double() * weights.view(-1, 1, 1)).sum(0)
    projected = common @ (common.mT @ summed)
    return _polar_preserving_zeros(projected, eps).to(frames.dtype)


def difference_subspace(
    frame1: torch.Tensor, frame2: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Classical two-class difference subspace (Fukui & Maki).

    Basis of normalized differences of canonical vector pairs: with
    F1^T F2 = U diag(cos theta) V^T, the canonical vectors are a_i = F1 u_i,
    b_i = F2 v_i and d_i = (a_i - b_i) / ||a_i - b_i||. Pairs with
    cos(theta_i) ~ 1 (shared directions) yield near-zero differences and are
    dropped — the multi-directional, eigen-based generalization of the
    Procrustes frame difference.

    Args:
        frame1: Orthonormal frame (d, k1).
        frame2: Orthonormal frame (d, k2).

    Returns:
        torch.Tensor: Difference-subspace basis (d, r), r <= min(k1, k2),
            dtype of `frame1`.
    """
    f1, f2 = frame1.double(), frame2.double()
    u, _, vh = torch.linalg.svd(f1.mT @ f2, full_matrices=False)
    diffs = f1 @ u - f2 @ vh.mT
    norms = diffs.norm(dim=-2)
    keep = norms > eps
    return (diffs[:, keep] / norms[keep]).to(frame1.dtype)
