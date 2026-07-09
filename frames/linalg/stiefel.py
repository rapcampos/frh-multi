"""Riemannian geometry of the Stiefel manifold St(d, k) — canonical metric.

Building blocks for intrinsic (geodesic) frame aggregation, the RQ3 / E2
alternatives to the extrinsic chordal mean in `Concept.average`:

- `stiefel_exp` — closed-form geodesic (Edelman, Arias & Smith 1998, eq. 2.42).
- `stiefel_log` — iterative shooting algorithm (Zimmermann 2017). Only locally
  defined: raises if the target is outside the convergence region.
- `frechet_mean` — Karcher (geodesic Fréchet) mean via exp/log fixed-point
  iteration, initialized at the extrinsic mean.
- `aligned_mean` — generalized-Procrustes mean: each frame is rotated within
  its span (right O(k) action) to best match the current mean before chordal
  averaging. Invariant to right-rotations of the inputs up to output gauge —
  the property the plain extrinsic mean lacks entirely (the Stiefel/Grassmann
  mismatch of Step 5).

All functions accept batched inputs `(..., d, k)`; `frechet_mean` and
`aligned_mean` take the collection of points in the FIRST dimension,
`(m, ..., d, k)`. Internally everything runs in float64 (concept frames come
from fp16 models and are only approximately orthonormal — inputs to the mean
functions are re-projected onto the manifold first) and is cast back to the
input dtype.
"""

from __future__ import annotations

import torch


def _polar(matrix: torch.Tensor) -> torch.Tensor:
    """Orthogonal polar factor, preserving dtype (unlike solve_procrustes,
    which round-trips through float32)."""
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def _logm(matrix: torch.Tensor) -> torch.Tensor:
    """Principal matrix logarithm of a (near-)orthogonal matrix.

    Orthogonal matrices are normal, so the complex eigendecomposition is
    stable; the principal branch requires no eigenvalue at exactly -1.
    """
    evals, evecs = torch.linalg.eig(matrix)
    log_evals = torch.log(evals)
    return (evecs @ torch.diag_embed(log_evals) @ torch.linalg.inv(evecs)).real


def tangent_project(point: torch.Tensor, ambient: torch.Tensor) -> torch.Tensor:
    """Project an ambient vector onto the tangent space at `point`:
    Delta = V - X sym(X^T V), so X^T Delta is skew-symmetric."""
    xtv = point.mT @ ambient
    return ambient - point @ ((xtv + xtv.mT) / 2)


def canonical_norm(point: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    """Canonical-metric norm of a tangent vector at `point`:
    ||Delta||_c^2 = ||Delta||_F^2 - 0.5 ||X^T Delta||_F^2."""
    full = tangent.pow(2).sum(dim=(-2, -1))
    vertical = (point.mT @ tangent).pow(2).sum(dim=(-2, -1))
    return (full - vertical / 2).clamp(min=0).sqrt()


def stiefel_exp(point: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    """Geodesic from `point` with initial velocity `tangent` (canonical
    metric), evaluated at t=1."""
    x, delta = point.double(), tangent.double()
    k = x.size(-1)
    a = x.mT @ delta
    q, r = torch.linalg.qr(delta - x @ a)
    block = torch.cat(
        [
            torch.cat([a, -r.mT], dim=-1),
            torch.cat([r, torch.zeros_like(a)], dim=-1),
        ],
        dim=-2,
    )
    mexp = torch.matrix_exp(block)
    y = x @ mexp[..., :k, :k] + q @ mexp[..., k:, :k]
    return y.to(point.dtype)


def stiefel_log(
    point: torch.Tensor,
    target: torch.Tensor,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> torch.Tensor:
    """Riemannian logarithm log_point(target), canonical metric.

    Zimmermann's shooting algorithm (2017): iterate on a 2k x 2k orthogonal
    matrix until its lower-right log-block vanishes. Cheap here because k is
    the frame rank (a handful of token vectors), regardless of d.

    Raises RuntimeError if the iteration does not converge — the target is
    then outside the algorithm's convergence region (roughly, too far from
    `point` on the manifold).
    """
    x, y = point.double(), target.double()
    k = x.size(-1)

    m = x.mT @ y
    q, n = torch.linalg.qr(y - x @ m)

    # [M; N] has orthonormal columns; complete it to an orthogonal 2k x 2k
    # matrix V = [[M, *], [N, *]] via full QR (sign-corrected so the first
    # k columns are exactly [M; N]).
    stacked = torch.cat([m, n], dim=-2)
    v_full, r_full = torch.linalg.qr(stacked, mode="complete")
    signs = torch.diagonal(r_full, dim1=-2, dim2=-1).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    v_full = torch.cat([v_full[..., :k] * signs.unsqueeze(-2), v_full[..., k:]], dim=-1)

    # force det(V) = +1: with det = -1 an orthogonal matrix has an eigenvalue
    # at -1 and NO real principal logarithm — _logm would silently return
    # garbage. The completion columns' gauge is free, so flip one of them.
    det_signs = torch.linalg.det(v_full).sign()
    v_full = torch.cat(
        [v_full[..., :-1], v_full[..., -1:] * det_signs[..., None, None]], dim=-1
    )

    for _ in range(max_iter):
        log_v = _logm(v_full)
        c = log_v[..., k:, k:]
        residual = c.norm(dim=(-2, -1)).max()
        if residual <= tol:
            break
        lower_right = torch.matrix_exp(-c)
        v_full = torch.cat([v_full[..., :k], v_full[..., k:] @ lower_right], dim=-1)
    else:
        raise RuntimeError(
            f"stiefel_log did not converge (residual {residual:.2e} > {tol:.0e}); "
            "the target is likely outside the convergence region"
        )

    a = log_v[..., :k, :k]
    b = log_v[..., k:, :k]
    delta = x @ a + q @ b
    return delta.to(point.dtype)


def _prepare_points(
    points: torch.Tensor, weights: list[float] | torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-project points onto the manifold (fp16-born frames are only
    approximately orthonormal) and normalize nonnegative weights to sum 1."""
    points = _polar(points.double())
    if weights is None:
        weights = torch.ones(points.size(0), dtype=torch.float64)
    weights = torch.as_tensor(weights, dtype=torch.float64).to(points.device)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(
            "intrinsic means are Fréchet functionals: weights must be "
            "nonnegative with a positive sum (use the extrinsic mean or "
            "concept subtraction for repulsion)"
        )
    weights = weights / weights.sum()
    weights = weights.view(-1, *([1] * (points.dim() - 1)))
    return points, weights


def frechet_mean(
    points: torch.Tensor,
    weights: list[float] | torch.Tensor | None = None,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Weighted Karcher (geodesic Fréchet) mean on St(d, k), canonical metric.

    Fixed-point iteration X <- Exp_X(sum_i w_i Log_X(P_i)), initialized at the
    extrinsic (chordal) mean. `points` holds the collection in dim 0:
    (m, ..., d, k).
    """
    dtype = points.dtype
    points, weights = _prepare_points(points, weights)
    x = _polar((weights * points).sum(0))

    for _ in range(max_iter):
        tangents = torch.stack([stiefel_log(x, p) for p in points])
        step = (weights * tangents).sum(0)
        if canonical_norm(x, step).max() <= tol:
            break
        x = stiefel_exp(x, step)
    else:
        raise RuntimeError(f"frechet_mean did not converge in {max_iter} iterations")

    return x.to(dtype)


def aligned_mean(
    points: torch.Tensor,
    weights: list[float] | torch.Tensor | None = None,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> torch.Tensor:
    """Weighted generalized-Procrustes (rotation-aligned chordal) mean.

    Alternates two steps until the mean stabilizes: (1) rotate each frame by
    the k x k orthogonal R_i = polar(P_i^T X) that best matches the current
    mean — optimizing over each frame's right O(k) orbit; (2) take the
    extrinsic mean of the aligned frames.

    Gauge note: the fixed points form right-O(k) orbits (rotating any input
    leaves the aligned frames unchanged, and rotating the mean rotates it
    within its orbit). The mean is therefore invariant to right-rotations of
    the inputs only UP TO a right rotation of the output; the representative
    returned here is anchored by the extrinsic-mean initialization — the
    gauge FRH's rotation-sensitive trace scoring is known to work in.
    `points` holds the collection in dim 0: (m, ..., d, k).
    """
    dtype = points.dtype
    points, weights = _prepare_points(points, weights)
    x = _polar((weights * points).sum(0))

    for _ in range(max_iter):
        aligned = torch.stack([p @ _polar(p.mT @ x) for p in points])
        new_x = _polar((weights * aligned).sum(0))
        gap = (new_x - x).norm(dim=(-2, -1)).max()
        x = new_x
        if gap <= tol:
            break
    else:
        raise RuntimeError(f"aligned_mean did not converge in {max_iter} iterations")

    return x.to(dtype)
