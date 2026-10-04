"""Math-only tests for norm-normalized subspace alignment.

``subspace_alignment`` scores a candidate by the projection magnitude ``‖Bᵀh‖``,
which is unbounded and grows with ``‖h‖``. The normalized mode divides by ``‖h‖``
to score the *fraction* of hidden-state energy inside the subspace, putting the
alignment term on a bounded [0, 1] scale — comparable to the frame pipeline's
bounded correlation, and matching the cosine metric the battery reports.

Project convention: pytest for math, experiments in notebooks.
"""

import pytest
import torch

from subspace.generation import subspace_alignment


def _orthonormal_basis(d: int, r: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, r, generator=g))
    return q


def test_default_is_unnormalized_and_unchanged():
    """The default path must stay bit-for-bit identical: NB68 depends on it."""
    d, r = 6, 3
    basis = _orthonormal_basis(d, r, seed=1)
    h = torch.randn(4, 7, d, generator=torch.Generator().manual_seed(2))

    expected = (h.float() @ basis.float()).norm(dim=-1)
    assert torch.equal(subspace_alignment(h, basis), expected)
    assert torch.equal(subspace_alignment(h, basis, normalize=False), expected)


def test_normalized_is_bounded_in_unit_interval():
    d, r = 8, 3
    basis = _orthonormal_basis(d, r, seed=3)
    h = torch.randn(5, 9, d, generator=torch.Generator().manual_seed(4))

    scores = subspace_alignment(h, basis, normalize=True)
    assert scores.shape == h.shape[:-1]
    assert (scores >= 0).all()
    assert (scores <= 1 + 1e-6).all()


def test_normalized_is_scale_invariant_in_h():
    """The unnormalized score rewards large ‖h‖; the normalized one must not."""
    d, r = 8, 4
    basis = _orthonormal_basis(d, r, seed=5)
    h = torch.randn(3, 6, d, generator=torch.Generator().manual_seed(6))

    base = subspace_alignment(h, basis, normalize=True)
    for scale in (0.1, 2.0, 50.0):
        scaled = subspace_alignment(h * scale, basis, normalize=True)
        assert torch.allclose(scaled, base, atol=1e-5)

    # The unnormalized score does scale — this is the behaviour being replaced.
    raw = subspace_alignment(h, basis)
    assert not torch.allclose(subspace_alignment(h * 2.0, basis), raw, atol=1e-3)


def test_normalized_hits_one_inside_the_subspace():
    d, r = 6, 2
    basis = _orthonormal_basis(d, r, seed=7)
    # A hidden state lying exactly in span(basis).
    coeffs = torch.randn(4, r, generator=torch.Generator().manual_seed(8))
    h = coeffs @ basis.mT

    scores = subspace_alignment(h, basis, normalize=True)
    assert torch.allclose(scores, torch.ones(4), atol=1e-5)


def test_normalized_is_zero_orthogonal_to_the_subspace():
    d, r = 6, 2
    full = _orthonormal_basis(d, d, seed=9)
    basis, orthogonal = full[:, :r], full[:, r:]
    coeffs = torch.randn(4, d - r, generator=torch.Generator().manual_seed(10))
    h = coeffs @ orthogonal.mT  # orthogonal to span(basis)

    scores = subspace_alignment(h, basis, normalize=True)
    assert torch.allclose(scores, torch.zeros(4), atol=1e-5)


def test_normalized_repulsion_stays_signed_and_bounded():
    d, r = 10, 3
    attract = _orthonormal_basis(d, r, seed=11)
    repel = _orthonormal_basis(d, r, seed=12)
    h = torch.randn(4, 5, d, generator=torch.Generator().manual_seed(13))

    signed = subspace_alignment(h, attract, repel, normalize=True)
    unsigned = subspace_alignment(h, attract, normalize=True)

    assert signed.shape == unsigned.shape
    assert (signed <= unsigned + 1e-6).all()
    # Difference of two [0, 1] quantities.
    assert (signed >= -1 - 1e-6).all() and (signed <= 1 + 1e-6).all()


def test_zero_hidden_state_does_not_produce_nan():
    """Padding positions can be exactly zero; the score must stay finite."""
    d, r = 6, 2
    basis = _orthonormal_basis(d, r, seed=14)
    h = torch.zeros(2, 3, d)

    scores = subspace_alignment(h, basis, normalize=True)
    assert torch.isfinite(scores).all()
    assert torch.allclose(scores, torch.zeros_like(scores))


@pytest.mark.parametrize("normalize", [False, True])
def test_shape_is_preserved(normalize):
    d, r = 7, 3
    basis = _orthonormal_basis(d, r, seed=15)
    h = torch.randn(2, 11, d, generator=torch.Generator().manual_seed(16))
    assert subspace_alignment(h, basis, normalize=normalize).shape == (2, 11)
