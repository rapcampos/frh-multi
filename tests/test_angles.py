"""CPU unit tests for Grassmann angle metrics (metric v3 building blocks).

Small synthetic frames (d=8, k=3), float64, seeded. Verifies boundary
values, gauge invariance (the defining design property), symmetry, and
scale invariance.
"""

import math

import torch

from frames.linalg.angles import (
    grassmann_distance,
    mean_canonical_correlation,
    principal_angles,
    text_frame,
    vector_subspace_cosine,
)

D, K = 8, 3


def random_point(seed: int, d: int = D, k: int = K) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(d, k, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(gaussian)
    return q


def random_rotation(seed: int, k: int = K) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(k, k, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    return q * torch.sign(torch.diagonal(r))


class TestPrincipalAngles:
    def test_self_angles_are_zero(self):
        x = random_point(0)
        assert torch.allclose(
            principal_angles(x, x), torch.zeros(K, dtype=torch.float64), atol=1e-7
        )
        assert float(grassmann_distance(x, x)) < 1e-7
        assert abs(float(mean_canonical_correlation(x, x)) - 1.0) < 1e-12

    def test_orthogonal_spans_are_maximally_distant(self):
        eye = torch.eye(D, dtype=torch.float64)
        x, y = eye[:, :K], eye[:, K : 2 * K]
        theta = principal_angles(x, y)
        assert torch.allclose(
            theta, torch.full((K,), math.pi / 2, dtype=torch.float64), atol=1e-12
        )
        assert abs(float(grassmann_distance(x, y)) - math.pi / 2 * math.sqrt(K)) < 1e-9
        assert float(mean_canonical_correlation(x, y)) < 1e-12

    def test_gauge_invariance(self):
        x, y = random_point(1), random_point(2)
        rotated = y @ random_rotation(3)
        assert torch.allclose(
            grassmann_distance(x, y), grassmann_distance(x, rotated), atol=1e-9
        )
        assert torch.allclose(
            mean_canonical_correlation(x, y),
            mean_canonical_correlation(x, rotated),
            atol=1e-9,
        )

    def test_symmetry(self):
        x, y = random_point(4), random_point(5)
        assert torch.allclose(
            grassmann_distance(x, y), grassmann_distance(y, x), atol=1e-12
        )

    def test_different_ranks_give_min_k_angles(self):
        x = random_point(6, k=K)
        y = random_point(7, k=K - 1)
        assert principal_angles(x, y).shape == (K - 1,)


class TestVectorSubspaceCosine:
    def test_vector_in_span_scores_one(self):
        x = random_point(8)
        h = (x @ torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)).unsqueeze(0)
        assert abs(float(vector_subspace_cosine(h, x)) - 1.0) < 1e-12

    def test_orthogonal_vector_scores_zero(self):
        eye = torch.eye(D, dtype=torch.float64)
        x = eye[:, :K]
        h = eye[:, K].unsqueeze(0)
        assert float(vector_subspace_cosine(h, x)) < 1e-12

    def test_scale_invariance(self):
        x = random_point(9)
        generator = torch.Generator().manual_seed(10)
        h = torch.randn(4, D, generator=generator, dtype=torch.float64)
        assert torch.allclose(
            vector_subspace_cosine(h, x),
            vector_subspace_cosine(37.5 * h, x),
            atol=1e-12,
        )

    def test_zero_vector_scores_zero(self):
        x = random_point(11)
        h = torch.zeros(1, D, dtype=torch.float64)
        assert float(vector_subspace_cosine(h, x)) == 0.0


class TestTextFrame:
    def test_output_is_orthonormal(self):
        generator = torch.Generator().manual_seed(12)
        h = torch.randn(20, D, generator=generator, dtype=torch.float64)
        q = text_frame(h, K)
        assert q.shape == (D, K)
        assert torch.allclose(q.mT @ q, torch.eye(K, dtype=torch.float64), atol=1e-10)

    def test_recovers_dominant_directions_of_low_rank_text(self):
        x = random_point(13)
        generator = torch.Generator().manual_seed(14)
        coefficients = torch.randn(50, K, generator=generator, dtype=torch.float64)
        h = coefficients @ x.mT  # all positions lie in span(x)
        q = text_frame(h, K)
        # arccos is ill-conditioned at sigma = 1, so allow a looser tolerance
        assert float(grassmann_distance(q, x)) < 1e-6

    def test_k_capped_by_positions(self):
        generator = torch.Generator().manual_seed(15)
        h = torch.randn(2, D, generator=generator, dtype=torch.float64)
        assert text_frame(h, K).shape == (D, 2)
