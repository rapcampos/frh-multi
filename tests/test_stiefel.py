"""CPU unit tests for Stiefel geometry (exp/log/Karcher mean, GPA mean).

Small synthetic frames (d=8, k=3), float64. Random points are drawn close
enough to stay inside stiefel_log's convergence region.
"""

import pytest
import torch

from frames.linalg.stiefel import (
    aligned_mean,
    canonical_norm,
    frechet_mean,
    stiefel_exp,
    stiefel_log,
    tangent_project,
)
from frames.representations.concept import Concept

D, K = 8, 3


def random_point(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(D, K, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(gaussian)
    return q


def random_tangent(point: torch.Tensor, seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    ambient = torch.randn(D, K, generator=generator, dtype=torch.float64)
    tangent = tangent_project(point, ambient)
    return scale * tangent / canonical_norm(point, tangent)


def assert_on_manifold(point: torch.Tensor, atol: float = 1e-10):
    gram = point.mT @ point
    assert torch.allclose(gram, torch.eye(point.size(-1), dtype=point.dtype), atol=atol)


class TestExpLog:
    def test_tangent_projection_makes_xtv_skew(self):
        x = random_point(0)
        delta = tangent_project(x, torch.randn(D, K, dtype=torch.float64))
        xtv = x.mT @ delta
        assert torch.allclose(xtv, -xtv.mT, atol=1e-12)

    def test_exp_stays_on_manifold(self):
        x = random_point(1)
        y = stiefel_exp(x, random_tangent(x, 2, scale=1.2))
        assert_on_manifold(y)

    def test_exp_of_zero_is_identity(self):
        x = random_point(3)
        assert torch.allclose(stiefel_exp(x, torch.zeros_like(x)), x, atol=1e-12)

    def test_log_of_self_is_zero(self):
        x = random_point(4)
        assert float(canonical_norm(x, stiefel_log(x, x))) < 1e-8

    def test_exp_log_roundtrip(self):
        x = random_point(5)
        delta = random_tangent(x, 6, scale=0.8)
        recovered = stiefel_log(x, stiefel_exp(x, delta))
        assert torch.allclose(recovered, delta, atol=1e-8)

    def test_log_exp_roundtrip(self):
        x = random_point(7)
        y = stiefel_exp(x, random_tangent(x, 8, scale=1.0))
        assert torch.allclose(stiefel_exp(x, stiefel_log(x, y)), y, atol=1e-8)

    def test_far_pair_with_reflection_completion(self):
        # regression: the orthogonal completion in stiefel_log can come out
        # with det = -1, which has no real principal logarithm — the log then
        # silently "converged" to garbage. d=16, seeds (0, 1) hit that case.
        generator = torch.Generator().manual_seed(0)
        x, _ = torch.linalg.qr(
            torch.randn(16, 3, generator=generator, dtype=torch.float64)
        )
        generator = torch.Generator().manual_seed(1)
        y, _ = torch.linalg.qr(
            torch.randn(16, 3, generator=generator, dtype=torch.float64)
        )
        delta = stiefel_log(x, y)
        assert float((stiefel_exp(x, delta) - y).norm()) < 1e-8

    def test_distance_is_symmetric(self):
        x = random_point(9)
        y = stiefel_exp(x, random_tangent(x, 10, scale=0.9))
        d_xy = float(canonical_norm(x, stiefel_log(x, y)))
        d_yx = float(canonical_norm(y, stiefel_log(y, x)))
        assert d_xy == pytest.approx(d_yx, rel=1e-6)
        assert d_xy == pytest.approx(0.9, rel=1e-6)


class TestFrechetMean:
    def test_identical_points_fixed_point(self):
        x = random_point(11)
        mean = frechet_mean(torch.stack([x, x, x]))
        assert torch.allclose(mean, x, atol=1e-8)

    def test_zero_weight_ignores_a_point(self):
        a, b = random_point(12), random_point(13)
        mean = frechet_mean(torch.stack([a, b]), weights=[1.0, 0.0])
        assert torch.allclose(mean, a, atol=1e-8)

    def test_two_point_mean_is_geodesic_midpoint(self):
        a = random_point(14)
        b = stiefel_exp(a, random_tangent(a, 15, scale=1.0))
        mean = frechet_mean(torch.stack([a, b]))
        midpoint = stiefel_exp(a, stiefel_log(a, b) / 2)
        assert torch.allclose(mean, midpoint, atol=1e-6)
        d_a = float(canonical_norm(mean, stiefel_log(mean, a)))
        d_b = float(canonical_norm(mean, stiefel_log(mean, b)))
        assert d_a == pytest.approx(d_b, rel=1e-5)

    def test_result_on_manifold(self):
        points = torch.stack([random_point(s) for s in (16, 17, 18)])
        assert_on_manifold(frechet_mean(points), atol=1e-8)

    def test_negative_weights_rejected(self):
        points = torch.stack([random_point(19), random_point(20)])
        with pytest.raises(ValueError, match="nonnegative"):
            frechet_mean(points, weights=[1.0, -1.0])

    def test_differs_from_extrinsic_for_far_points(self):
        from frames.linalg.orthogonalization import solve_procrustes

        points = torch.stack([random_point(s) for s in (21, 22, 23)])
        extrinsic = solve_procrustes(points.mean(0).float()).double()
        intrinsic = frechet_mean(points)
        assert not torch.allclose(intrinsic, extrinsic, atol=1e-3)


class TestAlignedMean:
    def test_identical_points_fixed_point(self):
        x = random_point(24)
        mean = aligned_mean(torch.stack([x, x]))
        assert torch.allclose(mean, x, atol=1e-10)

    def test_invariant_to_right_rotation_of_inputs_up_to_gauge(self):
        from frames.linalg.stiefel import _polar

        a, b = random_point(25), random_point(26)
        generator = torch.Generator().manual_seed(27)
        rotation, _ = torch.linalg.qr(
            torch.randn(K, K, generator=generator, dtype=torch.float64)
        )
        plain = aligned_mean(torch.stack([a, b]))
        rotated = aligned_mean(torch.stack([a @ rotation, b]))
        # fixed points form right-O(k) orbits: results agree after re-gauging
        regauged = rotated @ _polar(rotated.mT @ plain)
        assert torch.allclose(regauged, plain, atol=1e-6)

    def test_extrinsic_mean_lacks_that_invariance(self):
        from frames.linalg.orthogonalization import solve_procrustes

        a, b = random_point(28), random_point(29)
        generator = torch.Generator().manual_seed(30)
        rotation, _ = torch.linalg.qr(
            torch.randn(K, K, generator=generator, dtype=torch.float64)
        )
        plain = solve_procrustes((a + b).float())
        rotated = solve_procrustes((a @ rotation + b).float())
        assert not torch.allclose(plain, rotated, atol=1e-3)

    def test_result_on_manifold(self):
        points = torch.stack([random_point(s) for s in (31, 32, 33)])
        assert_on_manifold(aligned_mean(points), atol=1e-10)


def make_concept(name: str, seed: int, k_eff: int, k: int = K) -> Concept:
    tensor = random_point(seed)[:, :k_eff]
    tensor = torch.nn.functional.pad(tensor, (0, k - k_eff))
    return Concept(synset=name, tensor=tensor.unsqueeze(0).float())


class TestConceptAverageMethods:
    def test_default_method_unchanged(self):
        a, b = make_concept("a.n.01", 34, K), make_concept("b.n.01", 35, K)
        default = Concept.average([a, b])
        explicit = Concept.average([a, b], method="extrinsic")
        assert torch.equal(default.tensor, explicit.tensor)
        assert default.synset == "a.n.01 | b.n.01"

    def test_unknown_method_rejected(self):
        a, b = make_concept("a.n.01", 36, K), make_concept("b.n.01", 37, K)
        with pytest.raises(ValueError, match="unknown average method"):
            Concept.average([a, b], method="grassmann")

    @pytest.mark.parametrize("method", ["aligned", "frechet"])
    def test_intrinsic_methods_return_orthonormal_concept(self, method):
        a, b = make_concept("a.n.01", 38, K), make_concept("b.n.01", 39, K)
        mean = Concept.average([a, b], method=method)
        assert mean.synset == "a.n.01 | b.n.01"
        assert mean.tensor.shape == a.tensor.shape
        gram = mean.tensor[0].mT.double() @ mean.tensor[0].double()
        assert torch.allclose(gram, torch.eye(K, dtype=torch.float64), atol=1e-3)

    @pytest.mark.parametrize("method", ["aligned", "frechet"])
    def test_padded_columns_restored_as_zeros(self, method):
        a = make_concept("a.n.01", 40, k_eff=2)
        b = make_concept("b.n.01", 41, k_eff=2)
        mean = Concept.average([a, b], method=method)
        assert torch.equal(mean.tensor[..., 2], torch.zeros_like(mean.tensor[..., 2]))
        gram = mean.tensor[0, :, :2].mT.double() @ mean.tensor[0, :, :2].double()
        assert torch.allclose(gram, torch.eye(2, dtype=torch.float64), atol=1e-3)

    def test_unequal_effective_ranks_rejected(self):
        a = make_concept("a.n.01", 42, k_eff=3)
        b = make_concept("b.n.01", 43, k_eff=2)
        with pytest.raises(ValueError, match="equal effective ranks"):
            Concept.average([a, b], method="frechet")

    def test_negative_weights_rejected_for_intrinsic_only(self):
        a, b = make_concept("a.n.01", 44, K), make_concept("b.n.01", 45, K)
        Concept.average([a, b], weights=[1.0, -1.0])  # extrinsic: fine (repel)
        with pytest.raises(ValueError, match="nonnegative"):
            Concept.average([a, b], weights=[1.0, -1.0], method="aligned")
