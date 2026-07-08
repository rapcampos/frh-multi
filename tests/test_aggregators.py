"""CPU unit tests for Family 2 score aggregators and normalization."""

import pytest
import torch

from frames.representations import aggregators


def uniform(n: int) -> torch.Tensor:
    return torch.full((n,), 1.0 / n)


class TestWeightedSum:
    def test_known_argmax(self):
        # candidate 0 strong on concept 0, candidate 1 strong on concept 1
        scores = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weights = torch.tensor([2.0, 1.0])
        out = aggregators.weighted_sum(scores, weights)
        assert torch.allclose(out, torch.tensor([2.0, 1.0]))
        assert out.argmax() == 0

    def test_negative_weight_repels(self):
        scores = torch.tensor([[0.9, 0.9], [0.9, 0.0]])
        weights = torch.tensor([1.0, -1.0])
        out = aggregators.weighted_sum(scores, weights)
        assert out.argmax() == 1  # candidate aligned with the repelled concept loses

    def test_or_semantics_one_strong_concept_compensates(self):
        balanced = torch.tensor([0.4, 0.4])
        lopsided = torch.tensor([0.9, 0.0])
        scores = torch.stack([balanced, lopsided])
        out = aggregators.weighted_sum(scores, uniform(2))
        assert out.argmax() == 1


class TestSoftmin:
    def test_approaches_hard_min_as_tau_to_zero(self):
        scores = torch.tensor([[0.3, 0.8, 0.5]])
        out = aggregators.softmin(scores, torch.ones(3), tau=1e-4)
        assert torch.allclose(out, torch.tensor([0.3]), atol=1e-3)

    def test_and_semantics_prefers_balanced_candidate(self):
        balanced = torch.tensor([0.4, 0.4])
        lopsided = torch.tensor([0.9, 0.0])
        scores = torch.stack([balanced, lopsided])
        out = aggregators.softmin(scores, torch.ones(2), tau=0.1)
        assert out.argmax() == 0  # opposite of weighted_sum's choice

    def test_large_tau_blends_toward_average(self):
        scores = torch.tensor([[0.0, 1.0]])
        out = aggregators.softmin(scores, torch.ones(2), tau=100.0)
        # for large tau: softmin -> mean - tau*log(n_concepts) + ...; ordering
        # matches the average, so compare against another candidate
        scores2 = torch.tensor([[0.6, 0.6], [0.0, 1.0]])
        out2 = aggregators.softmin(scores2, torch.ones(2), tau=100.0)
        assert out2.argmax() == 0  # mean 0.6 > mean 0.5
        assert out.shape == (1,)

    def test_scorer_factory_binds_tau(self):
        scorer = aggregators.softmin_scorer(tau=1e-4)
        scores = torch.tensor([[0.3, 0.8]])
        assert torch.allclose(
            scorer(scores, torch.ones(2)), torch.tensor([0.3]), atol=1e-3
        )


class TestConstrained:
    def test_maximizes_primary_subject_to_thresholds(self):
        # (primary, secondary): candidate 1 has best primary but violates threshold
        scores = torch.tensor([[0.5, 0.4], [0.9, 0.1], [0.6, 0.3]])
        out = aggregators.constrained(scores, torch.ones(2), thresholds=[0.25])
        assert out.argmax() == 2  # 0.6 wins among satisfying candidates
        assert torch.isinf(out[1]) and out[1] < 0

    def test_finite_penalty_for_decoding_loop(self):
        scorer = aggregators.constrained_scorer(thresholds=[0.25], penalty=-1e4)
        scores = torch.tensor([[0.9, 0.1]])
        out = scorer(scores, torch.ones(2))
        assert torch.isfinite(out).all()
        assert out.item() == -1e4

    def test_multiple_secondary_constraints(self):
        scores = torch.tensor([[0.9, 0.5, 0.1], [0.5, 0.5, 0.5]])
        out = aggregators.constrained(scores, torch.ones(3), thresholds=[0.4, 0.4])
        assert out.argmax() == 1


class TestNormalizeScores:
    def test_zscore_zero_mean_unit_std(self):
        scores = torch.randn(8, 3)
        out = aggregators.normalize_scores(scores, "zscore", dim=0)
        assert torch.allclose(out.mean(0), torch.zeros(3), atol=1e-6)
        assert torch.allclose(out.std(0, unbiased=False), torch.ones(3), atol=1e-4)

    def test_zscore_invariant_to_affine_rescaling(self):
        # a concept whose raw scores are shifted/scaled (rank/norm artifacts)
        # normalizes to the same values — this is the point of the flag
        scores = torch.randn(8, 1)
        rescaled = 37.0 * scores + 5.0
        out_a = aggregators.normalize_scores(scores, "zscore", dim=0)
        out_b = aggregators.normalize_scores(rescaled, "zscore", dim=0)
        assert torch.allclose(out_a, out_b, atol=1e-4)

    def test_zscore_constant_scores_map_to_zero_not_nan(self):
        scores = torch.full((5, 2), 3.14)
        out = aggregators.normalize_scores(scores, "zscore", dim=0)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_rank_invariant_under_monotone_transform(self):
        scores = torch.randn(6, 2)
        out_a = aggregators.normalize_scores(scores, "rank", dim=0)
        out_b = aggregators.normalize_scores(scores.exp(), "rank", dim=0)
        assert torch.equal(out_a, out_b)

    def test_rank_spans_zero_to_one(self):
        scores = torch.tensor([[3.0], [1.0], [2.0]])
        out = aggregators.normalize_scores(scores, "rank", dim=0)
        assert torch.equal(out.squeeze(), torch.tensor([1.0, 0.0, 0.5]))

    def test_none_is_noop(self):
        scores = torch.randn(4, 2)
        assert torch.equal(aggregators.normalize_scores(scores, None), scores)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            aggregators.normalize_scores(torch.randn(4, 2), "minmax")


class TestLinearityConfound:
    def test_unnormalized_weighted_sum_equals_single_merged_concept_score(self):
        # trace is linear in the concept frame: sum_i w_i * <F, C_i> equals
        # <F, sum_i w_i C_i>. This is why normalization is mandatory —
        # without it, F2.a collapses into frame averaging (F1.a sans Procrustes).
        torch.manual_seed(0)
        frames = torch.randn(5, 16)  # 5 candidates, flattened frames
        c1, c2 = torch.randn(16), torch.randn(16)
        w1, w2 = 0.7, 0.3

        per_concept = torch.stack([frames @ c1, frames @ c2], dim=-1)
        f2a = aggregators.weighted_sum(per_concept, torch.tensor([w1, w2]))
        f1a = frames @ (w1 * c1 + w2 * c2)
        assert torch.allclose(f2a, f1a, atol=1e-5)

    def test_zscore_breaks_the_equivalence(self):
        # concept 2's scores are 10x larger in scale (rank/norm artifact).
        # raw weighted sum is dominated by concept 2 and picks candidate B;
        # after z-scoring, the balanced candidate C wins.
        per_concept = torch.tensor(
            [
                [1.0, 0.0],  # A: best on concept 1 only
                [0.0, 10.0],  # B: best on concept 2 only (inflated scale)
                [0.9, 9.0],  # C: near-best on both
            ]
        )
        weights = torch.tensor([0.5, 0.5])

        raw = aggregators.weighted_sum(per_concept, weights)
        normalized = aggregators.weighted_sum(
            aggregators.normalize_scores(per_concept, "zscore", dim=0), weights
        )
        assert raw.argmax() == 1  # scale domination
        assert normalized.argmax() == 2  # balance wins once scales are equalized
