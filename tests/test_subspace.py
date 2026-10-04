"""Math-only tests for the subspace concept pipeline.

These exercise the pure-linear-algebra core (SVD subspace construction,
projection, candidate selection) on synthetic tensors — no model, no GPU, no cuDF.
The model-in-the-loop behaviour is demonstrated in
``playground/subspace_pipeline.ipynb`` instead (project convention: pytest for
math, experiments in notebooks).
"""

import numpy as np
import torch

from subspace.concept import (
    ConceptSubspace,
    blend_bases,
    lemma_token_rows,
    subspace_from_features,
    subspace_from_lemma_features,
)
from subspace.generation import (
    select_best_candidate,
    select_best_candidate_fluent,
    subspace_alignment,
)


def _orthonormal_basis(d: int, r: int, seed: int = 0) -> torch.Tensor:
    """An orthonormal (d, r) basis for tests."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, r, generator=g))
    return q


def test_subspace_from_features_is_orthonormal():
    g = torch.Generator().manual_seed(1)
    features = torch.randn(10, 6, generator=g)
    basis = subspace_from_features(features, rank=4)

    assert basis.shape == (6, 4)
    gram = basis.mT @ basis
    assert torch.allclose(gram, torch.eye(4), atol=1e-5)


def test_subspace_spans_input_rows():
    # Features that live exactly in a known 3-dim subspace of R^6.
    d, true_rank = 6, 3
    sub = _orthonormal_basis(d, true_rank, seed=2)
    g = torch.Generator().manual_seed(3)
    coeffs = torch.randn(20, true_rank, generator=g)
    features = coeffs @ sub.mT  # each row is in span(sub)

    basis = subspace_from_features(features, rank=true_rank)
    concept = ConceptSubspace(synset="x.n.01", basis=basis)

    # Every input row should be fully captured by the recovered subspace.
    ef = concept.energy_fraction(features)
    assert torch.allclose(ef, torch.ones(20), atol=1e-4)


def test_fixed_rank_zero_pads_deficient_directions():
    # Features span only 2 dims but we ask for rank 4 -> 2 zero columns.
    d = 6
    sub = _orthonormal_basis(d, 2, seed=4)
    g = torch.Generator().manual_seed(5)
    features = torch.randn(15, 2, generator=g) @ sub.mT

    basis = subspace_from_features(features, rank=4)
    assert basis.shape == (6, 4)

    col_norms = basis.norm(dim=0)
    assert (col_norms[:2] > 0.5).all()
    assert torch.allclose(col_norms[2:], torch.zeros(2), atol=1e-5)


def test_fewer_tokens_than_rank_zero_pads():
    # Only one token vector, rank 4 -> a single unit column, rest zero.
    features = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    basis = subspace_from_features(features, rank=4)
    assert basis.shape == (4, 4)
    assert basis.norm(dim=0)[0] > 0.5
    assert torch.allclose(basis.norm(dim=0)[1:], torch.zeros(3), atol=1e-5)


def test_project_and_energy_fraction():
    d, r = 5, 2
    basis = _orthonormal_basis(d, r, seed=6)
    concept = ConceptSubspace(synset="y.n.01", basis=basis)

    inside = basis[:, 0] * 2.0  # lives in the subspace

    # Build a vector orthogonal to the subspace by projecting the span out.
    v = torch.randn(d, generator=torch.Generator().manual_seed(7))
    ortho = v - basis @ (basis.mT @ v)
    assert (basis.mT @ ortho).norm() < 1e-5

    assert torch.isclose(concept.project(inside), torch.tensor(2.0), atol=1e-5)
    assert torch.isclose(concept.energy_fraction(inside), torch.tensor(1.0), atol=1e-5)
    assert concept.project(ortho) < 1e-4
    assert concept.energy_fraction(ortho) < 1e-4


def test_rank_and_len():
    basis = _orthonormal_basis(8, 3, seed=8)
    concept = ConceptSubspace(synset="z.n.01", basis=basis)
    assert concept.rank == 3
    assert len(concept) == 3


def test_subspace_alignment_matches_project():
    d, r = 6, 3
    basis = _orthonormal_basis(d, r, seed=9)
    concept = ConceptSubspace(synset="a.n.01", basis=basis)
    h = torch.randn(4, 7, d, generator=torch.Generator().manual_seed(10))
    assert torch.allclose(subspace_alignment(h, basis), concept.project(h), atol=1e-5)


def test_select_best_candidate_picks_highest_cumulative():
    # n=1 input, k=2 candidates, T=2 positions.
    # candidate 0 cumulative at last position: 0.1 + 0.1 = 0.2
    # candidate 1 cumulative at last position: 0.3 + 0.4 = 0.7 -> selected
    scores = torch.tensor([[0.1, 0.1], [0.3, 0.4]])
    col_idx, probe = select_best_candidate(scores, n=1, k=2)
    assert col_idx.tolist() == [1]
    assert torch.isclose(probe[0, -1], torch.tensor(0.7), atol=1e-6)


def test_select_best_candidate_batched():
    # n=2 inputs, k=2 candidates each.
    # Rows ordered as [in0-c0, in0-c1, in1-c0, in1-c1].
    scores = torch.tensor(
        [
            [0.5, 0.5],  # in0 candidate0 -> 1.0
            [0.1, 0.1],  # in0 candidate1 -> 0.2
            [0.0, 0.0],  # in1 candidate0 -> 0.0
            [0.2, 0.9],  # in1 candidate1 -> 1.1
        ]
    )
    col_idx, _ = select_best_candidate(scores, n=2, k=2)
    assert col_idx.tolist() == [0, 1]


# -- hierarchical (subspace-of-subspaces) construction ------------------------


def test_lemma_features_orthonormal_and_zero_pads():
    # 3 lemmas, r_lemma=1 -> 3 concept directions; rank 4 -> 1 trailing zero col.
    g = torch.Generator().manual_seed(11)
    lemmas = [torch.randn(rows, 6, generator=g) for rows in (5, 3, 4)]
    basis = subspace_from_lemma_features(lemmas, rank=4, r_lemma=1)

    assert basis.shape == (6, 4)
    col_norms = basis.norm(dim=0)
    assert (col_norms[:3] > 0.5).all()
    assert torch.allclose(col_norms[3:], torch.zeros(1), atol=1e-5)

    kept = basis[:, :3]
    assert torch.allclose(kept.mT @ kept, torch.eye(3), atol=1e-5)


def test_lemma_features_recovers_shared_subspace():
    # Each lemma's tokens lie along one axis of a shared 3-dim subspace;
    # combining the per-lemma directions must recover the whole subspace.
    shared = _orthonormal_basis(6, 3, seed=20)  # columns e1, e2, e3
    lemmas = []
    for i in range(3):
        g = torch.Generator().manual_seed(30 + i)
        coeffs = torch.randn(8, 1, generator=g)  # 8 tokens along axis i
        lemmas.append(coeffs @ shared[:, i : i + 1].mT)  # (8, 6)

    basis = subspace_from_lemma_features(lemmas, rank=3, r_lemma=1)
    concept = ConceptSubspace(synset="x.n.01", basis=basis)

    g = torch.Generator().manual_seed(40)
    features = torch.randn(20, 3, generator=g) @ shared.mT  # live in span(shared)
    ef = concept.energy_fraction(features)
    assert torch.allclose(ef, torch.ones(20), atol=1e-3)


def test_lemma_balancing_vs_pooled():
    # The defining behavioural difference. Lemma A: many tokens spread over
    # directions a1 (dominant) and a2. Lemma B: a single token on b.
    dirs = _orthonormal_basis(6, 3, seed=21)  # columns a1, a2, b
    a1, a2, b = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    g = torch.Generator().manual_seed(22)
    ca = torch.randn(50, 2, generator=g)
    ca[:, 0] *= 3.0  # a1 carries more variance than a2 -> A's top dir is a1
    lemma_a = ca @ torch.stack([a1, a2])  # (50, 6)
    lemma_b = b.unsqueeze(0)  # (1, 6)

    pooled = ConceptSubspace(
        synset="p",
        basis=subspace_from_features(torch.cat([lemma_a, lemma_b]), rank=2),
    )
    hier = ConceptSubspace(
        synset="h",
        basis=subspace_from_lemma_features([lemma_a, lemma_b], rank=2, r_lemma=1),
    )

    # Pooled spends its rank-2 budget on A's internal spread {a1, a2}; the
    # lone B token is dropped. Hierarchical gives B a guaranteed slot {a1, b}.
    assert pooled.energy_fraction(b).item() < 0.1
    assert pooled.energy_fraction(a2).item() > 0.9
    assert hier.energy_fraction(b).item() > 0.9
    assert hier.energy_fraction(a2).item() < 0.1


def test_lemma_features_r_lemma_gt_one_keeps_more_directions():
    # One lemma spanning a 2-dim plane; r_lemma=2 keeps both of its directions.
    plane = _orthonormal_basis(6, 2, seed=23)
    g = torch.Generator().manual_seed(24)
    lemma = torch.randn(12, 2, generator=g) @ plane.mT  # (12, 6) in span(plane)

    basis = subspace_from_lemma_features([lemma], rank=4, r_lemma=2)
    concept = ConceptSubspace(synset="q.n.01", basis=basis)

    assert basis.shape == (6, 4)
    assert (basis.norm(dim=0)[:2] > 0.5).all()
    assert torch.allclose(basis.norm(dim=0)[2:], torch.zeros(2), atol=1e-5)
    # Both plane directions are captured.
    ef = concept.energy_fraction(plane.mT)  # the two basis rows of the plane
    assert torch.allclose(ef, torch.ones(2), atol=1e-3)


def test_lemma_token_rows_splits_per_lemma_and_drops_pad():
    # A synset's stacked (n_lemmas, max_tokens) array, pad_id=0, with a trailing
    # all-pad row: must yield one id row per real lemma, pad stripped.
    stacked = np.array([[5, 6, 0], [7, 0, 0], [0, 0, 0]])
    rows = lemma_token_rows(stacked, pad_id=0)
    assert [r.tolist() for r in rows] == [[5, 6], [7]]  # all-pad row dropped


def test_lemma_token_rows_regression_not_one_block():
    # The bug: treating the whole (m, t) array as a single block collapses the
    # concept to rank 1. The split must instead give m per-lemma blocks, so a
    # hierarchical build over them spans m directions (here 3, capped by rank).
    stacked = np.array([[11, 12], [13, 14], [15, 16]])  # 3 lemmas, no pad
    rows = lemma_token_rows(stacked, pad_id=0)
    assert len(rows) == 3
    d = 6
    blocks = [torch.randn(len(r), d, generator=torch.Generator().manual_seed(i))
              for i, r in enumerate(rows)]
    basis = subspace_from_lemma_features(blocks, rank=4, r_lemma=1)
    assert int((basis.norm(dim=0) > 0.5).sum()) == 3  # 3 lemma dirs, not 1


def test_lemma_features_empty_blocks_return_zero_basis():
    # All-empty lemma blocks (e.g. only pad tokens) -> a zero basis, no crash.
    empty = [torch.zeros(0, 6), torch.zeros(0, 6)]
    basis = subspace_from_lemma_features(empty, rank=3, r_lemma=1)
    assert basis.shape == (6, 3)
    assert torch.allclose(basis, torch.zeros(6, 3))


# -- blend composition (Family 1) ---------------------------------------------


def test_blend_bases_union_of_orthogonal_subspaces():
    # Two disjoint 1-dim subspaces blend into the 2-dim subspace they span.
    axes = _orthonormal_basis(6, 2, seed=50)
    b1 = torch.zeros(6, 2)
    b1[:, 0] = axes[:, 0]  # rank-1 basis (one zero-padded column) along axis 0
    b2 = torch.zeros(6, 2)
    b2[:, 0] = axes[:, 1]  # rank-1 basis along axis 1

    blended = blend_bases([b1, b2], rank=2)
    assert blended.shape == (6, 2)
    gram = blended.mT @ blended
    assert torch.allclose(gram, torch.eye(2), atol=1e-5)

    # The blend spans exactly {axis0, axis1}: both are fully captured, and a
    # third orthogonal direction is not.
    concept = ConceptSubspace(synset=["a.n.01", "b.n.01"], basis=blended)
    assert torch.allclose(
        concept.energy_fraction(axes.mT), torch.ones(2), atol=1e-4
    )
    third = _orthonormal_basis(6, 3, seed=51)[:, 2]
    outside = third - axes @ (axes.mT @ third)  # orthogonal to both axes
    assert concept.energy_fraction(outside) < 1e-3


def test_blend_bases_drops_zero_padding_and_dedupes():
    # Two identical rank-1 bases (with zero padding) blend to a single direction;
    # fixed rank 3 -> two trailing zero columns.
    axis = _orthonormal_basis(5, 1, seed=52)
    b = torch.zeros(5, 3)
    b[:, 0] = axis[:, 0]

    blended = blend_bases([b, b.clone()], rank=3)
    assert blended.shape == (5, 3)
    col_norms = blended.norm(dim=0)
    assert col_norms[0] > 0.5
    assert torch.allclose(col_norms[1:], torch.zeros(2), atol=1e-5)


def test_concept_blend_classmethod_combines_synsets():
    axes = _orthonormal_basis(6, 2, seed=53)
    c1 = ConceptSubspace(synset="dog.n.01", basis=axes[:, :1])
    c2 = ConceptSubspace(synset="war.n.01", basis=axes[:, 1:])

    # rank=2 keeps both concepts' directions; synsets concatenate.
    blended = ConceptSubspace.blend([c1, c2], rank=2)
    assert isinstance(blended, ConceptSubspace)
    assert blended.synset == ["dog.n.01", "war.n.01"]
    assert blended.rank == 2
    assert torch.allclose(
        blended.energy_fraction(axes.mT), torch.ones(2), atol=1e-4
    )


def test_concept_blend_default_rank_is_max_input_rank():
    # Fixed-rank policy: the blend keeps the shared r (here max input rank = 2).
    axes = _orthonormal_basis(6, 4, seed=54)
    c1 = ConceptSubspace(synset="a", basis=axes[:, :2])
    c2 = ConceptSubspace(synset="b", basis=axes[:, 2:])
    assert ConceptSubspace.blend([c1, c2]).rank == 2


# -- fluency-blended candidate selection (lambda dial) ------------------------


def test_fluent_selection_reduces_to_alignment_when_lambda_zero():
    # lambda = 0 -> identical winner to the plain cumulative-alignment select.
    scores = torch.tensor([[0.1, 0.1], [0.3, 0.4]])  # n=1, k=2, T=2
    cum_logprob = torch.tensor([-0.1, -5.0])  # candidate 1 is far less fluent
    col_idx, probe = select_best_candidate_fluent(
        scores, cum_logprob, fluency_weight=0.0, n=1, k=2
    )
    base_idx, base_probe = select_best_candidate(scores, n=1, k=2)
    assert col_idx.tolist() == base_idx.tolist() == [1]
    assert torch.allclose(probe, base_probe, atol=1e-6)


def test_fluent_selection_prefers_fluent_candidate_under_lambda():
    # Alignment alone picks candidate 1 (0.7 > 0.2); a large lambda flips the
    # choice to the far more fluent candidate 0.
    scores = torch.tensor([[0.1, 0.1], [0.3, 0.4]])  # cum last: c0=0.2, c1=0.7
    cum_logprob = torch.tensor([-0.1, -5.0])
    # blended: c0 = 0.2 + 1.0*(-0.1) = 0.1 ; c1 = 0.7 + 1.0*(-5.0) = -4.3
    col_idx, probe = select_best_candidate_fluent(
        scores, cum_logprob, fluency_weight=1.0, n=1, k=2
    )
    assert col_idx.tolist() == [0]
    # Probe stays the CONCEPT trajectory of the winner (not the blended value).
    assert torch.isclose(probe[0, -1], torch.tensor(0.2), atol=1e-6)


def test_fluent_selection_batched():
    scores = torch.tensor(
        [
            [0.5, 0.5],  # in0 c0 -> cum 1.0
            [0.1, 0.1],  # in0 c1 -> cum 0.2
            [0.0, 0.0],  # in1 c0 -> cum 0.0
            [0.2, 0.9],  # in1 c1 -> cum 1.1
        ]
    )
    # Make in1's fluent choice flip to c0 with a strong logprob penalty on c1.
    cum_logprob = torch.tensor([0.0, 0.0, 0.0, -10.0])
    col_idx, _ = select_best_candidate_fluent(
        scores, cum_logprob, fluency_weight=1.0, n=2, k=2
    )
    assert col_idx.tolist() == [0, 0]
