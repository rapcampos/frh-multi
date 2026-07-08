"""CPU unit tests for Family 1 composition: Concept.average and joint_subspace."""

import torch

from frames.linalg.orthogonalization import solve_procrustes
from frames.representations.concept import Concept


def basis_concept(name: str, cols: list[int], d: int = 16) -> Concept:
    """Orthonormal concept whose vectors are standard basis vectors e_col."""
    tensor = torch.zeros(1, d, len(cols))
    for i, col in enumerate(cols):
        tensor[0, col, i] = 1.0
    return Concept(synset=name, tensor=tensor)


def random_orthonormal_concept(name: str, d: int = 16, k: int = 3) -> Concept:
    q, _ = torch.linalg.qr(torch.randn(d, k))
    return Concept(synset=name, tensor=q.unsqueeze(0))


def columns_orthonormal(tensor: torch.Tensor, k_eff: int) -> bool:
    cols = tensor[0, :, :k_eff]
    gram = cols.mT @ cols
    return torch.allclose(gram, torch.eye(k_eff), atol=1e-5)


class TestAverage:
    def test_result_is_orthonormal(self):
        torch.manual_seed(0)
        a = random_orthonormal_concept("a", k=3)
        b = random_orthonormal_concept("b", k=3)
        mean = Concept.average([a, b])
        assert columns_orthonormal(mean.tensor, 3)

    def test_weight_one_zero_recovers_first_concept(self):
        torch.manual_seed(1)
        a = random_orthonormal_concept("a", k=3)
        b = random_orthonormal_concept("b", k=3)
        mean = Concept.average([a, b], weights=[1.0, 0.0])
        assert torch.allclose(mean.tensor, a.tensor, atol=1e-5)

    def test_unequal_ranks_pad_without_garbage(self):
        # with w=[1,0] and rank(a)=2 < rank(b)=3, the summed frame has a
        # trailing zero column; the polar factor there is arbitrary, so it
        # must be excluded and restored as zeros — not silently filled
        torch.manual_seed(2)
        a = random_orthonormal_concept("a", k=2)
        b = random_orthonormal_concept("b", k=3)
        mean = Concept.average([a, b], weights=[1.0, 0.0])
        assert mean.tensor.shape == (1, 16, 3)
        assert torch.allclose(mean.tensor[..., :2], a.tensor, atol=1e-5)
        assert torch.equal(mean.tensor[..., 2], torch.zeros(1, 16))

    def test_uniform_weights_match_legacy_behavior(self):
        # pre-Step-5 average was solve_procrustes(sum of same-shape tensors)
        torch.manual_seed(3)
        a = random_orthonormal_concept("a", k=3)
        b = random_orthonormal_concept("b", k=3)
        mean = Concept.average([a, b])
        legacy = solve_procrustes(a.tensor + b.tensor)
        assert torch.allclose(mean.tensor, legacy, atol=1e-6)

    def test_scale_invariance_of_weights(self):
        torch.manual_seed(4)
        a = random_orthonormal_concept("a", k=3)
        b = random_orthonormal_concept("b", k=3)
        mean_1 = Concept.average([a, b], weights=[0.7, 0.3])
        mean_2 = Concept.average([a, b], weights=[7.0, 3.0])
        assert torch.allclose(mean_1.tensor, mean_2.tensor, atol=1e-5)

    def test_synset_name_joined(self):
        a = basis_concept("a", [0])
        b = basis_concept("b", [1])
        assert Concept.average([a, b]).synset == "a | b"


class TestJointSubspace:
    def test_result_is_orthonormal(self):
        a = basis_concept("a", [0, 1])
        b = basis_concept("b", [2, 3])
        sub = Concept.joint_subspace([a, b])
        assert columns_orthonormal(sub.tensor, len(sub))

    def test_contains_each_constituent_span(self):
        torch.manual_seed(5)
        a = random_orthonormal_concept("a", k=2)
        b = random_orthonormal_concept("b", k=3)
        sub = Concept.joint_subspace([a, b])
        basis = sub.tensor[0]
        for concept in (a, b):
            vectors = concept.tensor[0]  # (d, k)
            projected = basis @ (basis.mT @ vectors)
            assert torch.allclose(projected, vectors, atol=1e-4)

    def test_disjoint_spans_give_full_rank(self):
        a = basis_concept("a", [0, 1])
        b = basis_concept("b", [2, 3])
        sub = Concept.joint_subspace([a, b])
        assert len(sub) == 4

    def test_duplicate_concept_is_rank_truncated(self):
        a = basis_concept("a", [0, 1])
        sub = Concept.joint_subspace([a, a])
        assert len(sub) == 2  # union span is 2-dimensional, not 4

    def test_partial_overlap_rank(self):
        a = basis_concept("a", [0, 1])
        b = basis_concept("b", [1, 2])
        sub = Concept.joint_subspace([a, b])
        assert len(sub) == 3  # union of spans is {e0, e1, e2}

    def test_synset_name_joined(self):
        a = basis_concept("a", [0])
        b = basis_concept("b", [1])
        assert Concept.joint_subspace([a, b]).synset == "a + b"
