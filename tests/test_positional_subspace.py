"""CPU mathematical and integration contracts for positional concept subspaces."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from subspace import ConceptSubspace, PositionalConceptSubspace, positional_alignment
from subspace.concept import subspace_from_features
from subspace.positional import positional_token_ids
from subspace.representation import SubspaceUnembeddingRepresentation


def concept(bases=None):
    if bases is None:
        bases = torch.eye(3).T.unsqueeze(-1)
    return PositionalConceptSubspace(
        synset="test", bases=bases, sample_counts=[2] * len(bases)
    )


def test_extract_preserves_positions_duplicates_and_excludes_long_words():
    rows = np.array([[1, 2, 0, 0], [1, 0, 3, 0], [4, 5, 6, 7], [0, 0, 0, 0]])
    columns = positional_token_ids([rows], 3, 0)
    assert [c.tolist() for c in columns] == [[1, 1], [2], [3]]
    assert [c.tolist() for c in positional_token_ids([[[1, 2]]], 3, None)] == [
        [1],
        [2],
        [],
    ]
    with pytest.raises(ValueError, match="positive"):
        positional_token_ids([rows], 0, 0)


def test_construction_rank_threshold_empty_and_zero_positions():
    c = PositionalConceptSubspace.from_features(
        "x",
        [
            torch.diag(torch.tensor([3.0, 1.0, 1e-5])),
            torch.empty(0, 3),
            torch.zeros(2, 3),
        ],
        rank=4,
    )
    assert c.bases.shape == (3, 3, 4)
    assert c.effective_ranks == [2, 0, 0]
    assert c.sample_counts == [3, 0, 2]
    assert torch.allclose(c.bases[0, :, :2].T @ c.bases[0, :, :2], torch.eye(2))
    assert torch.count_nonzero(c.bases[0, :, 2:]) == 0
    with pytest.raises(ValueError, match="active"):
        PositionalConceptSubspace.from_features("x", [torch.zeros(2, 3)])


@pytest.mark.parametrize("normalize", [False, True])
def test_t1_matches_pooled_construction_and_score(normalize):
    features = torch.randn(8, 4, generator=torch.Generator().manual_seed(1))
    c = PositionalConceptSubspace.from_features("x", [features], rank=2)
    assert torch.equal(c.bases[0], subspace_from_features(features, 2))
    hidden = torch.randn(2, 7, 4)
    expected = (hidden @ c.bases[0]).norm(dim=-1)
    if normalize:
        expected /= hidden.norm(dim=-1)
    assert torch.allclose(c.project(hidden, normalize), expected)


def test_window_orientation_fixed_denominator_and_batched_shape():
    c = concept()
    hidden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = torch.tensor([(1 + 5) / 3, 4 / 3])
    assert torch.allclose(c.project(hidden), expected)
    assert c.project(hidden.expand(2, 4, -1, -1)).shape == (2, 4, 2)
    empty_slot = concept(
        torch.stack([c.bases[0], torch.zeros_like(c.bases[0]), c.bases[2]])
    )
    assert torch.allclose(empty_slot.project(hidden), torch.tensor([0.5, 2.0]))
    assert c.project(torch.empty(0, 3)).shape == (0,)


def test_basis_gauge_invariance_and_position_sensitivity():
    torch.manual_seed(2)
    bases = torch.stack([torch.linalg.qr(torch.randn(6, 2)).Q for _ in range(3)])
    rotations = torch.stack([torch.linalg.qr(torch.randn(2, 2)).Q for _ in range(3)])
    h = torch.randn(2, 9, 6)
    assert torch.allclose(
        concept(bases).project(h), concept(bases @ rotations).project(h), atol=1e-6
    )
    c = concept()
    assert not torch.allclose(
        c.project(torch.eye(3)), concept(c.bases.flip(0)).project(torch.eye(3))
    )


def test_cumulative_interior_cancellation_and_boundary_effect():
    c = concept()
    reversed_c = concept(c.bases.flip(0))
    # Once the first T-1 states are zero, every nonzero state occurs in all slots.
    h = torch.randn(10, 3, generator=torch.Generator().manual_seed(3))
    h[:2] = 0
    assert torch.allclose(c.project(h).sum(), reversed_c.project(h).sum(), atol=1e-6)
    h[0] = torch.tensor([1.0, 0.0, 0.0])
    assert not torch.allclose(c.project(h).sum(), reversed_c.project(h).sum())


def test_normalized_bounds_scale_invariance_zero_and_expression_masks():
    c = concept()
    h = torch.randn(2, 6, 3)
    scores = c.project(h, normalize=True)
    assert ((scores >= 0) & (scores <= 1 + 1e-6)).all()
    assert torch.allclose(scores, c.project(h * 4, normalize=True), atol=1e-6)
    assert torch.equal(c.project(torch.zeros_like(h), True), torch.zeros(2, 6))
    # Three real pairs: (h0,S0), (h1,S1), (h1,S0); energies 1, 1, 0.
    assert torch.allclose(c.expression(torch.eye(3)[:2]), torch.tensor(2 / 3))
    padded = torch.tensor([[99.0, 99.0, 99.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(
        c.expression(padded, torch.tensor([False, True, True])), torch.tensor(2 / 3)
    )
    assert c.expression(padded, torch.zeros(3, dtype=torch.bool)) == 0
    assert c.expression(torch.zeros(2, 3)) == 0
    with pytest.raises(ValueError, match="mask"):
        c.expression(padded, torch.ones(2, dtype=torch.bool))


def test_blend_preserves_position_and_metadata_and_rejects_mixed_guides():
    a = concept(torch.tensor([[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]]]))
    b = concept(torch.tensor([[[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]]]))
    blended = PositionalConceptSubspace.blend([a, b], rank=2)
    assert blended.effective_ranks == [2, 2]
    assert blended.sample_counts == [4, 4]
    assert len(blended.metadata["constituents"]) == 2
    expected = torch.stack(
        [
            torch.diag(torch.tensor([1.0, 1.0, 0.0])),
            torch.diag(torch.tensor([0.0, 1.0, 1.0])),
        ]
    )
    assert torch.allclose(blended.bases @ blended.bases.mT, expected)
    assert PositionalConceptSubspace.blend([a, blended]).rank == 2
    pooled = ConceptSubspace(synset="x", basis=torch.eye(3))
    for inputs in ([a, pooled], [a, concept()]):
        with pytest.raises(ValueError):
            PositionalConceptSubspace.blend(inputs)
    with pytest.raises(ValueError):
        ConceptSubspace.blend([pooled, a])


def test_signed_scores_use_separate_active_denominators():
    a = concept()
    b = concept(torch.stack([a.bases[1], torch.zeros_like(a.bases[1]), a.bases[0]]))
    h = torch.randn(2, 4, 3)
    assert torch.allclose(positional_alignment(h, a, b), a.project(h) - b.project(h))
    scores = positional_alignment(h, a, b, True)
    assert (scores.abs() <= 1 + 1e-6).all()
    for bad in (concept(a.bases[:2]), ConceptSubspace(synset="x", basis=torch.eye(3))):
        with pytest.raises(ValueError):
            positional_alignment(h, a, bad)


class FakeFeatures:
    table = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    def get_token_representations(self, ids):
        return self.table[ids]


class FakeData:
    language_codes = SimpleNamespace(name="Default")
    calls = 0

    def get_all_synsets(self, *args):
        self.calls += 1
        return pd.DataFrame(
            {"synset": ["test"], "tokens": [np.array([[1, 2, 0], [1, 0, 3]])]}
        )


def representation():
    return SubspaceUnembeddingRepresentation.model_construct(
        model=SimpleNamespace(
            id="mock/model", device="cpu", tokenizer=SimpleNamespace(pad_token_id=0)
        ),
        unembedding=FakeFeatures(),
        data=FakeData(),
        rank=2,
    )


def test_dataframe_construction_requires_width_and_preserves_columns():
    r = representation()
    df = r.data.get_all_synsets()
    with pytest.raises(ValueError, match="max_token_count"):
        r.compute_concept(df, method="positional")
    c = r.compute_concept(df, method="positional", max_token_count=3)
    assert c.sample_counts == [2, 1, 1]
    assert c.effective_ranks == [1, 1, 1]
    assert torch.allclose(c.bases[:, :, 0].abs(), torch.eye(3))
    with pytest.raises(ValueError, match="empty"):
        r.compute_concept(df.iloc[:0], method="positional", max_token_count=3)


def test_cache_roundtrip_separation_and_version_validation(tmp_path, monkeypatch):
    monkeypatch.setattr("subspace.representation.SUBSPACE_CACHE_DIR", tmp_path)
    r = representation()
    a = r.get_concept_cached("test", 1, 3, method="positional")
    calls = r.data.calls
    b = r.get_concept_cached("test", 1, 3, method="positional")
    assert r.data.calls == calls
    assert torch.equal(a.bases, b.bases)
    assert a.construction_metadata() == b.construction_metadata()
    paths = {
        r._concept_cache_path("test", 1, 3, method=m)
        for m in ("pooled", "hierarchical", "positional")
    }
    assert len(paths) == 3
    # Existing tensor-only caches remain readable.
    pooled = r.get_concept_cached("test", 1, 3)
    assert torch.equal(pooled.basis, r.get_concept_cached("test", 1, 3).basis)
    path = r._concept_cache_path("test", 1, 3, method="positional")
    torch.save({"version": 999}, path)
    with pytest.raises(ValueError, match="version"):
        r.get_concept_cached("test", 1, 3, method="positional")


class GenerationModel:
    device = "cpu"
    tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0)

    def make_input(self, text, padding=True):
        assert text == ["prompt"] and padding is True
        return {"input_ids": torch.tensor([[1]])}

    def decode(self, tokens):
        return [str(row.tolist()) for row in tokens]


class MockGeneration(SubspaceUnembeddingRepresentation):
    def _candidates(self, tokens, k):
        if len(tokens) == 1:
            return (
                torch.tensor([[[1, 10], [1, 11]]]),
                torch.empty(0),
                torch.tensor([[-5.0, -0.1]]),
            )
        sequences = torch.cat(
            [
                tokens[:, None, :].expand(-1, k, -1),
                torch.zeros(len(tokens), k, 1, dtype=torch.long),
            ],
            -1,
        )
        hidden = torch.tensor([[[0.8, 0.0], [0.8, 0.0]], [[1.0, 0.95], [1.0, 0.95]]])
        return sequences, hidden, torch.zeros(len(tokens), k)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_mock_generation_scores_and_concept_only_probes(normalize, lam):
    r = MockGeneration.model_construct(
        model=GenerationModel(), unembedding=None, data=None
    )
    a = concept(torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]]))
    b = concept(torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]]))
    texts, probe = r.generate_with_topk_guide(
        "prompt",
        a,
        k=2,
        steps=1,
        repel_guide=b,
        normalize_score=normalize,
        fluency_weight=lam,
    )
    h = r._candidates(torch.tensor([[1, 10], [1, 11]]), 2)[1]
    cumulative = positional_alignment(h, a, b, normalize).cumsum(-1)
    winner = (cumulative[:, -1] + lam * torch.tensor([-5.0, -0.1])).argmax()
    assert torch.allclose(probe, cumulative[winner][None])
    assert str(10 + winner.item()) in texts[0]
    with pytest.raises(ValueError, match="mix"):
        r.generate_with_topk_guide(
            "prompt", a, repel_guide=ConceptSubspace(synset="x", basis=torch.eye(2))
        )


class ProjectionModel:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def tokenize(self, text, padding=True):
        assert text and padding is True
        return torch.tensor([[0, 1, 2], [1, 2, 0]])

    def forward_last_hiden_state(self, tokens):
        return torch.tensor([[99.0, 99.0, 99.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])[
            tokens
        ]


def test_representation_project_and_expression_mask_tokenizer_padding():
    r = SubspaceUnembeddingRepresentation.model_construct(
        model=ProjectionModel(), unembedding=None, data=None
    )
    c = concept()
    tokens = r.model.tokenize("text")
    hidden = r.model.forward_last_hiden_state(tokens)
    assert torch.equal(r.project(["left", "right"], c), c.project(hidden))
    assert torch.allclose(r.expression(["left", "right"], c), torch.full((2,), 2 / 3))
    assert torch.equal(r.expression(tokens, c), r.expression(["left", "right"], c))
    assert c.expression(torch.empty(0, 3)) == 0


def test_validation_rejects_inconsistent_metadata_and_dimensions():
    c = concept()
    with pytest.raises(ValueError, match="effective_ranks"):
        PositionalConceptSubspace(
            synset="x",
            bases=c.bases,
            sample_counts=[1, 1, 1],
            effective_ranks=[2, 2, 2],
        )
    with pytest.raises(ValueError, match="sample_counts"):
        PositionalConceptSubspace(synset="x", bases=c.bases, sample_counts=[1])
    with pytest.raises(ValueError, match="hidden states"):
        c.project(torch.zeros(3, 2))
    with pytest.raises(ValueError, match="dimension"):
        positional_alignment(torch.zeros(3, 3), c, concept(torch.ones(3, 2, 1)))
    r = representation()
    with pytest.raises(ValueError, match="unknown method"):
        r.get_concept_cached("test", 1, 3, method="typo")
