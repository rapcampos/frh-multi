"""CPU tests for signed subspace score-space steering."""

import torch

from subspace.concept import ConceptSubspace
from subspace.generation import (
    select_best_candidate,
    select_best_candidate_fluent,
    subspace_alignment,
)
from subspace.representation import SubspaceUnembeddingRepresentation


def test_repulsion_subtracts_projection_and_default_is_unchanged():
    # Candidate 1 has more attractive alignment, but nearly all of it is also
    # repelled.  Signed selection must therefore pick candidate 0.
    attract = torch.tensor([[1.0], [0.0]])
    repel = torch.tensor([[0.0], [1.0]])
    hidden = torch.tensor(
        [
            [[0.8, 0.0], [0.8, 0.0]],
            [[1.0, 0.95], [1.0, 0.95]],
        ]
    )

    unsigned = subspace_alignment(hidden, attract)
    signed = subspace_alignment(hidden, attract, repel)

    assert torch.allclose(unsigned, torch.tensor([[0.8, 0.8], [1.0, 1.0]]))
    assert torch.allclose(signed, torch.tensor([[0.8, 0.8], [0.05, 0.05]]))
    assert select_best_candidate(unsigned, n=1, k=2)[0].tolist() == [1]
    assert select_best_candidate(signed, n=1, k=2)[0].tolist() == [0]


def test_signed_score_still_uses_the_existing_fluency_selection_rule():
    attract = torch.tensor([[1.0], [0.0]])
    repel = torch.tensor([[0.0], [1.0]])
    hidden = torch.tensor(
        [
            [[0.8, 0.0], [0.8, 0.0]],
            [[1.0, 0.95], [1.0, 0.95]],
        ]
    )
    signed = subspace_alignment(hidden, attract, repel)
    # lambda = 0 retains the signed-alignment winner.  At lambda = 1, the
    # second candidate's much higher log probability wins; the probe remains
    # its signed concept trajectory.
    logprob = torch.tensor([-5.0, -0.1])
    index, probe = select_best_candidate_fluent(signed, logprob, 0.0, n=1, k=2)
    assert index.tolist() == [0]
    assert torch.allclose(probe, torch.tensor([[0.8, 1.6]]))

    index, probe = select_best_candidate_fluent(signed, logprob, 1.0, n=1, k=2)
    assert index.tolist() == [1]
    assert torch.allclose(probe, torch.tensor([[0.05, 0.10]]))


class _GenerationTokenizer:
    eos_token_id = 99


class _GenerationModel:
    device = torch.device("cpu")
    tokenizer = _GenerationTokenizer()

    @staticmethod
    def make_input(text, padding=True):
        assert padding is True
        return {"input_ids": torch.tensor([[1]])}

    @staticmethod
    def decode(tokens):
        return ["generated"] * len(tokens)


class _MockSubspaceRepresentation(SubspaceUnembeddingRepresentation):
    """One deterministic guided step; avoids loading a Hugging Face model."""

    def _candidates(self, tokens, k):
        if tokens.size(0) == 1:
            return (
                torch.tensor([[[1, 10], [1, 11]]]),
                torch.empty(0),
                torch.tensor([[-5.0, -0.1]]),
            )
        sequences = torch.cat(
            [
                tokens.unsqueeze(1).expand(-1, k, -1),
                torch.zeros(len(tokens), k, 1, dtype=torch.long),
            ],
            dim=-1,
        )
        hidden = torch.tensor(
            [
                [[0.8, 0.0], [0.8, 0.0]],
                [[1.0, 0.95], [1.0, 0.95]],
            ]
        )
        return sequences, hidden, torch.zeros(len(tokens), k)


def test_generate_with_topk_guide_wires_repel_guide_into_signed_and_fluent_scores():
    representation = _MockSubspaceRepresentation.model_construct(
        model=_GenerationModel(),
        unembedding=None,
        data=None,
        rank=1,
        method="pooled",
        r_lemma=1,
    )
    attractive = ConceptSubspace(synset="attract", basis=torch.tensor([[1.0], [0.0]]))
    repelled = ConceptSubspace(synset="repel", basis=torch.tensor([[0.0], [1.0]]))

    _, unsigned_probe = representation.generate_with_topk_guide(
        ["prompt"], attractive, k=2, steps=1
    )
    _, signed_probe = representation.generate_with_topk_guide(
        ["prompt"], attractive, k=2, steps=1, repel_guide=repelled
    )
    _, fluent_probe = representation.generate_with_topk_guide(
        ["prompt"],
        attractive,
        k=2,
        steps=1,
        fluency_weight=1.0,
        repel_guide=repelled,
    )

    assert torch.allclose(unsigned_probe, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(signed_probe, torch.tensor([[0.8, 1.6]]))
    assert torch.allclose(fluent_probe, torch.tensor([[0.05, 0.10]]))
