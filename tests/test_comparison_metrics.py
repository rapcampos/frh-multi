"""CPU tests for the matched frame/subspace comparison metrics."""

import math

import pytest
import torch

from frames.experiments.comparison_metrics import (
    evaluate_continuation,
    ppl_metrics,
    summarize_cosines,
    token_cosine,
)


def test_token_cosine_ignores_trailing_zero_basis_columns():
    hidden = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    padded_e0 = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert token_cosine(hidden, padded_e0) == pytest.approx(0.3)


def test_token_cosine_rejects_empty_hidden_states():
    with pytest.raises(ValueError, match="at least one"):
        token_cosine(torch.empty(0, 2), torch.eye(2))


def test_summarize_cosines_orients_repulsion_as_a_reduction():
    per_concept = {
        "joy": {"steered": 0.8, "baseline": 0.5, "delta": 0.3},
        "sadness": {"steered": 0.2, "baseline": 0.6, "delta": -0.4},
    }
    summary = summarize_cosines(per_concept, positives=["joy"], negatives=["sadness"])

    assert summary == {
        "attraction_mean_delta": pytest.approx(0.3),
        "repulsion_mean_reduction": pytest.approx(0.4),
        "net_attract_minus_repel": pytest.approx(0.7),
        "signed_mean_delta": pytest.approx(0.35),
        "signed_min_delta": pytest.approx(0.3),
        "all_requested_improved": True,
        "requested_count": 2,
    }


def test_summarize_cosines_requires_the_same_constituents_in_every_view():
    with pytest.raises(ValueError, match="missing requested"):
        summarize_cosines({"joy": {"delta": 0.1}}, positives=["joy", "hope"])


def test_ppl_metrics_reports_absolute_delta_ratio_and_invalid_values():
    assert ppl_metrics(15.0, 10.0) == {
        "valid": True,
        "invalid_reason": None,
        "steered_ppl": 15.0,
        "baseline_ppl": 10.0,
        "delta_ppl": 5.0,
        "ppl_ratio": 1.5,
    }
    invalid = ppl_metrics(math.inf, 10.0)
    assert invalid["valid"] is False
    assert invalid["delta_ppl"] is None
    assert invalid["invalid_reason"] == "non_finite_ppl"
    assert ppl_metrics(torch.tensor(12.0), torch.tensor(10.0))["delta_ppl"] == 2.0


class _Tokenizer:
    all_special_ids = [0, 99]


class _Model:
    tokenizer = _Tokenizer()

    @staticmethod
    def tokenize(text):
        return {
            "steered": torch.tensor([[0, 1, 1, 99]]),
            "baseline": torch.tensor([[0, 2, 2, 99]]),
        }[text]

    @staticmethod
    def forward_last_hiden_state(tokens):
        if tokens[0, 1].item() == 1:
            # Special positions contain arbitrary values and must not count.
            return torch.tensor([[[9.0, 9.0], [1.0, 0.0], [1.0, 0.0], [9.0, 9.0]]])
        return torch.tensor([[[9.0, 9.0], [0.0, 1.0], [0.0, 1.0], [9.0, 9.0]]])


class _Fur:
    model = _Model()


def test_evaluate_continuation_uses_all_fixed_views_and_excludes_special_tokens():
    refs = {
        "wordnet_frame": {"joy": torch.tensor([[1.0], [0.0]])},
        "wordnet_subspace": {"joy": torch.tensor([[1.0], [0.0]])},
        "custom_frame": {"joy": torch.tensor([[1.0], [0.0]])},
        "custom_subspace": {"joy": torch.tensor([[1.0], [0.0]])},
    }
    result = evaluate_continuation(
        _Fur(), "steered", "baseline", refs, positive=["joy"]
    )

    assert set(result) == set(refs)
    for view in result.values():
        assert view["valid"] is True
        assert view["per_concept"]["joy"] == {
            "steered": pytest.approx(1.0),
            "baseline": pytest.approx(0.0),
            "delta": pytest.approx(1.0),
        }
        assert view["summary"]["all_requested_improved"] is True


def test_evaluate_continuation_marks_empty_text_invalid_without_model_call():
    refs = {"wordnet_frame": {"joy": torch.tensor([[1.0], [0.0]])}}
    result = evaluate_continuation(_Fur(), "", "baseline", refs, positive=["joy"])
    assert result["wordnet_frame"]["valid"] is False
    assert result["wordnet_frame"]["invalid_reason"] == "empty_continuation"
    assert result["wordnet_frame"]["per_concept"] == {}


def test_evaluate_continuation_reuses_a_caller_owned_hidden_cache():
    refs = {"wordnet_frame": {"joy": torch.tensor([[1.0], [0.0]])}}
    cache = {}
    evaluate_continuation(_Fur(), "steered", "baseline", refs, hidden_cache=cache)
    first = cache["baseline"]
    evaluate_continuation(_Fur(), "steered", "baseline", refs, hidden_cache=cache)
    assert cache["baseline"] is first
