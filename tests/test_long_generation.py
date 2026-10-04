import math
from pathlib import Path

import pytest

from frames.experiments.long_generation import (
    LongGenerationConfigError,
    generation_metadata,
    load_long_generation_experiment,
    matched_segment_fluency,
    repetition_metrics,
    token_bins,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "long-generation-inputs.yaml"


def test_repository_experiment_53_selection_and_budget():
    experiment = load_long_generation_experiment(CONFIG, "experiment_53")

    assert experiment.steps == 200
    assert experiment.lambdas == (0.0, 0.4)
    assert len(experiment.prompts) == 5
    assert len(experiment.composition_pairs) == 15
    assert not experiment.negative_directions
    assert experiment.guided_run_count(method_count=3) == 450
    assert experiment.output_token_budget(method_count=3) == 90_000


def test_repository_experiment_54_all_uses_catalog_order():
    experiment = load_long_generation_experiment(CONFIG, "experiment_54")

    assert experiment.steps == 24
    assert [prompt["id"] for prompt in experiment.prompts] == [
        "men",
        "women",
        "short_story",
        "favorite_place",
        "weekends",
        "good_life",
        "most_important",
        "weather",
        "dinner",
        "happiness",
    ]
    assert [direction["id"] for direction in experiment.negative_directions] == [
        "woman_minus_man",
        "man_minus_woman",
    ]
    assert experiment.guided_run_count(method_count=3) == 60


def test_loader_rejects_unknown_and_duplicate_selection_ids(tmp_path):
    source = CONFIG.read_text()
    unknown = source.replace("      - short_story", "      - not_a_prompt", 1)
    path = tmp_path / "unknown.yaml"
    path.write_text(unknown)
    with pytest.raises(LongGenerationConfigError, match="unknown prompt"):
        load_long_generation_experiment(path, "experiment_53")

    duplicate = source.replace("      - favorite_place", "      - short_story", 1)
    path = tmp_path / "duplicate.yaml"
    path.write_text(duplicate)
    with pytest.raises(LongGenerationConfigError, match="duplicate prompt"):
        load_long_generation_experiment(path, "experiment_53")


def test_token_bins_preserve_order_and_short_final_bin():
    bins = token_bins(range(45), width=20)

    assert [(part.index, part.start, part.stop) for part in bins] == [
        (0, 0, 20),
        (1, 20, 40),
        (2, 40, 45),
    ]
    assert bins[-1].token_ids == (40, 41, 42, 43, 44)


def test_matched_segment_fluency_uses_shared_prefix_conditioned_losses():
    result = matched_segment_fluency(
        steered_token_nll=[1.0, 2.0, 3.0, 4.0],
        baseline_token_nll=[0.5, 1.0, 1.5, 2.0, 2.5],
        start=1,
        stop=5,
        min_shared_tokens=3,
    )

    assert result["shared_tokens"] == 3
    assert result["segment_mean_nll"] == pytest.approx(3.0)
    assert result["baseline_segment_mean_nll"] == pytest.approx(1.5)
    assert result["segment_log_ppl_ratio"] == pytest.approx(1.5)
    assert result["segment_ppl_ratio"] == pytest.approx(math.exp(1.5))


def test_matched_segment_fluency_masks_short_shared_tail():
    result = matched_segment_fluency(
        [1.0, 2.0, 3.0],
        [0.5, 1.0, 1.5],
        start=1,
        stop=3,
        min_shared_tokens=3,
    )

    assert result["shared_tokens"] == 2
    assert math.isnan(result["segment_ppl_ratio"])
    assert math.isnan(result["segment_log_ppl_ratio"])


def test_repetition_metrics_cover_empty_unique_and_repeated_tokens():
    assert repetition_metrics([]) == {
        "token_count": 0,
        "distinct_token_ratio": 0.0,
        "repeated_ngram_rate": 0.0,
        "max_repeated_span_tokens": 0,
    }

    unique = repetition_metrics([1, 2, 3, 4, 5])
    assert unique["distinct_token_ratio"] == 1.0
    assert unique["repeated_ngram_rate"] == 0.0
    assert unique["max_repeated_span_tokens"] == 0

    repeated = repetition_metrics([1, 2, 3, 4, 1, 2, 3, 4])
    assert repeated["distinct_token_ratio"] == 0.5
    assert repeated["repeated_ngram_rate"] == pytest.approx(0.2)
    assert repeated["max_repeated_span_tokens"] == 4


def test_generation_metadata_makes_historical_step_offset_explicit():
    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [1, 2, 99] if text == "early" else [1, 2, 3]}

    capped = generation_metadata(Tokenizer(), "capped", 2, 0.25)
    assert capped["token_count"] == 3
    assert capped["historical_non_eos_token_cap"] == 3
    assert capped["termination"] == "step_cap"
    assert capped["ended_early"] is False

    early = generation_metadata(Tokenizer(), "early", 4, 0.5)
    assert early["termination"] == "eos"
    assert early["ended_early"] is True
