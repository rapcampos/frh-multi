"""CPU contracts for NB69's lemma-count-matched custom source."""

import pytest

from frames.experiments.steering_comparison import matrix_size
from frames.experiments.topc_comparison import (
    EXCLUDED_PROMPTS,
    NEW_SOURCE,
    make_experiment,
    top_c_token_rows,
)


class Tokenizer:
    bos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [2] * (5 if text.strip() == "long" else 2)}


def test_top_c_keeps_first_c_words_that_fit():
    rows, kept, skipped = top_c_token_rows(
        Tokenizer(), ["a", "long", "b", "c"], 2, max_tokens=4
    )
    assert kept == ["a", "b"]
    assert skipped == ["long"]
    assert len(rows) == 8  # four surface forms per kept word


def test_top_c_rejects_too_few_words():
    with pytest.raises(ValueError, match="Only 1"):
        top_c_token_rows(Tokenizer(), ["a", "long"], 2, max_tokens=4)


def test_manifest_reuses_nb68_matrix_for_the_new_source_only():
    manifest = make_experiment()
    assert manifest["sources"] == [NEW_SOURCE]
    assert len(manifest["configs"]) == 74
    assert set(EXCLUDED_PROMPTS) <= {p["id"] for p in manifest["prompts"]}
    assert matrix_size(manifest) == 74 * 8 * 2 * 2
    smoke = make_experiment(smoke=True)
    assert matrix_size(smoke) < matrix_size(manifest)
    assert {c["unit_count"] for c in smoke["configs"]} == {1, 2, 3, 4}
