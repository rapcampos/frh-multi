"""Input and battery checks for the matched steering comparison."""

import json
from collections import Counter
from pathlib import Path

import pytest

from frames.experiments.comparison_manifest import (
    CONCEPT_SYNSETS,
    SOURCE_FILES,
    _consensus_words,
    build_manifest,
    stable_hash,
)


def test_consensus_uses_original_lengths_first_occurrence_and_normalization():
    lists = {
        SOURCE_FILES[0]: [" Alpha ", "BETA", "alpha"],
        SOURCE_FILES[1]: ["beta", "alpha"],
        SOURCE_FILES[2]: ["gamma", "alpha"],
        SOURCE_FILES[3]: ["beta", "gamma"],
    }
    sources = {
        filename: {"test": {"words": words}} for filename, words in lists.items()
    }
    words, details = _consensus_words("test", sources)
    assert words == ["beta", "alpha", "gamma"]
    alpha = next(row for row in details if row["word"] == "alpha")
    assert alpha["score"] == pytest.approx(2.0)
    assert alpha["sources"][SOURCE_FILES[0]] == {
        "rank": 1,
        "list_length": 3,
        "score": 1.0,
    }


def test_consensus_ties_choose_shorter_then_alphabetical():
    lists = ["cccc", "bb", "aa", "ddd"]
    sources = {
        filename: {"test": {"words": [word]}}
        for filename, word in zip(SOURCE_FILES, lists)
    }
    words, _ = _consensus_words("test", sources)
    assert words == ["aa", "bb", "ddd", "cccc"]


def test_manifest_has_exact_source_provenance_and_battery():
    manifest = build_manifest()
    assert len(manifest["concepts"]) == 16
    assert {
        row["concept"]: row["synset"] for row in manifest["concepts"]
    } == CONCEPT_SYNSETS
    assert set(manifest["input_hashes"]) == set(SOURCE_FILES)
    assert all(len(row["words"]) == 100 for row in manifest["concepts"])
    assert all(len(set(row["words"])) == 100 for row in manifest["concepts"])
    assert all(
        row["provenance"][0]["word"] == row["words"][0] for row in manifest["concepts"]
    )

    configs = manifest["configs"]
    assert len(configs) == 74
    assert len({row["id"] for row in configs}) == 74
    assert len({row["experiment_id"] for row in configs}) == 60
    assert Counter(row["unit_count"] for row in configs) == {1: 32, 2: 14, 3: 14, 4: 14}
    for count in (1, 2, 3, 4):
        rows = [row for row in configs if row["unit_count"] == count]
        assert {
            name for row in rows for name in row["positive"] + row["negative"]
        } == set(CONCEPT_SYNSETS)
        assert all(row["atomic_count"] == count + bool(row["pair"]) for row in rows)
        assert all(len(row["negative"]) <= 1 for row in rows)

    pair_rows = [row for row in configs if row["pair"]]
    assert Counter(row["unit_count"] for row in pair_rows) == {1: 16, 2: 4, 3: 4, 4: 4}
    for experiment in {row["experiment_id"] for row in pair_rows}:
        forward, reverse = [
            row for row in pair_rows if row["experiment_id"] == experiment
        ]
        assert forward["positive"][0] == reverse["negative"][0]
        assert forward["negative"][0] == reverse["positive"][0]
        assert forward["positive"][1:] == reverse["positive"][1:]


def test_input_validation_rejects_incorrect_synset(tmp_path):
    for filename in SOURCE_FILES:
        entries = json.loads((Path("concepts") / filename).read_text())
        if filename == SOURCE_FILES[0]:
            entries[0]["synset"] = "joy.n.02"
        (tmp_path / filename).write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="unexpected synset"):
        build_manifest(tmp_path)


def test_stable_hash_is_independent_of_mapping_order():
    assert stable_hash({"a": 1, "b": [2, 3]}) == stable_hash({"b": [2, 3], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
