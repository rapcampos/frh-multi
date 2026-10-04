"""Frozen concepts and configurations for the frame/subspace comparison.

This module only handles experiment inputs. It deliberately has no model, WordNet,
or notebook dependency so the planned battery can be inspected before GPU work.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_FILES = (
    "gpt_astra_medium.json",
    "gpt_sol_medium.json",
    "claude_sonnet5_high.json",
    "claude_opus5_high.json",
)

# These are the requested concepts and their exact WordNet senses. The files must
# agree with this mapping; matching one another alone cannot catch a common typo.
CONCEPT_SYNSETS = {
    "joy": "joy.n.01",
    "sadness": "sadness.n.01",
    "hope": "hope.n.02",
    "despair": "despair.n.02",
    "affection": "affection.n.01",
    "hostility": "hostility.n.03",
    "trust": "trust.n.05",
    "suspicion": "suspicion.n.02",
    "calmness": "calmness.n.03",
    "agitation": "agitation.n.03",
    "courage": "courage.n.01",
    "fear": "fear.n.01",
    "protect": "protect.v.01",
    "harm": "harm.v.01",
    "man": "man.n.01",
    "woman": "woman.n.01",
}

COUNTERFACTUAL_PAIRS = (
    ("joy", "sadness"),
    ("hope", "despair"),
    ("affection", "hostility"),
    ("trust", "suspicion"),
    ("calmness", "agitation"),
    ("courage", "fear"),
    ("protect", "harm"),
    ("man", "woman"),
)

# Each row is one named experiment. A two-element tuple marks a pair unit;
# every pair row becomes two directed configurations.
MULTI_ROWS: dict[int, tuple[tuple[str | tuple[str, str], ...], ...]] = {
    2: (
        ("joy", "affection"),
        ("sadness", "despair"),
        ("hope", "courage"),
        ("hostility", "harm"),
        ("trust", "protect"),
        ("suspicion", "fear"),
        ("calmness", "man"),
        ("agitation", "woman"),
        ("joy", "suspicion"),
        ("sadness", "courage"),
        (("joy", "sadness"), "protect"),
        (("man", "woman"), "hope"),
    ),
    3: (
        ("joy", "affection", "trust"),
        ("sadness", "despair", "fear"),
        ("hope", "courage", "protect"),
        ("hostility", "harm", "agitation"),
        ("trust", "protect", "calmness"),
        ("suspicion", "fear", "agitation"),
        ("calmness", "man", "affection"),
        ("agitation", "woman", "courage"),
        ("joy", "suspicion", "man"),
        ("sadness", "courage", "woman"),
        (("trust", "suspicion"), "courage", "woman"),
        (("protect", "harm"), "hope", "man"),
    ),
    4: (
        ("joy", "affection", "trust", "woman"),
        ("sadness", "despair", "fear", "man"),
        ("hope", "courage", "protect", "woman"),
        ("hostility", "harm", "agitation", "man"),
        ("trust", "protect", "calmness", "man"),
        ("suspicion", "fear", "agitation", "woman"),
        ("calmness", "man", "affection", "hope"),
        ("agitation", "woman", "courage", "joy"),
        ("joy", "suspicion", "man", "harm"),
        ("sadness", "courage", "woman", "protect"),
        (("calmness", "agitation"), "affection", "protect", "man"),
        (("courage", "fear"), "joy", "trust", "woman"),
    ),
}


def stable_hash(value: Any) -> str:
    """SHA-256 of a JSON value using canonical key and whitespace settings."""
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_sources(
    concepts_dir: str | Path,
) -> tuple[dict[str, dict[str, dict]], dict[str, str]]:
    root = Path(concepts_dir)
    sources: dict[str, dict[str, dict]] = {}
    hashes: dict[str, str] = {}
    for filename in SOURCE_FILES:
        raw = (root / filename).read_bytes()
        hashes[filename] = hashlib.sha256(raw).hexdigest()
        entries = json.loads(raw)
        if not isinstance(entries, list):
            raise ValueError(f"{filename}: expected a list of concept entries")
        by_name: dict[str, dict] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("concept"), str):
                raise ValueError(f"{filename}: invalid concept entry")
            name = entry["concept"]
            if name in by_name:
                raise ValueError(f"{filename}: duplicate concept {name!r}")
            if entry.get("synset") != CONCEPT_SYNSETS.get(name):
                raise ValueError(f"{filename}: unexpected synset for {name!r}")
            words = entry.get("words")
            if (
                not isinstance(words, list)
                or not words
                or any(not isinstance(word, str) or not word.strip() for word in words)
            ):
                raise ValueError(f"{filename}: invalid words for {name!r}")
            by_name[name] = entry
        if set(by_name) != set(CONCEPT_SYNSETS):
            raise ValueError(f"{filename}: expected exactly the 16 comparison concepts")
        sources[filename] = by_name
    return sources, hashes


def _consensus_words(
    name: str, sources: dict[str, dict[str, dict]]
) -> tuple[list[str], list[dict]]:
    ranked: dict[str, dict] = {}
    for filename in SOURCE_FILES:
        source_words = sources[filename][name]["words"]
        length = len(source_words)
        for index, original in enumerate(source_words):
            word = original.strip().lower()
            row = ranked.setdefault(word, {"word": word, "score": 0.0, "sources": {}})
            if filename in row["sources"]:
                continue  # the first occurrence supplies this source's rank
            contribution = (length - index) / length
            row["score"] += contribution
            row["sources"][filename] = {
                "rank": index + 1,
                "list_length": length,
                "score": contribution,
            }
    details = sorted(
        ranked.values(), key=lambda row: (-row["score"], len(row["word"]), row["word"])
    )
    return [row["word"] for row in details[:100]], details


def _config(
    experiment_id: str,
    units: tuple[str | tuple[str, str], ...],
    direction: int | None = None,
) -> dict:
    pair_units = [unit for unit in units if isinstance(unit, tuple)]
    if len(pair_units) > 1:
        raise ValueError(
            f"{experiment_id}: at most one counterfactual pair is supported"
        )
    pair = pair_units[0] if pair_units else None
    if (
        pair is None
        and direction is not None
        or pair is not None
        and direction not in (0, 1)
    ):
        raise ValueError(f"{experiment_id}: invalid pair direction")
    positive: list[str] = []
    negative: list[str] = []
    for unit in units:
        if isinstance(unit, tuple):
            positive.append(unit[direction])
            negative.append(unit[1 - direction])
        else:
            positive.append(unit)
    if len(set(positive + negative)) != len(positive) + len(negative):
        raise ValueError(f"{experiment_id}: repeated concept in configuration")
    config_id = (
        experiment_id
        if pair is None
        else f"{experiment_id}-{positive[0]}-minus-{negative[0]}"
    )
    return {
        "id": config_id,
        "experiment_id": experiment_id,
        "unit_count": len(units),
        "kind": (
            "counterfactual" if pair else "single" if len(units) == 1 else "positive"
        ),
        "positive": positive,
        "negative": negative,
        "pair": list(pair) if pair else None,
        "atomic_count": len(positive) + len(negative),
    }


def _configurations() -> list[dict]:
    configs = [_config(f"n1-{name}", (name,)) for name in CONCEPT_SYNSETS]
    for first, second in COUNTERFACTUAL_PAIRS:
        experiment = f"n1-{first}-{second}"
        configs.extend(_config(experiment, ((first, second),), d) for d in (0, 1))
    for count, rows in MULTI_ROWS.items():
        for number, units in enumerate(rows, 1):
            experiment = f"n{count}-{number:02d}"
            pair = any(isinstance(unit, tuple) for unit in units)
            configs.extend(
                _config(experiment, units, d) for d in ((0, 1) if pair else (None,))
            )
    if len(configs) != 74 or len({row["id"] for row in configs}) != 74:
        raise AssertionError(
            "comparison battery must contain 74 unique directed configurations"
        )
    for count in (1, 2, 3, 4):
        represented = {
            name
            for row in configs
            if row["unit_count"] == count
            for name in row["positive"] + row["negative"]
        }
        if represented != set(CONCEPT_SYNSETS):
            raise AssertionError(f"n={count} battery does not cover all 16 concepts")
    return configs


def build_manifest(concepts_dir: str | Path = "concepts") -> dict:
    """Return JSON-ready custom concepts and the fixed experiment battery.

    ``provenance`` includes every candidate word's total score and per-source
    first rank, source length, and contribution; ``words`` contains its top 100.
    Input hashes bind a saved manifest to the four exact source files.
    """
    sources, input_hashes = _read_sources(concepts_dir)
    concepts = []
    for name, synset in CONCEPT_SYNSETS.items():
        words, details = _consensus_words(name, sources)
        if len(words) < 100:
            raise ValueError(f"{name}: fewer than 100 unique candidate words")
        concepts.append(
            {"concept": name, "synset": synset, "words": words, "provenance": details}
        )
    return {
        "version": 1,
        "input_hashes": input_hashes,
        "ranking": "sum((original_list_length - zero_based_first_rank) / original_list_length); ties: shorter, alphabetical",
        "concepts": concepts,
        "configs": _configurations(),
    }
