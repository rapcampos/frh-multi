"""Shared configuration and text diagnostics for long-generation notebooks.

The expensive notebooks deliberately keep model construction and CGD method
definitions local.  This module owns the CPU-only seams that should be checked
before a model is loaded: resolving catalog IDs, reporting the run matrix, and
computing token-level degeneration diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


class LongGenerationConfigError(ValueError):
    """Raised when ``long-generation-inputs.yaml`` is internally inconsistent."""


@dataclass(frozen=True)
class ResolvedLongGenerationExperiment:
    """One experiment with catalog references expanded in deterministic order."""

    name: str
    steps: int
    lambdas: tuple[float, ...]
    prompts: tuple[dict[str, Any], ...]
    composition_pairs: tuple[dict[str, Any], ...] = ()
    negative_directions: tuple[dict[str, Any], ...] = ()

    def guided_run_count(self, method_count: int) -> int:
        """Number of guided continuations in the resolved Cartesian product."""
        if (
            isinstance(method_count, bool)
            or not isinstance(method_count, int)
            or method_count < 1
        ):
            raise ValueError("method_count must be a positive integer")
        concepts = len(self.composition_pairs) or len(self.negative_directions)
        return len(self.prompts) * concepts * method_count * len(self.lambdas)

    def output_token_budget(self, method_count: int) -> int:
        """Nominal guided-token cap; early EOS can make the actual count lower."""
        return self.guided_run_count(method_count) * self.steps


@dataclass(frozen=True)
class TokenBin:
    """A half-open slice of token IDs used for longitudinal measurements."""

    index: int
    start: int
    stop: int
    token_ids: tuple[int, ...]


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LongGenerationConfigError(f"{location} must be a mapping")
    return value


def _catalog(
    raw: Mapping[str, Any],
    field: str,
    required_fields: tuple[str, ...],
) -> tuple[tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
    rows = raw.get(field)
    if not isinstance(rows, list) or not rows:
        raise LongGenerationConfigError(f"{field} must be a non-empty list")

    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(rows):
        row = dict(_require_mapping(value, f"{field}[{index}]"))
        missing = [name for name in required_fields if name not in row]
        if missing:
            raise LongGenerationConfigError(
                f"{field}[{index}] is missing: {', '.join(missing)}"
            )
        row_id = row["id"]
        if not isinstance(row_id, str) or not row_id.strip():
            raise LongGenerationConfigError(f"{field}[{index}].id must be a string")
        if row_id in by_id:
            raise LongGenerationConfigError(f"duplicate ID {row_id!r} in {field}")
        ordered.append(row)
        by_id[row_id] = row
    return tuple(ordered), by_id


def _resolve_ids(
    selector: Any,
    catalog_name: str,
    ordered: tuple[dict[str, Any], ...],
    by_id: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if selector is None:
        return ()
    if selector == "all":
        return ordered
    if not isinstance(selector, list) or not selector:
        raise LongGenerationConfigError(
            f"{catalog_name} selection must be 'all' or a non-empty ID list"
        )
    if not all(isinstance(item, str) for item in selector):
        raise LongGenerationConfigError(f"{catalog_name} IDs must be strings")

    duplicate_ids = sorted({item for item in selector if selector.count(item) > 1})
    if duplicate_ids:
        raise LongGenerationConfigError(
            f"duplicate {catalog_name} selection IDs: {', '.join(duplicate_ids)}"
        )
    unknown_ids = [item for item in selector if item not in by_id]
    if unknown_ids:
        raise LongGenerationConfigError(
            f"unknown {catalog_name} selection IDs: {', '.join(unknown_ids)}"
        )
    return tuple(by_id[item] for item in selector)


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LongGenerationConfigError(f"{location} must be a positive integer")
    return value


def _lambdas(value: Any, location: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise LongGenerationConfigError(f"{location} must be a non-empty list")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0:
            raise LongGenerationConfigError(
                f"{location}[{index}] must be a non-negative number"
            )
        number = float(item)
        if number in result:
            raise LongGenerationConfigError(f"duplicate value {number} in {location}")
        result.append(number)
    return tuple(result)


def load_long_generation_experiment(
    path: str | Path, experiment_name: str
) -> ResolvedLongGenerationExperiment:
    """Load and validate one experiment from the shared YAML catalog.

    Explicit ID lists retain their listed order.  The special selector ``all``
    expands in catalog order, which makes full and resumed runs deterministic.
    """
    path = Path(path)
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise LongGenerationConfigError(f"invalid YAML in {path}: {exc}") from exc
    raw = _require_mapping(loaded, str(path))

    prompts, prompts_by_id = _catalog(
        raw, "prompt_catalog", ("id", "user", "assistant_prefix")
    )
    pairs, pairs_by_id = _catalog(
        raw, "composition_pair_catalog", ("id", "a", "b", "rho")
    )
    directions, directions_by_id = _catalog(
        raw, "negative_direction_catalog", ("id", "attract", "repel")
    )

    experiments = _require_mapping(raw.get("experiments"), "experiments")
    if experiment_name not in experiments:
        raise LongGenerationConfigError(f"unknown experiment {experiment_name!r}")
    experiment = _require_mapping(
        experiments[experiment_name], f"experiments.{experiment_name}"
    )

    resolved_prompts = _resolve_ids(
        experiment.get("prompt_ids"), "prompt", prompts, prompts_by_id
    )
    resolved_pairs = _resolve_ids(
        experiment.get("composition_pair_ids"),
        "composition pair",
        pairs,
        pairs_by_id,
    )
    resolved_directions = _resolve_ids(
        experiment.get("negative_direction_ids"),
        "negative direction",
        directions,
        directions_by_id,
    )
    if not resolved_prompts:
        raise LongGenerationConfigError(
            f"experiments.{experiment_name} must select at least one prompt"
        )
    if bool(resolved_pairs) == bool(resolved_directions):
        raise LongGenerationConfigError(
            f"experiments.{experiment_name} must select exactly one of "
            "composition_pair_ids and negative_direction_ids"
        )

    return ResolvedLongGenerationExperiment(
        name=experiment_name,
        steps=_positive_int(
            experiment.get("steps"), f"experiments.{experiment_name}.steps"
        ),
        lambdas=_lambdas(
            experiment.get("lambdas"), f"experiments.{experiment_name}.lambdas"
        ),
        prompts=resolved_prompts,
        composition_pairs=resolved_pairs,
        negative_directions=resolved_directions,
    )


def token_bins(token_ids: Iterable[int], width: int = 20) -> tuple[TokenBin, ...]:
    """Partition a token sequence into deterministic half-open bins."""
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    tokens = tuple(int(token_id) for token_id in token_ids)
    return tuple(
        TokenBin(
            index // width,
            index,
            min(index + width, len(tokens)),
            tokens[index : index + width],
        )
        for index in range(0, len(tokens), width)
    )


def matched_segment_fluency(
    steered_token_nll: Sequence[float],
    baseline_token_nll: Sequence[float],
    start: int,
    stop: int,
    min_shared_tokens: int = 10,
) -> dict[str, float | int]:
    """Score one aligned continuation span without resetting it to BOS.

    ``steered_token_nll`` and ``baseline_token_nll`` must already contain the
    prefix-conditioned loss for each continuation token.  The comparison uses
    the same number of available tokens on both sides.  Short shared tails are
    retained through ``shared_tokens`` but receive NaN fluency values so they
    cannot create unstable PPL-ratio spikes.

    The log ratio is the difference in mean NLL.  Exponentiating it gives the
    per-cell PPL ratio; downstream groups should average the log ratio and then
    exponentiate (a geometric mean), not average raw ratios.
    """
    for value, name in (
        (start, "start"),
        (stop, "stop"),
        (min_shared_tokens, "min_shared_tokens"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if start < 0 or stop < start:
        raise ValueError("require 0 <= start <= stop")
    if min_shared_tokens < 1:
        raise ValueError("min_shared_tokens must be positive")

    shared_stop = min(stop, len(steered_token_nll), len(baseline_token_nll))
    shared_tokens = max(shared_stop - start, 0)
    empty = {
        "shared_tokens": shared_tokens,
        "segment_mean_nll": float("nan"),
        "baseline_segment_mean_nll": float("nan"),
        "segment_ppl": float("nan"),
        "baseline_segment_ppl": float("nan"),
        "segment_log_ppl_ratio": float("nan"),
        "segment_ppl_ratio": float("nan"),
    }
    if shared_tokens < min_shared_tokens:
        return empty

    steered_mean_nll = (
        math.fsum(float(value) for value in steered_token_nll[start:shared_stop])
        / shared_tokens
    )
    baseline_mean_nll = (
        math.fsum(float(value) for value in baseline_token_nll[start:shared_stop])
        / shared_tokens
    )
    log_ratio = steered_mean_nll - baseline_mean_nll
    return {
        "shared_tokens": shared_tokens,
        "segment_mean_nll": steered_mean_nll,
        "baseline_segment_mean_nll": baseline_mean_nll,
        "segment_ppl": math.exp(steered_mean_nll),
        "baseline_segment_ppl": math.exp(baseline_mean_nll),
        "segment_log_ppl_ratio": log_ratio,
        "segment_ppl_ratio": math.exp(log_ratio),
    }


def longest_nonoverlapping_repeated_span(token_ids: Sequence[int]) -> int:
    """Length of the longest repeated contiguous token span without overlap."""
    tokens = tuple(token_ids)
    previous = [0] * (len(tokens) + 1)
    longest = 0
    for left in range(1, len(tokens) + 1):
        current = [0] * (len(tokens) + 1)
        for right in range(left + 1, len(tokens) + 1):
            if tokens[left - 1] == tokens[right - 1]:
                current[right] = previous[right - 1] + 1
                longest = max(longest, min(current[right], right - left))
        previous = current
    return longest


def repetition_metrics(token_ids: Sequence[int], ngram_size: int = 4) -> dict[str, Any]:
    """Token-level diversity and repetition diagnostics for one continuation."""
    if (
        isinstance(ngram_size, bool)
        or not isinstance(ngram_size, int)
        or ngram_size < 1
    ):
        raise ValueError("ngram_size must be a positive integer")
    tokens = tuple(int(token_id) for token_id in token_ids)
    ngrams = [tokens[i : i + ngram_size] for i in range(len(tokens) - ngram_size + 1)]
    return {
        "token_count": len(tokens),
        "distinct_token_ratio": len(set(tokens)) / len(tokens) if tokens else 0.0,
        "repeated_ngram_rate": (
            1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
        ),
        "max_repeated_span_tokens": longest_nonoverlapping_repeated_span(tokens),
    }


def generation_metadata(
    tokenizer: Any,
    continuation: str,
    nominal_steps: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Derived length, termination, timing, and repetition data for a log row.

    The historical CGD loop proposes one seed token before iterating
    ``nominal_steps`` times, so a non-EOS continuation can contain
    ``nominal_steps + 1`` generated tokens.  Both values are recorded rather
    than silently redefining the established ``steps`` parameter.
    """
    encoded = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("continuation tokenization must produce one sequence")
        encoded = encoded[0]
    token_ids = [int(token_id) for token_id in encoded]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    termination = "eos" if eos_token_id in token_ids else "step_cap"
    return {
        **repetition_metrics(token_ids),
        "nominal_steps": nominal_steps,
        "historical_non_eos_token_cap": nominal_steps + 1,
        "termination": termination,
        "ended_early": termination == "eos" and len(token_ids) < nominal_steps + 1,
        "elapsed_seconds": float(elapsed_seconds),
    }
