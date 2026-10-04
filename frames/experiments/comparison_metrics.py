"""Shared, model-light metrics for frame/subspace steering comparisons.

The comparison notebook supplies one fixed set of reference bases for every
method and concept source.  This module only turns continuations into hidden
states and evaluates those bases; it does not know how a continuation was
generated.  All reported cosine deltas are ``steered - baseline``.  For a
repelled concept, the summary reverses that delta so positive means a desirable
reduction in alignment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from numbers import Real
from typing import Any

import torch


def token_cosine(
    hidden: torch.Tensor, basis: torch.Tensor, eps: float = 1e-12
) -> float:
    """Mean token-to-span cosine ``mean_t ||B.T h_t|| / ||h_t||``.

    ``hidden`` has shape ``(T, d)`` and ``basis`` has shape ``(d, r)``.  Zero
    basis columns contribute zero coordinates, so fixed-rank trailing padding is
    naturally rank-neutral.  Zero hidden vectors contribute zero rather than a
    NaN.
    """
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape (tokens, dimension)")
    if basis.ndim != 2:
        raise ValueError("basis must have shape (dimension, rank)")
    if hidden.size(-1) != basis.size(0):
        raise ValueError("hidden and basis dimensions must agree")
    if hidden.size(0) == 0:
        raise ValueError("token cosine requires at least one non-special token")

    hidden64 = hidden.double()
    projected = (hidden64 @ basis.to(hidden.device).double()).norm(dim=-1)
    norm = hidden64.norm(dim=-1)
    values = torch.where(
        norm > eps,
        projected / norm.clamp_min(eps),
        torch.zeros_like(norm),
    )
    return float(values.clamp(0.0, 1.0).mean())


def summarize_cosines(
    per_concept: Mapping[str, Mapping[str, Real]],
    positives: Sequence[str] = (),
    negatives: Sequence[str] = (),
) -> dict[str, Any]:
    """Summarize constituent cosine deltas with attraction/repulsion signs.

    The returned keys are stable for generic table flattening:

    - ``attraction_mean_delta`` is the mean raw delta for positive concepts.
    - ``repulsion_mean_reduction`` is the mean of ``-delta`` for negatives.
    - ``signed_mean_delta`` / ``signed_min_delta`` use raw deltas for positives
      and negated deltas for negatives, so positive always means improvement.
    - ``all_requested_improved`` is true only when every requested signed delta
      is strictly positive.
    - ``net_attract_minus_repel`` is mean attraction delta minus the *raw*
      negative-concept delta.  Thus reduced repelled alignment increases it;
      equivalently it is attraction gain plus repulsion reduction.

    Empty positive or negative selections yield ``None`` for their respective
    mean and for the net comparison.  A selected name absent from
    ``per_concept`` is rejected to prevent silently evaluating different sets
    across reference views.
    """
    positive_names = tuple(positives)
    negative_names = tuple(negatives)
    if len(set(positive_names)) != len(positive_names):
        raise ValueError("positive concept names must be unique")
    if len(set(negative_names)) != len(negative_names):
        raise ValueError("negative concept names must be unique")
    overlap = sorted(set(positive_names) & set(negative_names))
    if overlap:
        raise ValueError(f"concepts cannot be both positive and negative: {overlap}")

    requested = positive_names + negative_names
    missing = [name for name in requested if name not in per_concept]
    if missing:
        raise ValueError(f"missing requested concepts: {', '.join(missing)}")

    positive_deltas = [float(per_concept[name]["delta"]) for name in positive_names]
    negative_deltas = [float(per_concept[name]["delta"]) for name in negative_names]
    signed = positive_deltas + [-delta for delta in negative_deltas]

    attraction = _mean_or_none(positive_deltas)
    repulsion_reduction = _mean_or_none([-delta for delta in negative_deltas])
    return {
        "attraction_mean_delta": attraction,
        "repulsion_mean_reduction": repulsion_reduction,
        "net_attract_minus_repel": (
            attraction + repulsion_reduction
            if attraction is not None and repulsion_reduction is not None
            else None
        ),
        "signed_mean_delta": _mean_or_none(signed),
        "signed_min_delta": min(signed) if signed else None,
        "all_requested_improved": (
            all(delta > 0.0 for delta in signed) if signed else None
        ),
        "requested_count": len(signed),
    }


def ppl_metrics(
    steered_ppl: Real | torch.Tensor | None,
    baseline_ppl: Real | torch.Tensor | None,
) -> dict[str, Any]:
    """Return continuation-only PPL delta/ratio, or an explicit invalid record."""
    steered = _finite_number(steered_ppl)
    baseline = _finite_number(baseline_ppl)
    if steered is None or baseline is None:
        return _invalid_ppl("non_finite_ppl")
    if baseline <= 0.0:
        return _invalid_ppl("non_positive_baseline_ppl")
    if steered <= 0.0:
        return _invalid_ppl("non_positive_steered_ppl")
    return {
        "valid": True,
        "invalid_reason": None,
        "steered_ppl": steered,
        "baseline_ppl": baseline,
        "delta_ppl": steered - baseline,
        "ppl_ratio": steered / baseline,
    }


def evaluate_continuation(
    fur: Any,
    continuation: str,
    baseline: str,
    references: Mapping[str, Mapping[str, torch.Tensor]],
    positive: Sequence[str] = (),
    negative: Sequence[str] = (),
    hidden_cache: (
        MutableMapping[str, tuple[torch.Tensor | None, str | None]] | None
    ) = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate one continuation pair in every fixed reference view.

    ``fur.model`` must supply ``tokenize(text)`` and
    ``forward_last_hiden_state(token_ids)``.  Token positions whose IDs occur
    in ``tokenizer.all_special_ids`` are excluded (including BOS/padding).  An
    empty string or a continuation containing only special tokens returns one
    invalid record per view; model failures deliberately propagate.

    ``hidden_cache`` is an optional caller-owned cache keyed by continuation
    text.  Scope it to a single loaded model and clear it between model runs.
    It is useful when one prompt-matched baseline is reused by many steering
    configurations; cached tensors retain the model's device and dtype.

    Each output view is ``{"valid", "invalid_reason", "per_concept",
    "summary"}``.  The summary has the exact keys documented by
    :func:`summarize_cosines`, so callers can flatten it uniformly across all
    four shared reference families.
    """
    steered_hidden, steered_reason = _cached_continuation_hidden(
        fur, continuation, hidden_cache
    )
    baseline_hidden, baseline_reason = _cached_continuation_hidden(
        fur, baseline, hidden_cache
    )
    if steered_reason or baseline_reason:
        reason = steered_reason or baseline_reason
        return {view: _invalid_view(reason) for view in references}

    result: dict[str, dict[str, Any]] = {}
    for view, concepts in references.items():
        per_concept = {
            name: _cosine_record(steered_hidden, baseline_hidden, basis)
            for name, basis in concepts.items()
        }
        result[view] = {
            "valid": True,
            "invalid_reason": None,
            "per_concept": per_concept,
            "summary": summarize_cosines(per_concept, positive, negative),
        }
    return result


def _cached_continuation_hidden(
    fur: Any,
    text: str,
    cache: MutableMapping[str, tuple[torch.Tensor | None, str | None]] | None,
) -> tuple[torch.Tensor | None, str | None]:
    if cache is not None and text in cache:
        return cache[text]
    result = _continuation_hidden(fur, text)
    if cache is not None:
        cache[text] = result
    return result


@torch.inference_mode()
def _continuation_hidden(fur: Any, text: str) -> tuple[torch.Tensor | None, str | None]:
    if not isinstance(text, str) or not text.strip():
        return None, "empty_continuation"
    tokens = fur.model.tokenize(text)
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.size(0) != 1:
        raise ValueError("continuation evaluation requires one tokenized text")
    if tokens.size(1) == 0:
        return None, "no_tokens"

    hidden = fur.model.forward_last_hiden_state(tokens)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.size(0) != 1:
        raise ValueError(
            "model must return hidden states of shape (1, tokens, dimension)"
        )
    if hidden.size(1) != tokens.size(1):
        raise ValueError("hidden-state and token lengths must agree")

    special_ids = set(getattr(fur.model.tokenizer, "all_special_ids", ()) or ())
    keep = torch.ones(tokens.size(1), dtype=torch.bool, device=hidden.device)
    if special_ids:
        special = torch.tensor(sorted(special_ids), device=tokens.device)
        keep = ~torch.isin(tokens.squeeze(0), special).to(hidden.device)
    if not keep.any():
        return None, "no_non_special_tokens"
    return hidden.squeeze(0)[keep], None


def _cosine_record(
    steered_hidden: torch.Tensor,
    baseline_hidden: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, float]:
    steered = token_cosine(steered_hidden, basis)
    baseline = token_cosine(baseline_hidden, basis)
    return {"steered": steered, "baseline": baseline, "delta": steered - baseline}


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _finite_number(value: Real | torch.Tensor | None) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _invalid_ppl(reason: str) -> dict[str, Any]:
    return {
        "valid": False,
        "invalid_reason": reason,
        "steered_ppl": None,
        "baseline_ppl": None,
        "delta_ppl": None,
        "ppl_ratio": None,
    }


def _invalid_view(reason: str) -> dict[str, Any]:
    return {
        "valid": False,
        "invalid_reason": reason,
        "per_concept": {},
        "summary": {
            "attraction_mean_delta": None,
            "repulsion_mean_reduction": None,
            "net_attract_minus_repel": None,
            "signed_mean_delta": None,
            "signed_min_delta": None,
            "all_requested_improved": None,
            "requested_count": 0,
        },
    }
