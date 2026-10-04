"""Reporting contracts: matched differences, direction balance, and null metrics."""

import csv
import json

import pytest

from frames.experiments.comparison_manifest import stable_hash
from frames.experiments.comparison_report import (
    _macro_metric,
    _pair_rows,
    _run_rows,
    _summary_rows,
    write_report,
)


def _record(config, method, prompt, delta, ppl=2.0, valid=True):
    summary = {
        "attraction_mean_delta": delta if valid else None,
        "repulsion_mean_reduction": None,
        "net_attract_minus_repel": None,
        "signed_mean_delta": delta if valid else None,
        "signed_min_delta": delta if valid else None,
        "all_requested_improved": delta > 0 if valid else None,
        "requested_count": 1 if valid else 0,
    }
    view = {
        "valid": valid,
        "invalid_reason": None if valid else "empty_continuation",
        "per_concept": (
            {"joy": {"steered": 0.5, "baseline": 0.2, "delta": 0.3}} if valid else {}
        ),
        "summary": summary,
    }
    return {
        "run_id": f"{config['id']}/{method}/{prompt}",
        "source": "custom",
        "method": method,
        "lambda": 0.4,
        "config": config,
        "prompt": {"id": prompt},
        "continuation": "text",
        "metrics": {
            "cosine": {
                view_name: view
                for view_name in (
                    "wordnet_frame",
                    "wordnet_subspace",
                    "custom_frame",
                    "custom_subspace",
                )
            },
            "fluency": {
                "valid": valid,
                "invalid_reason": None if valid else "non_finite_ppl",
                "steered_ppl": 10 + ppl if valid else None,
                "baseline_ppl": 10 if valid else None,
                "delta_ppl": ppl if valid else None,
                "ppl_ratio": 1 + ppl / 10 if valid else None,
            },
        },
    }


def _config(id, experiment, kind="counterfactual"):
    return {
        "id": id,
        "experiment_id": experiment,
        "kind": kind,
        "unit_count": 1,
        "atomic_count": 2 if kind == "counterfactual" else 1,
        "positive": ["joy"],
        "negative": ["sadness"] if kind == "counterfactual" else [],
    }


def test_paired_differences_are_config_prompt_and_view_matched():
    a = _config("a", "pair")
    b = _config("b", "pair")
    records = [
        _record(a, "frh_blend", "p1", 0.3, 4),
        _record(a, "subspace", "p1", 0.1, 1),
        _record(b, "frh_blend", "p1", -0.2, 2),
        _record(b, "subspace", "p1", -0.4, 5),
        _record(a, "frh_blend", "p2", 0.5, 9, valid=False),
        _record(a, "subspace", "p2", 0.2, 1),
    ]
    runs, concepts = _run_rows(records)
    pairs = _pair_rows(runs)
    selected = {
        (r["config_id"], r["prompt_id"]): r
        for r in pairs
        if r["view"] == "custom_frame"
    }
    assert len(concepts) == 5 * 4
    assert (
        abs(selected["a", "p1"]["frh_minus_subspace_signed_mean_delta"] - 0.2) < 1e-12
    )
    assert selected["a", "p1"]["frh_minus_subspace_delta_ppl"] == 3
    assert selected["b", "p1"]["frh_minus_subspace_delta_ppl"] == -3
    assert selected["a", "p2"]["cosine_pair_valid"] is False
    assert selected["a", "p2"]["frh_minus_subspace_delta_ppl"] is None


def test_macro_averages_directions_then_experiments_and_counts_invalid():
    pair_a = _config("a", "pair")
    pair_b = _config("b", "pair")
    single = _config("c", "single", "single")
    # The invalid second pair direction at p2 cannot silently halve a pair.
    records = [
        _record(pair_a, "frh_blend", "p1", 0.1),
        _record(pair_b, "frh_blend", "p1", 0.3),
        _record(pair_a, "frh_blend", "p2", 0.9),
        _record(pair_b, "frh_blend", "p2", 0.5, valid=False),
        _record(single, "frh_blend", "p1", 0.8),
        _record(single, "frh_blend", "p2", 1.0),
    ]
    runs, _ = _run_rows(records)
    selected = [
        r
        for r in runs
        if r["view"] == "wordnet_frame" and r["kind"] == "counterfactual"
    ]
    value, n_experiments, n_prompts = _macro_metric(
        selected,
        {"pair": {"a", "b"}},
        ["p1", "p2"],
        "signed_mean_delta",
        "cosine_valid",
    )
    assert abs(value - 0.2) < 1e-12
    assert n_experiments == 1 and n_prompts == 1
    manifest = {
        "configs": [pair_a, pair_b, single],
        "prompts": [{"id": "p1"}, {"id": "p2"}],
        "sources": ["custom"],
        "methods": ["frh_blend"],
        "settings": {"lambdas": [0.4]},
    }
    summaries = _summary_rows(runs, manifest)
    pair_summary = next(
        row
        for row in summaries
        if row["view"] == "wordnet_frame" and row["family"] == "counterfactual"
    )
    assert pair_summary["expected_runs"] == 4
    assert pair_summary["cosine_invalid_runs"] == 1
    assert pair_summary["signed_mean_delta_complete_prompts"] == 1
    assert abs(pair_summary["signed_mean_delta"] - 0.2) < 1e-12


def test_macro_gives_experiments_equal_weight_with_unequal_available_prompts():
    rows = [
        {
            "experiment_id": "a",
            "prompt_id": "p1",
            "config_id": "a",
            "cosine_valid": True,
            "signed_mean_delta": 0.0,
        },
        {
            "experiment_id": "a",
            "prompt_id": "p2",
            "config_id": "a",
            "cosine_valid": True,
            "signed_mean_delta": 0.0,
        },
        {
            "experiment_id": "b",
            "prompt_id": "p1",
            "config_id": "b",
            "cosine_valid": True,
            "signed_mean_delta": 1.0,
        },
    ]
    value, experiments, prompts = _macro_metric(
        rows,
        {"a": {"a"}, "b": {"b"}},
        ["p1", "p2"],
        "signed_mean_delta",
        "cosine_valid",
    )
    assert value == 0.5
    assert experiments == 2 and prompts == 3


def test_write_report_marks_partial_and_quotes_gallery_text(tmp_path):
    config = _config("a", "single", "single")
    manifest = {
        "configs": [config],
        "prompts": [{"id": "p1", "text": "prompt text"}],
        "sources": ["custom"],
        "methods": ["frh_blend", "subspace"],
        "settings": {"lambdas": [0.4]},
    }
    prefix = tmp_path / "comparison"
    (tmp_path / "comparison_manifest.json").write_text(json.dumps(manifest))
    construction = {"test_backend": True}
    identity = stable_hash({"manifest": manifest, "construction": construction})
    (tmp_path / "comparison_construction.json").write_text(json.dumps(construction))
    baseline = {
        "identity": identity,
        "prompts": manifest["prompts"],
        "continuations": ["baseline\n# unexpected heading"],
    }
    (tmp_path / "comparison_baselines.json").write_text(json.dumps(baseline))
    record = _record(config, "frh_blend", "p1", 0.4)
    record.update(
        identity=identity,
        baseline_hash=stable_hash(baseline),
        run_id=stable_hash([identity, "custom", "frh_blend", config["id"], 0.4, "p1"]),
    )
    record["prompt"] = manifest["prompts"][0]
    record["continuation"] = "first line\n# generated heading"
    (tmp_path / "comparison.jsonl").write_text(json.dumps(record) + "\n")
    paths = write_report(prefix)
    status = json.loads(paths["report_status"].read_text())
    assert status["status"] == "partial"
    assert status["expected_runs"] == 2 and status["missing_runs"] == 1
    gallery = paths["gallery"].read_text()
    assert "> # generated heading" in gallery
    assert "> # unexpected heading" in gallery
    assert "**partial**" in gallery
    with paths["paired"].open(newline="") as stream:
        paired = list(csv.DictReader(stream))
    assert len(paired) == 4
    assert paired[0]["pair_present"] == "False"
    assert paired[0]["frh_minus_subspace_delta_ppl"] == ""
    baseline["continuations"][0] = "modified"
    (tmp_path / "comparison_baselines.json").write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="provenance"):
        write_report(prefix)
