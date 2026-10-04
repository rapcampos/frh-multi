"""Derived, model-free tables and gallery for the NB68 comparison battery.

The directed generation log is the source of truth.  A pair experiment has two
directions; its directions are averaged within each prompt before experiments
receive equal weight in group summaries.  Missing or invalid directions never
become zeros.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

VIEWS = ("wordnet_frame", "wordnet_subspace", "custom_frame", "custom_subspace")
SUMMARY_FIELDS = (
    "attraction_mean_delta",
    "repulsion_mean_reduction",
    "net_attract_minus_repel",
    "signed_mean_delta",
    "signed_min_delta",
    "all_requested_improved",
)
FLUENCY_FIELDS = ("steered_ppl", "baseline_ppl", "delta_ppl", "ppl_ratio")
GROUP_FIELDS = ("source", "method", "lambda", "unit_count", "family", "view")
PAIR_GROUP_FIELDS = ("source", "lambda", "unit_count", "family", "view")


def _finite(value):
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values):
    numbers = [_finite(value) for value in values]
    numbers = [number for number in numbers if number is not None]
    return mean(numbers) if numbers else None


def _write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load(prefix):
    from .comparison_manifest import stable_hash
    from .steering_comparison import read_records

    manifest = json.loads(Path(f"{prefix}_manifest.json").read_text())
    baseline = json.loads(Path(f"{prefix}_baselines.json").read_text())
    construction = json.loads(Path(f"{prefix}_construction.json").read_text())
    identity = stable_hash({"manifest": manifest, "construction": construction})
    if baseline["identity"] != identity:
        raise ValueError("Baseline identity differs from the frozen experiment")
    baseline_hash = stable_hash(baseline)
    records = read_records(f"{prefix}.jsonl")
    prompts = {prompt["id"]: i for i, prompt in enumerate(manifest["prompts"])}
    if baseline["prompts"] != manifest["prompts"] or len(
        baseline["continuations"]
    ) != len(prompts):
        raise ValueError("Baseline prompts do not match the manifest")
    config_ids = {config["id"]: config for config in manifest["configs"]}
    seen = set()
    for record in records:
        if record["identity"] != identity or record["baseline_hash"] != baseline_hash:
            raise ValueError(
                "Generation provenance differs from its manifest/baselines"
            )
        config = record["config"]
        prompt = record["prompt"]
        if config_ids.get(config["id"]) != config or prompt["id"] not in prompts:
            raise ValueError("Generation record is outside its manifest")
        if prompt != manifest["prompts"][prompts[prompt["id"]]]:
            raise ValueError("Generation prompt differs from its manifest")
        key = (
            record["source"],
            record["method"],
            record["lambda"],
            config["id"],
            prompt["id"],
        )
        if (
            record["source"] not in manifest["sources"]
            or record["method"] not in manifest["methods"]
            or record["lambda"] not in manifest["settings"]["lambdas"]
        ):
            raise ValueError("Generation condition is outside its manifest")
        expected_id = stable_hash([identity, key[0], key[1], key[3], key[2], key[4]])
        if record["run_id"] != expected_id:
            raise ValueError("Generation run ID differs from its condition")
        if key in seen:
            raise ValueError(f"Duplicate configuration/prompt in log: {key}")
        seen.add(key)
    return manifest, baseline, records, prompts


def _run_rows(records):
    rows = []
    per_concept = []
    for record in records:
        config = record["config"]
        fluency = record["metrics"]["fluency"]
        for view in VIEWS:
            metric = record["metrics"]["cosine"].get(view)
            if metric is None:
                raise ValueError(f"Missing reference view {view}")
            row = {
                "run_id": record["run_id"],
                "source": record["source"],
                "method": record["method"],
                "lambda": record["lambda"],
                "experiment_id": config["experiment_id"],
                "config_id": config["id"],
                "prompt_id": record["prompt"]["id"],
                "unit_count": config["unit_count"],
                "atomic_count": config["atomic_count"],
                "kind": config["kind"],
                "family": "counterfactual" if config["negative"] else "positive",
                "view": view,
                "cosine_valid": bool(metric["valid"]),
                "cosine_invalid_reason": metric["invalid_reason"],
                "fluency_valid": bool(fluency["valid"]),
                "fluency_invalid_reason": fluency["invalid_reason"],
                "continuation_tokens": record["metrics"].get("continuation_tokens"),
                "baseline_tokens": record["metrics"].get("baseline_tokens"),
            }
            for field in SUMMARY_FIELDS:
                row[field] = metric["summary"].get(field) if metric["valid"] else None
            row["requested_count"] = (
                metric["summary"].get("requested_count") if metric["valid"] else None
            )
            for field in FLUENCY_FIELDS:
                row[field] = fluency.get(field) if fluency["valid"] else None
            rows.append(row)
            if metric["valid"]:
                for name in sorted(metric["per_concept"]):
                    values = metric["per_concept"][name]
                    role = (
                        "attract"
                        if name in config["positive"]
                        else "repel" if name in config["negative"] else "reference"
                    )
                    per_concept.append(
                        {
                            **{
                                field: row[field]
                                for field in (
                                    "run_id",
                                    "source",
                                    "method",
                                    "lambda",
                                    "experiment_id",
                                    "config_id",
                                    "prompt_id",
                                    "unit_count",
                                    "kind",
                                    "family",
                                    "view",
                                )
                            },
                            "concept": name,
                            "role": role,
                            "steered": values["steered"],
                            "baseline": values["baseline"],
                            "delta": values["delta"],
                            "signed_delta": (
                                -values["delta"]
                                if role == "repel"
                                else values["delta"] if role == "attract" else None
                            ),
                        }
                    )
    return rows, per_concept


def _expected_groups(manifest):
    by_group = defaultdict(lambda: defaultdict(set))
    for config in manifest["configs"]:
        for source in manifest["sources"]:
            for method in manifest["methods"]:
                for lam in manifest["settings"]["lambdas"]:
                    for view in VIEWS:
                        family = "counterfactual" if config["negative"] else "positive"
                        key = (source, method, lam, config["unit_count"], family, view)
                        by_group[key][config["experiment_id"]].add(config["id"])
    return by_group


def _macro_metric(rows, expected_experiments, prompts, field, validity):
    """Average complete directions per prompt, then prompts, then experiments."""
    prompts = tuple(prompts)
    indexed = {
        (row["experiment_id"], row["prompt_id"], row["config_id"]): row for row in rows
    }
    experiment_values = []
    complete_prompt_count = 0
    for experiment_id, config_ids in expected_experiments.items():
        prompt_values = []
        for prompt_id in prompts:
            direction_rows = [
                indexed.get((experiment_id, prompt_id, config_id))
                for config_id in config_ids
            ]
            if any(
                row is None or not row[validity] or _finite(row[field]) is None
                for row in direction_rows
            ):
                continue
            prompt_values.append(mean(float(row[field]) for row in direction_rows))
        complete_prompt_count += len(prompt_values)
        if prompt_values:
            experiment_values.append(mean(prompt_values))
    return _mean(experiment_values), len(experiment_values), complete_prompt_count


def _summary_rows(rows, manifest):
    observed = defaultdict(list)
    for row in rows:
        observed[tuple(row[field] for field in GROUP_FIELDS)].append(row)
    summaries = []
    for key, experiments in sorted(_expected_groups(manifest).items()):
        group = observed[key]
        expected = sum(len(configs) for configs in experiments.values()) * len(
            manifest["prompts"]
        )
        result = dict(zip(GROUP_FIELDS, key))
        result.update(
            expected_runs=expected,
            recorded_runs=len(group),
            missing_runs=expected - len(group),
            cosine_valid_runs=sum(row["cosine_valid"] for row in group),
            cosine_invalid_runs=sum(not row["cosine_valid"] for row in group),
            fluency_valid_runs=sum(row["fluency_valid"] for row in group),
            fluency_invalid_runs=sum(not row["fluency_valid"] for row in group),
            expected_experiments=len(experiments),
        )
        for field in SUMMARY_FIELDS + FLUENCY_FIELDS:
            validity = "fluency_valid" if field in FLUENCY_FIELDS else "cosine_valid"
            value, n_experiments, n_prompts = _macro_metric(
                group,
                experiments,
                (p["id"] for p in manifest["prompts"]),
                field,
                validity,
            )
            result[field] = value
            result[f"{field}_experiments"] = n_experiments
            result[f"{field}_complete_prompts"] = n_prompts
        summaries.append(result)
    return summaries


def _pair_rows(rows):
    indexed = defaultdict(dict)
    for row in rows:
        key = (
            row["source"],
            row["lambda"],
            row["config_id"],
            row["prompt_id"],
            row["view"],
        )
        indexed[key][row["method"]] = row
    paired = []
    for key, methods in sorted(indexed.items()):
        frh = methods.get("frh_blend")
        subspace = methods.get("subspace")
        exemplar = frh or subspace
        result = {
            "source": key[0],
            "lambda": key[1],
            "config_id": key[2],
            "prompt_id": key[3],
            "view": key[4],
            "experiment_id": exemplar["experiment_id"],
            "unit_count": exemplar["unit_count"],
            "kind": exemplar["kind"],
            "family": exemplar["family"],
            "pair_present": frh is not None and subspace is not None,
            "cosine_pair_valid": bool(
                frh and subspace and frh["cosine_valid"] and subspace["cosine_valid"]
            ),
            "fluency_pair_valid": bool(
                frh and subspace and frh["fluency_valid"] and subspace["fluency_valid"]
            ),
        }
        for field in SUMMARY_FIELDS + FLUENCY_FIELDS:
            valid = result[
                "fluency_pair_valid" if field in FLUENCY_FIELDS else "cosine_pair_valid"
            ]
            a = _finite(frh[field]) if frh else None
            b = _finite(subspace[field]) if subspace else None
            result[f"frh_minus_subspace_{field}"] = (
                a - b if valid and a is not None and b is not None else None
            )
        paired.append(result)
    return paired


def _paired_summary_rows(paired, manifest):
    observed = defaultdict(list)
    for row in paired:
        observed[tuple(row[field] for field in PAIR_GROUP_FIELDS)].append(row)
    result_rows = []
    expected = _expected_groups(manifest)
    for full_key, experiments in sorted(expected.items()):
        source, method, lam, count, family, view = full_key
        if method != "frh_blend":
            continue
        key = (source, lam, count, family, view)
        rows = observed[key]
        total = sum(len(configs) for configs in experiments.values()) * len(
            manifest["prompts"]
        )
        result = dict(zip(PAIR_GROUP_FIELDS, key))
        result.update(
            expected_pairs=total,
            recorded_pairs=len(rows),
            missing_pairs=total - len(rows),
            present_pairs=sum(row["pair_present"] for row in rows),
            cosine_valid_pairs=sum(row["cosine_pair_valid"] for row in rows),
            fluency_valid_pairs=sum(row["fluency_pair_valid"] for row in rows),
        )
        for field in SUMMARY_FIELDS + FLUENCY_FIELDS:
            paired_field = f"frh_minus_subspace_{field}"
            validity = (
                "fluency_pair_valid" if field in FLUENCY_FIELDS else "cosine_pair_valid"
            )
            value, n_experiments, n_prompts = _macro_metric(
                rows,
                experiments,
                (p["id"] for p in manifest["prompts"]),
                paired_field,
                validity,
            )
            result[paired_field] = value
            result[f"{paired_field}_experiments"] = n_experiments
            result[f"{paired_field}_complete_prompts"] = n_prompts
        result_rows.append(result)
    return result_rows


def _plot(rows, path, lambdas):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(lambdas), len(VIEWS), figsize=(17, 4.5 * len(lambdas)), squeeze=False
    )
    styles = {
        ("wordnet", "frh_blend"): ("#1f77b4", "o"),
        ("wordnet", "subspace"): ("#ff7f0e", "^"),
        ("custom", "frh_blend"): ("#2ca02c", "s"),
        ("custom", "subspace"): ("#d62728", "D"),
    }
    for i, lam in enumerate(lambdas):
        for j, view in enumerate(VIEWS):
            ax = axes[i][j]
            for (source, method), (color, marker) in styles.items():
                selected = [
                    r
                    for r in rows
                    if r["lambda"] == lam
                    and r["view"] == view
                    and r["source"] == source
                    and r["method"] == method
                    and r["cosine_valid"]
                    and r["fluency_valid"]
                ]
                for kind, alpha in (
                    ("counterfactual", 0.35),
                    ("single", 0.7),
                    ("positive", 0.7),
                ):
                    subset = [r for r in selected if r["kind"] == kind]
                    ax.scatter(
                        [r["delta_ppl"] for r in subset],
                        [r["signed_mean_delta"] for r in subset],
                        c=color,
                        marker=marker,
                        s=13,
                        alpha=alpha,
                        label=f"{source}/{method}" if kind == "single" else None,
                    )
            ax.axhline(0, color="gray", linewidth=0.7)
            ax.axvline(0, color="gray", linewidth=0.7)
            ax.set_title(f"{view}; λ={lam:g}")
            ax.set_xlabel("ΔPPL (steered − baseline)")
            ax.set_ylabel("Mean signed Δcosine")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _quoted(text):
    """Keep arbitrary model text inside a Markdown block quote."""
    return "\n".join("> " + line for line in str(text).splitlines()) or "> "


def _gallery(records, baseline, prompts, path, status):
    grouped = defaultdict(dict)
    for record in records:
        key = (
            record["source"],
            record["lambda"],
            record["config"]["unit_count"],
            record["config"]["id"],
            record["prompt"]["id"],
        )
        grouped[key][record["method"]] = record
    lines = [
        "# Matched FRH blend and subspace continuations",
        "",
        f"Status: **{status['status']}** — {status['recorded_runs']} of "
        f"{status['expected_runs']} expected runs ({status['missing_runs']} missing).",
        "",
    ]
    for (source, lam, count, config_id, prompt_id), methods in sorted(grouped.items()):
        prompt = prompts[prompt_id]
        baseline_text = baseline["continuations"][prompt["index"]]
        lines.extend(
            [
                f"## {source} · λ={lam:g} · {count} unit(s) · {config_id} · {prompt_id}",
                "",
                "**Prompt:**",
                "",
                _quoted(prompt["text"]),
                "",
                "**Baseline:**",
                "",
                _quoted(baseline_text),
                "",
            ]
        )
        for method in ("frh_blend", "subspace"):
            record = methods.get(method)
            lines.extend(
                [
                    f"**{method}:**",
                    "",
                    _quoted(record["continuation"] if record else "[missing]"),
                    "",
                ]
            )
    Path(path).write_text("\n".join(lines) + "\n")


def write_report(output_prefix):
    """Write deterministic CSV, chart, and text reports from a frozen battery.

    Returns output paths keyed by artifact name.  A partial log is allowed and
    missing runs are counted against the manifest's full expected matrix.
    """
    prefix = Path(output_prefix)
    manifest, baseline, records, prompt_index = _load(prefix)
    rows, concepts = _run_rows(records)
    summaries = _summary_rows(rows, manifest)
    paired = _pair_rows(rows)
    paired_summaries = _paired_summary_rows(paired, manifest)
    paths = {
        name: Path(f"{prefix}_{name}.{ext}")
        for name, ext in (
            ("per_concept", "csv"),
            ("runs", "csv"),
            ("summary", "csv"),
            ("paired", "csv"),
            ("paired_summary", "csv"),
            ("cosine_ppl", "png"),
            ("gallery", "md"),
            ("report_status", "json"),
        )
    }
    expected_runs = (
        len(manifest["configs"])
        * len(manifest["prompts"])
        * len(manifest["methods"])
        * len(manifest["sources"])
        * len(manifest["settings"]["lambdas"])
    )
    status = {
        "status": "complete" if len(records) == expected_runs else "partial",
        "expected_runs": expected_runs,
        "recorded_runs": len(records),
        "missing_runs": expected_runs - len(records),
        "cosine_invalid_view_rows": sum(not row["cosine_valid"] for row in rows),
        "fluency_invalid_view_rows": sum(not row["fluency_valid"] for row in rows),
    }
    paths["report_status"].write_text(json.dumps(status, indent=2) + "\n")
    common = [
        "run_id",
        "source",
        "method",
        "lambda",
        "experiment_id",
        "config_id",
        "prompt_id",
        "unit_count",
        "kind",
        "family",
        "view",
    ]
    _write_csv(
        paths["per_concept"],
        common + ["concept", "role", "steered", "baseline", "delta", "signed_delta"],
        concepts,
    )
    _write_csv(
        paths["runs"],
        common
        + [
            "atomic_count",
            "cosine_valid",
            "cosine_invalid_reason",
            "fluency_valid",
            "fluency_invalid_reason",
            "continuation_tokens",
            "baseline_tokens",
            "requested_count",
        ]
        + list(SUMMARY_FIELDS + FLUENCY_FIELDS),
        rows,
    )
    counts = [
        "expected_runs",
        "recorded_runs",
        "missing_runs",
        "cosine_valid_runs",
        "cosine_invalid_runs",
        "fluency_valid_runs",
        "fluency_invalid_runs",
        "expected_experiments",
    ]
    macro_fields = [
        part
        for field in SUMMARY_FIELDS + FLUENCY_FIELDS
        for part in (field, f"{field}_experiments", f"{field}_complete_prompts")
    ]
    _write_csv(paths["summary"], list(GROUP_FIELDS) + counts + macro_fields, summaries)
    paired_fields = [
        f"frh_minus_subspace_{field}" for field in SUMMARY_FIELDS + FLUENCY_FIELDS
    ]
    _write_csv(
        paths["paired"],
        list(PAIR_GROUP_FIELDS)
        + [
            "kind",
            "experiment_id",
            "config_id",
            "prompt_id",
            "pair_present",
            "cosine_pair_valid",
            "fluency_pair_valid",
        ]
        + paired_fields,
        paired,
    )
    pair_counts = [
        "expected_pairs",
        "recorded_pairs",
        "missing_pairs",
        "present_pairs",
        "cosine_valid_pairs",
        "fluency_valid_pairs",
    ]
    pair_macro = [
        part
        for field in paired_fields
        for part in (field, f"{field}_experiments", f"{field}_complete_prompts")
    ]
    _write_csv(
        paths["paired_summary"],
        list(PAIR_GROUP_FIELDS) + pair_counts + pair_macro,
        paired_summaries,
    )
    _plot(rows, paths["cosine_ppl"], manifest["settings"]["lambdas"])
    _gallery(
        records,
        baseline,
        {id: {**manifest["prompts"][i], "index": i} for id, i in prompt_index.items()},
        paths["gallery"],
        status,
    )
    return paths
