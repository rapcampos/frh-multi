"""Build notebook 55, the signed gender-composition qualitative experiment."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "55_signed_gender_composition.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str, *, parameters: bool = False):
    cell = nbf.v4.new_code_cell(source.strip())
    if parameters:
        cell.metadata["tags"] = ["parameters"]
    return cell


cells = [
    markdown(
        r"""
# 55 — Signed gender-concept composition samples

Generates a bounded qualitative audit over the three requested prompts and
the `man` / `woman` concept chains. Positive chains are compared with two
representations of each differential direction:

1. **atomic signed** — keep `man`, `-woman`, and the positive concepts as
   separate guides;
2. **differential preblend** — first construct `polar(man - woman)`, then use
   that differential frame as one guide beside the positive concepts.

Both directions run through all three thesis composition families: extrinsic
frame blending, z-score-normalized score voting, and round-robin scheduling.
The atomic schedule uses `SignedRoundRobin`, so a repulsive turn negates the
active concept score without shrinking it relative to the fluency term.

The full matrix is 42 guide configurations x 3 prompts x 3 methods x 2
fluency weights = **756 continuations**. Generation is disabled by default,
full execution requires an explicit confirmation string, and JSONL records
resume by cell key. Outputs are descriptive steering evidence, not a claim
about gender or social fairness; qualitative rows retain an empty manual
`stereotype_review` field.
"""
    ),
    code(
        r"""
# Papermill parameters
RUN_GENERATION = False
SMOKE = False
SMOKE_STEPS = 8
OVERWRITE = False
CONFIRM_FULL_RUN = ""
""",
        parameters=True,
    ),
    code(
        r"""
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

from frames.evaluation.harness import EvaluationHarness, RecordingScorer
from frames.experiments.long_generation import generation_metadata
from frames.experiments.notebook_helpers import llama_chat
from frames.representations import FrameUnembeddingRepresentation
from frames.representations.aggregators import weighted_sum
from frames.representations.concept import Concept
from frames.representations.schedulers import RoundRobin, SignedRoundRobin

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_KEY = "llama-3.1-8b-bf16"
MIN_LEMMAS = 11
MAX_TOKENS = 3
K = 4
STEPS = 24
LAMBDAS = (0.0, 0.4)
METHODS = ("blend", "vote", "schedule")

PROMPT_ROWS = (
    {"id": "men", "user": "What men can be?", "assistant_prefix": "1."},
    {"id": "women", "user": "What women can be?", "assistant_prefix": "1."},
    {"id": "story", "user": "Tell me a short story.", "assistant_prefix": "Once"},
)

SYNSET = {
    "man": "man.n.01",
    "woman": "woman.n.01",
    "education": "education.n.01",
    "music": "music.n.01",
    "war": "war.n.01",
    "universe": "universe.n.01",
    "joy": "joy.n.01",
    "sorrow": "sorrow.n.01",
}
REPLACEMENTS = ("music", "war", "universe")


def term(name, sign=1):
    return {"synset": SYNSET[name], "sign": int(sign)}


def concept_head(synset):
    return synset.split(".")[0]


GROUPS = []


def add_positive(names):
    semantic_id = "__".join(names)
    terms = [term(name) for name in names]
    GROUPS.append(
        {
            "group_id": f"{semantic_id}__positive",
            "semantic_id": semantic_id,
            "construction": "positive",
            "direction_id": None,
            "attract": None,
            "repel": None,
            "extras": [row["synset"] for row in terms],
            "replacement": next((name for name in names if name in REPLACEMENTS), None),
            "terms": terms,
            "n_semantic": len(terms),
            "n_guides": len(terms),
        }
    )


def add_direction(direction_id, attract, repel, extras):
    suffix = "__".join(extras) if extras else "direction_only"
    semantic_id = f"{direction_id}__{suffix}"
    terms = [term(attract, 1), term(repel, -1), *[term(name, 1) for name in extras]]
    for construction in ("atomic_signed", "differential_preblend"):
        GROUPS.append(
            {
                "group_id": f"{semantic_id}__{construction}",
                "semantic_id": semantic_id,
                "construction": construction,
                "direction_id": direction_id,
                "attract": SYNSET[attract],
                "repel": SYNSET[repel],
                "extras": [SYNSET[name] for name in extras],
                "replacement": next(
                    (name for name in extras if name in REPLACEMENTS), None
                ),
                "terms": terms,
                "n_semantic": len(terms),
                "n_guides": len(terms) if construction == "atomic_signed" else 1 + len(extras),
            }
        )


for person in ("man", "woman"):
    add_positive([person])
    add_positive([person, "education"])
    for replacement in REPLACEMENTS:
        add_positive([person, "education", replacement])
        for affect in ("joy", "sorrow"):
            add_positive([person, "education", replacement, affect])

for direction_id, attract, repel in (
    ("man_minus_woman", "man", "woman"),
    ("woman_minus_man", "woman", "man"),
):
    add_direction(direction_id, attract, repel, [])
    add_direction(direction_id, attract, repel, ["education"])
    for replacement in REPLACEMENTS:
        add_direction(direction_id, attract, repel, ["education", replacement])

assert len(GROUPS) == 42
assert Counter(group["construction"] for group in GROUPS) == {
    "positive": 22,
    "atomic_signed": 10,
    "differential_preblend": 10,
}
assert len({group["group_id"] for group in GROUPS}) == len(GROUPS)

active_groups = list(GROUPS)
active_prompts = list(PROMPT_ROWS)
active_lambdas = list(LAMBDAS)
run_steps = STEPS
if SMOKE:
    smoke_ids = {
        "man_minus_woman__education__atomic_signed",
        "man_minus_woman__education__differential_preblend",
    }
    active_groups = [group for group in GROUPS if group["group_id"] in smoke_ids]
    active_prompts = active_prompts[:1]
    active_lambdas = [0.4]
    run_steps = min(SMOKE_STEPS, STEPS)

full_cells = len(GROUPS) * len(PROMPT_ROWS) * len(METHODS) * len(LAMBDAS)
active_cells = (
    len(active_groups) * len(active_prompts) * len(METHODS) * len(active_lambdas)
)
confirmation = f"RUN_55_{full_cells}_CELLS"
print(
    f"full matrix: {len(GROUPS)} configurations x {len(PROMPT_ROWS)} prompts x "
    f"{len(METHODS)} methods x {len(LAMBDAS)} lambdas = {full_cells} continuations"
)
print(
    f"this execution: {active_cells} cells at steps={run_steps}; "
    f"SMOKE={SMOKE}, RUN_GENERATION={RUN_GENERATION}"
)
print(f"full-run confirmation: {confirmation}")

if RUN_GENERATION and not SMOKE and CONFIRM_FULL_RUN != confirmation:
    raise RuntimeError(
        "Full generation is locked. Set CONFIRM_FULL_RUN to the printed string."
    )
if SMOKE_STEPS < 1:
    raise ValueError("SMOKE_STEPS must be positive")
"""
    ),
    code(
        r"""
suffix = "_smoke" if SMOKE else ""
LOG_PATH = Path(f"resources/55_signed_composition{suffix}.jsonl")
BASELINE_PATH = Path(f"resources/55_signed_composition_baselines{suffix}.json")
METRICS_PATH = Path(f"resources/55_signed_composition_metrics{suffix}.csv")
SUMMARY_PATH = Path(f"resources/55_signed_composition_summary{suffix}.csv")
OVERVIEW_PATH = Path(f"resources/55_signed_composition_overview{suffix}.csv")
COMPARISON_PATH = Path(f"resources/55_signed_composition_comparison{suffix}.csv")
GALLERY_PATH = Path(f"resources/55_signed_composition_gallery{suffix}.csv")
FIGURE_PATH = Path(f"resources/55_signed_composition_tradeoff{suffix}.png")


def read_records(path):
    if not path.exists() or not path.read_text().strip():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def record_key(config):
    return (
        config["group_id"],
        config["prompt_id"],
        config["method"],
        float(config["lam"]),
        int(config["steps"]),
    )


generated_paths = (
    LOG_PATH,
    BASELINE_PATH,
    METRICS_PATH,
    SUMMARY_PATH,
    OVERVIEW_PATH,
    COMPARISON_PATH,
    GALLERY_PATH,
    FIGURE_PATH,
)
if RUN_GENERATION and OVERWRITE:
    for path in generated_paths:
        path.unlink(missing_ok=True)

planned = {
    (group["group_id"], prompt["id"], method, float(lam), run_steps)
    for group in active_groups
    for prompt in active_prompts
    for method in METHODS
    for lam in active_lambdas
}
existing = read_records(LOG_PATH)
existing_keys = [record_key(record["config"]) for record in existing]
if len(existing_keys) != len(set(existing_keys)):
    duplicates = [key for key, count in Counter(existing_keys).items() if count > 1]
    raise RuntimeError(f"duplicate JSONL cell keys: {duplicates[:5]}")
completed = set(existing_keys) & planned
print(f"resume state: {len(completed)}/{len(planned)} planned cells complete")
"""
    ),
    markdown(
        r"""
## Model, baselines, and concept frames

The three unsteered baselines are shared by every condition. Evaluation uses
the original eight WordNet concepts even when generation uses a preblended
differential guide, so attraction and repulsion remain separately observable.
"""
    ),
    code(
        r"""
fur = None
harness = None
concepts = {}
prompts_by_id = {
    row["id"]: llama_chat(row["user"], row["assistant_prefix"])
    for row in active_prompts
}

model_max_memory = None
if raw_max_memory := os.getenv("FRH_MODEL_MAX_MEMORY"):
    model_max_memory = {
        int(device): limit for device, limit in json.loads(raw_max_memory).items()
    }
model_device_map = os.getenv("FRH_MODEL_DEVICE_MAP", "auto")

if RUN_GENERATION:
    load_kwargs = {
        "id": MODEL_ID,
        "device_map": model_device_map,
        "torch_dtype": torch.bfloat16,
    }
    if model_max_memory is not None:
        load_kwargs["max_memory"] = model_max_memory
    fur = FrameUnembeddingRepresentation.from_model_id(**load_kwargs)
    harness = EvaluationHarness(
        fur,
        LOG_PATH,
        min_lemmas_per_synset=MIN_LEMMAS,
        max_token_count=MAX_TOKENS,
    )

    needed = sorted({row["synset"] for group in active_groups for row in group["terms"]})
    for name in needed:
        cached = fur.get_concept_cached(name, MIN_LEMMAS, MAX_TOKENS)
        concepts[name] = Concept(
            synset=cached.synset,
            tensor=cached.tensor.to(fur.model.device),
        )
    print(f"{len(concepts)} concept frames built")

    if BASELINE_PATH.exists() and not OVERWRITE:
        baseline_artifact = json.loads(BASELINE_PATH.read_text())
    else:
        prompt_ids = [row["id"] for row in active_prompts]
        prompt_texts = [prompts_by_id[prompt_id] for prompt_id in prompt_ids]
        baseline_texts = harness.generate_baseline(
            prompt_texts, max_new_tokens=run_steps + 2
        )
        baseline_artifact = {
            "config": {
                "model": MODEL_KEY,
                "steps": run_steps,
                "greedy": True,
                "smoke": bool(SMOKE),
            },
            "prompts": {
                prompt_id: {
                    "prompt": prompt,
                    "text": text,
                    "continuation": harness.continuation(text, prompt),
                }
                for prompt_id, prompt, text in zip(prompt_ids, prompt_texts, baseline_texts)
            },
        }
        BASELINE_PATH.write_text(json.dumps(baseline_artifact, indent=1))
    baseline_texts_by_id = {
        prompt_id: row["text"] for prompt_id, row in baseline_artifact["prompts"].items()
    }
else:
    baseline_artifact = (
        json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {"prompts": {}}
    )
    baseline_texts_by_id = {
        prompt_id: row["text"] for prompt_id, row in baseline_artifact["prompts"].items()
    }
    print("model load skipped")
"""
    ),
    markdown(
        r"""
## Guided generation

For atomic signed groups, blend receives signed frame weights, vote receives
L1-normalized signed weights, and schedule receives signs through
`SignedRoundRobin`. For preblended groups, the differential frame is built
first and the outer three families treat it as one positive guide. Per-step
score traces are recorded after any candidate-pool normalization.
"""
    ),
    code(
        r"""
FAMILY = {"blend": "frame", "vote": "score", "schedule": "time"}


class RecorderInputView:
    '''Expose one row of a batched RecordingScorer to EvaluationHarness.'''

    def __init__(self, recorder, index):
        self.recorder = recorder
        self.index = index

    def per_input_steps(self, index):
        if index != 0:
            raise IndexError(index)
        return self.recorder.per_input_steps(self.index)


def resolve_guides(group):
    if group["construction"] == "differential_preblend":
        differential = concepts[group["attract"]] - concepts[group["repel"]]
        guides = [differential, *[concepts[name] for name in group["extras"]]]
        signs = [1.0] * len(guides)
        labels = [f"[{concept_head(group['attract'])}-{concept_head(group['repel'])}]", *[
            concept_head(name) for name in group["extras"]
        ]]
        return guides, signs, labels

    guides = [concepts[row["synset"]] for row in group["terms"]]
    signs = [float(row["sign"]) for row in group["terms"]]
    labels = [
        ("-" if row["sign"] < 0 else "") + concept_head(row["synset"])
        for row in group["terms"]
    ]
    return guides, signs, labels


def generate(group, method, prompt_texts, lam):
    guides, signs, labels = resolve_guides(group)
    common = {
        "k": K,
        "steps": run_steps,
        "fluency_weight": float(lam),
        "batch_size": len(prompt_texts),
    }
    if method == "blend":
        composed = Concept.average(guides, weights=signs)
        recorder = RecordingScorer(weighted_sum)
        texts, probe = fur.generate_with_topk_multi_guide(
            prompt_texts,
            guides=[composed],
            scorer=recorder,
            **common,
        )
        method_weights = [1.0]
        trace_labels = ["blend(" + ", ".join(labels) + ")"]
    elif method == "vote":
        scale = sum(abs(sign) for sign in signs)
        method_weights = [sign / scale for sign in signs]
        recorder = RecordingScorer(weighted_sum)
        texts, probe = fur.generate_with_topk_multi_guide(
            prompt_texts,
            guides=guides,
            weights=method_weights,
            scorer=recorder,
            normalize="zscore",
            **common,
        )
        trace_labels = labels
    elif method == "schedule":
        scheduler = (
            SignedRoundRobin(signs) if any(sign < 0 for sign in signs) else RoundRobin()
        )
        recorder = RecordingScorer(scheduler)
        texts, probe = fur.generate_with_topk_multi_guide(
            prompt_texts,
            guides=guides,
            scorer=recorder,
            **common,
        )
        method_weights = signs
        trace_labels = labels
    else:
        raise ValueError(method)
    return texts, probe, recorder, method_weights, trace_labels


if RUN_GENERATION:
    for group in active_groups:
        evaluation_names = list(dict.fromkeys(row["synset"] for row in group["terms"]))
        evaluation_concepts = {name: concepts[name] for name in evaluation_names}
        for lam in active_lambdas:
            for method in METHODS:
                missing_rows = [
                    row
                    for row in active_prompts
                    if (
                        group["group_id"],
                        row["id"],
                        method,
                        float(lam),
                        run_steps,
                    )
                    not in completed
                ]
                if not missing_rows:
                    continue
                prompt_ids = [row["id"] for row in missing_rows]
                prompt_texts = [prompts_by_id[prompt_id] for prompt_id in prompt_ids]
                torch.cuda.empty_cache()
                started = time.perf_counter()
                texts, probe, recorder, method_weights, trace_labels = generate(
                    group, method, prompt_texts, lam
                )
                elapsed = time.perf_counter() - started
                per_prompt_elapsed = elapsed / len(prompt_ids)
                metadata = [
                    generation_metadata(
                        fur.model.tokenizer,
                        harness.continuation(text, prompt),
                        run_steps,
                        per_prompt_elapsed,
                    )
                    for text, prompt in zip(texts, prompt_texts)
                ]
                common_config = {
                    "experiment": "55",
                    "model": MODEL_KEY,
                    "group_id": group["group_id"],
                    "semantic_id": group["semantic_id"],
                    "construction": group["construction"],
                    "direction_id": group["direction_id"],
                    "attract": group["attract"],
                    "repel": group["repel"],
                    "replacement": group["replacement"],
                    "terms": group["terms"],
                    "n_semantic": group["n_semantic"],
                    "n_guides": group["n_guides"],
                    "method": method,
                    "family": FAMILY[method],
                    "method_weights": method_weights,
                    "trace_labels": trace_labels,
                    "normalize": "zscore" if method == "vote" else None,
                    "lam": float(lam),
                    "k": K,
                    "steps": run_steps,
                    "survivor_rule": "greedy",
                }
                for index, prompt_id in enumerate(prompt_ids):
                    harness.evaluate(
                        [prompt_texts[index]],
                        [texts[index]],
                        evaluation_names,
                        config={**common_config, "prompt_id": prompt_id},
                        probe=probe[index : index + 1],
                        recorder=RecorderInputView(recorder, index),
                        baseline_texts=[baseline_texts_by_id[prompt_id]],
                        concepts=evaluation_concepts,
                        angles=True,
                        metadata=[metadata[index]],
                    )
                    key = (
                        group["group_id"],
                        prompt_id,
                        method,
                        float(lam),
                        run_steps,
                    )
                    completed.add(key)
                print(
                    f"{len(completed):>3}/{len(planned)} {group['group_id']} / "
                    f"{method} / lambda={lam}: {elapsed:.1f}s"
                )
"""
    ),
    markdown(
        r"""
## Whole-continuation metrics

Every raw per-concept measurement is retained as JSON inside the record-level
CSV. Aggregate signed metrics multiply repelled-concept deltas by `-1`, so a
positive value consistently means movement in the requested direction. The
strict joint metric is the minimum signed delta over all semantic concepts.
"""
    ),
    code(
        r"""
parsed = read_records(LOG_PATH)


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def signed_map(values, terms):
    return {row["synset"]: row["sign"] * values[row["synset"]] for row in terms}


metric_rows = []
for record in parsed:
    config = record["config"]
    terms = config["terms"]
    expression = record["expression"]
    angles = record["angles"]
    expression_steered = {name: row["steered"] for name, row in expression.items()}
    expression_baseline = {name: row["baseline"] for name, row in expression.items()}
    expression_delta = {name: row["delta"] for name, row in expression.items()}
    signed_expression = signed_map(expression_delta, terms)

    angle_maps = {}
    signed_angles = {}
    for metric in ("token_cos", "geodesic", "mcc"):
        angle_maps[metric] = {
            "steered": {name: row["steered"][metric] for name, row in angles.items()},
            "baseline": {name: row["baseline"][metric] for name, row in angles.items()},
            "delta": {name: row["delta"][metric] for name, row in angles.items()},
        }
        signed_angles[metric] = signed_map(angle_maps[metric]["delta"], terms)

    metadata = record.get("metadata", {})
    metric_rows.append(
        {
            "group_id": config["group_id"],
            "semantic_id": config["semantic_id"],
            "construction": config["construction"],
            "direction_id": config["direction_id"],
            "replacement": config["replacement"] or "none",
            "prompt_id": config["prompt_id"],
            "method": config["method"],
            "family": config["family"],
            "lam": config["lam"],
            "n_semantic": config["n_semantic"],
            "n_guides": config["n_guides"],
            "terms": json_cell(terms),
            "trace_labels": json_cell(config["trace_labels"]),
            "expression_steered": json_cell(expression_steered),
            "expression_baseline": json_cell(expression_baseline),
            "expression_delta": json_cell(expression_delta),
            "signed_expression": json_cell(signed_expression),
            "joint_signed_expression": min(signed_expression.values()),
            "mean_signed_expression": np.mean(list(signed_expression.values())),
            "all_signed_expression": min(signed_expression.values()) > 0,
            "token_cos_steered": json_cell(angle_maps["token_cos"]["steered"]),
            "token_cos_baseline": json_cell(angle_maps["token_cos"]["baseline"]),
            "token_cos_delta": json_cell(angle_maps["token_cos"]["delta"]),
            "signed_token_cos": json_cell(signed_angles["token_cos"]),
            "joint_signed_token_cos": min(signed_angles["token_cos"].values()),
            "mean_signed_token_cos": np.mean(list(signed_angles["token_cos"].values())),
            "all_signed_token_cos": min(signed_angles["token_cos"].values()) > 0,
            "geodesic_steered": json_cell(angle_maps["geodesic"]["steered"]),
            "geodesic_baseline": json_cell(angle_maps["geodesic"]["baseline"]),
            "geodesic_delta": json_cell(angle_maps["geodesic"]["delta"]),
            "signed_geodesic": json_cell(signed_angles["geodesic"]),
            "joint_signed_geodesic": min(signed_angles["geodesic"].values()),
            "all_signed_geodesic": min(signed_angles["geodesic"].values()) > 0,
            "mcc_steered": json_cell(angle_maps["mcc"]["steered"]),
            "mcc_baseline": json_cell(angle_maps["mcc"]["baseline"]),
            "mcc_delta": json_cell(angle_maps["mcc"]["delta"]),
            "signed_mcc": json_cell(signed_angles["mcc"]),
            "joint_signed_mcc": min(signed_angles["mcc"].values()),
            "all_signed_mcc": min(signed_angles["mcc"].values()) > 0,
            "presence": json_cell(record["success"]),
            "ppl": record["ppl"],
            "baseline_ppl": record["baseline_ppl"],
            "ppl_ratio": record["ppl_ratio"],
            "fluency_flag": record["fluency_flag"],
            "probe_final": record["probe_final"],
            "score_trace_steps": len(record["steps"] or []),
            "continuation_tokens": metadata.get("token_count"),
            "termination": metadata.get("termination"),
            "distinct_token_ratio": metadata.get("distinct_token_ratio"),
            "repeated_4gram_rate": metadata.get("repeated_ngram_rate"),
            "max_repeated_span_tokens": metadata.get("max_repeated_span_tokens"),
            "elapsed_seconds": metadata.get("elapsed_seconds"),
            "continuation": record["continuation"],
            "text": record["text"],
            "stereotype_review": "",
        }
    )

metrics = pd.DataFrame(metric_rows)
summary = pd.DataFrame()
overview = pd.DataFrame()
comparison = pd.DataFrame()
if not metrics.empty:
    observed = set(record_key(record["config"]) for record in parsed)
    metrics["observed_records"] = len(observed & planned)
    metrics["expected_records"] = len(planned)
    metrics["complete"] = planned <= observed
    metrics.to_csv(METRICS_PATH, index=False)

    group_columns = [
        "group_id",
        "semantic_id",
        "construction",
        "direction_id",
        "replacement",
        "n_semantic",
        "n_guides",
        "method",
        "family",
        "lam",
    ]
    summary = (
        metrics.groupby(group_columns, dropna=False)
        .agg(
            joint_signed_expression=("joint_signed_expression", "mean"),
            mean_signed_expression=("mean_signed_expression", "mean"),
            all_signed_expression=("all_signed_expression", "mean"),
            joint_signed_token_cos=("joint_signed_token_cos", "mean"),
            mean_signed_token_cos=("mean_signed_token_cos", "mean"),
            all_signed_token_cos=("all_signed_token_cos", "mean"),
            joint_signed_geodesic=("joint_signed_geodesic", "mean"),
            all_signed_geodesic=("all_signed_geodesic", "mean"),
            joint_signed_mcc=("joint_signed_mcc", "mean"),
            all_signed_mcc=("all_signed_mcc", "mean"),
            ppl_ratio=("ppl_ratio", "mean"),
            ppl_ratio_geomean=(
                "ppl_ratio",
                lambda values: float(np.exp(np.log(values).mean())),
            ),
            fluency_flag_rate=("fluency_flag", "mean"),
            distinct_token_ratio=("distinct_token_ratio", "mean"),
            repeated_4gram_rate=("repeated_4gram_rate", "mean"),
            n_prompts=("prompt_id", "nunique"),
            n_records=("prompt_id", "size"),
        )
        .reset_index()
    )
    summary["observed_records"] = len(observed & planned)
    summary["expected_records"] = len(planned)
    summary["complete"] = planned <= observed
    summary.to_csv(SUMMARY_PATH, index=False)

    overview = (
        metrics.groupby(
            ["construction", "n_semantic", "n_guides", "method", "family", "lam"],
            dropna=False,
        )
        .agg(
            joint_signed_expression=("joint_signed_expression", "mean"),
            all_signed_expression=("all_signed_expression", "mean"),
            joint_signed_token_cos=("joint_signed_token_cos", "mean"),
            all_signed_token_cos=("all_signed_token_cos", "mean"),
            joint_signed_geodesic=("joint_signed_geodesic", "mean"),
            joint_signed_mcc=("joint_signed_mcc", "mean"),
            ppl_ratio=("ppl_ratio", "mean"),
            distinct_token_ratio=("distinct_token_ratio", "mean"),
            repeated_4gram_rate=("repeated_4gram_rate", "mean"),
            n_records=("prompt_id", "size"),
        )
        .reset_index()
    )
    overview.to_csv(OVERVIEW_PATH, index=False)

    signed_summary = summary[summary["construction"] != "positive"]
    comparison_metrics = [
        "joint_signed_expression",
        "joint_signed_token_cos",
        "all_signed_expression",
        "all_signed_token_cos",
        "ppl_ratio",
    ]
    comparison = signed_summary.set_index(
        [
            "semantic_id",
            "direction_id",
            "replacement",
            "method",
            "lam",
            "construction",
        ]
    )[comparison_metrics].unstack("construction")
    comparison.columns = [f"{metric}__{construction}" for metric, construction in comparison]
    comparison = comparison.reset_index()
    expected_comparisons = (
        len({group["semantic_id"] for group in GROUPS if group["direction_id"]})
        * len(METHODS)
        * len(LAMBDAS)
    )
    if not SMOKE:
        assert len(comparison) == expected_comparisons
        assert not comparison.filter(regex="__(atomic_signed|differential_preblend)$").isna().any().any()
    for metric in (
        "joint_signed_expression",
        "joint_signed_token_cos",
        "all_signed_expression",
        "all_signed_token_cos",
        "ppl_ratio",
    ):
        comparison[f"{metric}__preblend_minus_atomic"] = (
            comparison[f"{metric}__differential_preblend"]
            - comparison[f"{metric}__atomic_signed"]
        )
    comparison.to_csv(COMPARISON_PATH, index=False)

    gallery = metrics[
        [
            "group_id",
            "semantic_id",
            "construction",
            "direction_id",
            "replacement",
            "prompt_id",
            "method",
            "lam",
            "joint_signed_expression",
            "joint_signed_token_cos",
            "ppl_ratio",
            "continuation",
            "stereotype_review",
        ]
    ].sort_values(["semantic_id", "construction", "prompt_id", "method", "lam"])
    gallery.to_csv(GALLERY_PATH, index=False)

    print(
        f"wrote record metrics {METRICS_PATH}, group summary {SUMMARY_PATH}, "
        f"overview {OVERVIEW_PATH}, paired comparison {COMPARISON_PATH}, and "
        f"gallery {GALLERY_PATH}"
    )
    print(f"completeness: {len(observed & planned)}/{len(planned)}")
    display(overview.round(4))
else:
    print(f"no records at {LOG_PATH}; generation remains disabled or incomplete")
"""
    ),
    markdown(
        r"""
## Trade-off view and qualitative gallery

The plot aggregates only signed conditions and keeps construction, method,
and lambda visible. The displayed gallery focuses on the requested
`education + music` example in both directions; the saved gallery contains
every continuation and an empty manual-review column.
"""
    ),
    code(
        r"""
if not metrics.empty:
    signed = metrics[metrics["construction"] != "positive"]
    plot_rows = (
        signed.groupby(["construction", "method", "lam"], dropna=False)
        .agg(
            joint_expression=("joint_signed_expression", "mean"),
            joint_cos=("joint_signed_token_cos", "mean"),
            ppl_ratio=("ppl_ratio", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics_to_plot = (
        ("joint_expression", "mean signed joint expression"),
        ("joint_cos", "mean signed joint token cosine"),
        ("ppl_ratio", "mean perplexity ratio"),
    )
    colors = {"blend": "tab:blue", "vote": "tab:orange", "schedule": "tab:green"}
    markers = {"atomic_signed": "o", "differential_preblend": "s"}
    for axis, (metric, label) in zip(axes, metrics_to_plot):
        for _, row in plot_rows.iterrows():
            axis.scatter(
                row["lam"],
                row[metric],
                color=colors[row["method"]],
                marker=markers[row["construction"]],
                s=70,
                label=f"{row['method']} / {row['construction']}",
            )
        axis.set_xticks(list(LAMBDAS))
        axis.set_xlabel("fluency weight lambda")
        axis.set_ylabel(label)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), fontsize=7)
    plt.suptitle("Signed composition: atomic signs vs differential preblend")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.show()

    focus_ids = {
        "man_minus_woman__education__music",
        "woman_minus_man__education__music",
    }
    focus = metrics[
        metrics["semantic_id"].isin(focus_ids)
        & (metrics["lam"] == 0.4)
    ][
        [
            "semantic_id",
            "construction",
            "prompt_id",
            "method",
            "joint_signed_expression",
            "joint_signed_token_cos",
            "ppl_ratio",
            "continuation",
            "stereotype_review",
        ]
    ].sort_values(["semantic_id", "construction", "prompt_id", "method"])
    print(
        "Manual review required: describe occupational, familial, personality, "
        "agency, and competence stereotypes before making qualitative claims."
    )
    display(focus)
"""
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
nbf.write(notebook, OUTPUT)
print(f"wrote {OUTPUT}")
