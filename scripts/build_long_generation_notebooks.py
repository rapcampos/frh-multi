"""Build notebooks 53 and 54 from reviewable Python cell sources."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str, *, tags: list[str] | None = None):
    cell = nbf.v4.new_code_cell(dedent(source).strip() + "\n")
    if tags:
        cell.metadata["tags"] = tags
    return cell


def write_notebook(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbf.write(notebook, ROOT / name)


NB53 = [
    markdown(
        """
        # 53 — Long-generation composition (greedy tier)

        Tests whether blend, normalized score voting, and round-robin scheduling
        retain their two-concept effects over the 200-step horizon selected in
        `long-generation-inputs.yaml`. Each method is run at lambda 0 and 0.4.

        The notebook is safe by default: it resolves and prints the complete run
        matrix without loading model weights. Set `RUN_GENERATION = True` for a
        smoke run or provide the printed confirmation string for the full matrix.
        Existing JSONL logs are resumed by cell identity unless `OVERWRITE = True`.
        """
    ),
    code(
        """
        # Papermill parameters
        RUN_GENERATION = False
        SMOKE = False
        SMOKE_STEPS = 8
        OVERWRITE = False
        CONFIRM_FULL_RUN = ""
        RUN_BENCHMARK = False
        BENCHMARK_ONLY = False
        RUN_TRAJECTORIES = False
        """,
        tags=["parameters"],
    ),
    code(
        """
        import json
        import os
        import time
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import display

        from frames.evaluation.harness import EvaluationHarness
        from frames.experiments.long_generation import (
            generation_metadata,
            load_long_generation_experiment,
            matched_segment_fluency,
            token_bins,
        )
        from frames.experiments.notebook_helpers import llama_chat
        from frames.linalg.angles import vector_subspace_cosine
        from frames.representations import FrameUnembeddingRepresentation
        from frames.representations.concept import Concept
        from frames.representations.schedulers import RoundRobin

        CONFIG_PATH = Path("long-generation-inputs.yaml")
        EXPERIMENT = load_long_generation_experiment(CONFIG_PATH, "experiment_53")

        MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
        MODEL_KEY = "llama-3.1-8b-bf16"
        MIN_LEMMAS = 11
        MAX_TOKENS = 3
        K = 4
        METHODS = ("blend", "vote", "schedule")
        BIN_WIDTH = 20
        MIN_PPL_BIN_TOKENS = 10
        MIN_PPL_GROUP_FRACTION = 0.5
        PPL_TRAJECTORY_METRIC = "continuation-prefix-masked-v2"

        model_max_memory = None
        if raw_max_memory := os.getenv("FRH_MODEL_MAX_MEMORY"):
            model_max_memory = {
                int(device): limit
                for device, limit in json.loads(raw_max_memory).items()
            }
            print(f"model memory caps: {model_max_memory}")
        model_device_map = os.getenv("FRH_MODEL_DEVICE_MAP", "auto")
        print(f"model device map: {model_device_map}")

        prompts_by_id = {
            row["id"]: llama_chat(row["user"], row["assistant_prefix"])
            for row in EXPERIMENT.prompts
        }
        active_prompts = list(EXPERIMENT.prompts)
        active_pairs = list(EXPERIMENT.composition_pairs)
        active_lambdas = list(EXPERIMENT.lambdas)
        run_steps = EXPERIMENT.steps
        if SMOKE:
            active_prompts = active_prompts[:1]
            active_pairs = active_pairs[:1]
            run_steps = min(SMOKE_STEPS, EXPERIMENT.steps)

        full_cells = EXPERIMENT.guided_run_count(len(METHODS))
        active_cells = (
            len(active_prompts) * len(active_pairs) * len(METHODS) * len(active_lambdas)
        )
        confirmation = f"RUN_53_{full_cells}_CELLS"
        print(
            f"full matrix: {len(EXPERIMENT.composition_pairs)} pairs x "
            f"{len(EXPERIMENT.prompts)} prompts x {len(METHODS)} methods x "
            f"{len(EXPERIMENT.lambdas)} lambdas = {full_cells} guided continuations"
        )
        print(
            f"nominal full token budget: "
            f"{EXPERIMENT.output_token_budget(len(METHODS)):,} (early EOS excluded)"
        )
        print(
            f"this execution: {active_cells} cells at steps={run_steps}; "
            f"SMOKE={SMOKE}, RUN_GENERATION={RUN_GENERATION}"
        )
        print(f"full-run confirmation: {confirmation}")

        if (
            RUN_GENERATION
            and not SMOKE
            and not BENCHMARK_ONLY
            and CONFIRM_FULL_RUN != confirmation
        ):
            raise RuntimeError(
                "Full generation is locked. Set CONFIRM_FULL_RUN to the printed string."
            )
        if BENCHMARK_ONLY and not RUN_BENCHMARK:
            raise ValueError("BENCHMARK_ONLY requires RUN_BENCHMARK=True")
        if SMOKE_STEPS < 1:
            raise ValueError("SMOKE_STEPS must be positive")
        """
    ),
    code(
        """
        suffix = "_smoke" if SMOKE else ""
        LOG_PATH = Path(f"resources/53_long_composition{suffix}.jsonl")
        BASELINE_PATH = Path(f"resources/53_long_composition_baselines{suffix}.json")
        SUMMARY_PATH = Path(f"resources/53_long_composition_summary{suffix}.csv")
        TRAJECTORY_PATH = Path(
            f"resources/53_long_composition_trajectories{suffix}.csv"
        )
        TRAJECTORY_FIGURE_PATH = Path(
            f"resources/53_long_composition_trajectories{suffix}.png"
        )
        FIGURE_PATH = Path(f"resources/53_long_composition_metrics{suffix}.png")
        TIMING_PATH = Path("resources/53_long_composition_timing.csv")

        def read_records(path):
            if not path.exists() or not path.read_text().strip():
                return []
            return [json.loads(line) for line in path.read_text().splitlines() if line]


        def record_key(config):
            return (
                config["pair_id"],
                config["prompt_id"],
                config["method"],
                float(config["lam"]),
                int(config["steps"]),
            )


        if RUN_GENERATION and OVERWRITE:
            for path in (LOG_PATH, BASELINE_PATH, SUMMARY_PATH, TRAJECTORY_PATH):
                path.unlink(missing_ok=True)

        planned = {
            (pair["id"], prompt["id"], method, float(lam), run_steps)
            for pair in active_pairs
            for prompt in active_prompts
            for method in METHODS
            for lam in active_lambdas
        }
        completed = {record_key(row["config"]) for row in read_records(LOG_PATH)}
        unexpected = completed - planned
        if unexpected:
            raise RuntimeError(
                f"{LOG_PATH} contains {len(unexpected)} cells outside the resolved "
                "matrix; preserve it and start a new artifact or use OVERWRITE=True"
            )
        print(f"resume state: {len(completed)}/{active_cells} active cells already logged")
        """
    ),
    markdown(
        """
        ## Model, concepts, and shared baselines

        The model is loaded only for generation or the optional longitudinal
        analysis. Baselines are versioned by model, prompt text, and run horizon.
        A stale baseline cache is rejected instead of silently reused.
        """
    ),
    code(
        """
        need_model = RUN_GENERATION or RUN_TRAJECTORIES
        fur = harness = None
        concepts = {}
        baseline_texts_by_id = {}

        if need_model:
            fur = FrameUnembeddingRepresentation.from_model_id(
                id=MODEL_ID,
                device_map=model_device_map,
                torch_dtype=torch.bfloat16,
                max_memory=model_max_memory,
            )
            harness = EvaluationHarness(
                fur,
                LOG_PATH,
                min_lemmas_per_synset=MIN_LEMMAS,
                max_token_count=MAX_TOKENS,
            )

            used_synsets = sorted(
                {name for pair in active_pairs for name in (pair["a"], pair["b"])}
            )
            for name in used_synsets:
                cached = fur.get_concept_cached(name, MIN_LEMMAS, MAX_TOKENS)
                concepts[name] = Concept(
                    synset=cached.synset, tensor=cached.tensor.to(fur.model.device)
                )
            print(f"loaded {len(concepts)} cached concepts")

            need_baselines = (RUN_GENERATION and not BENCHMARK_ONLY) or RUN_TRAJECTORIES
            if need_baselines:
                baseline_spec = {
                    "model": MODEL_KEY,
                    "steps": run_steps,
                    "max_new_tokens": run_steps + 1,
                    "prompts": {
                        row["id"]: prompts_by_id[row["id"]]
                        for row in active_prompts
                    },
                }
                if BASELINE_PATH.exists():
                    baseline_payload = json.loads(BASELINE_PATH.read_text())
                    if baseline_payload.get("spec") != baseline_spec:
                        raise RuntimeError(
                            f"{BASELINE_PATH} does not match this run; use OVERWRITE=True"
                        )
                    baseline_texts_by_id = baseline_payload["texts"]
                    print(f"reused {len(baseline_texts_by_id)} baselines")
                elif RUN_GENERATION:
                    for prompt_row in active_prompts:
                        prompt_id = prompt_row["id"]
                        baseline_texts_by_id[prompt_id] = harness.generate_baseline(
                            [prompts_by_id[prompt_id]], max_new_tokens=run_steps + 1
                        )[0]
                    BASELINE_PATH.write_text(
                        json.dumps(
                            {"spec": baseline_spec, "texts": baseline_texts_by_id},
                            indent=2,
                            ensure_ascii=False,
                        )
                        + "\\n"
                    )
                    print(f"wrote {BASELINE_PATH}")
                else:
                    raise FileNotFoundError(
                        f"RUN_TRAJECTORIES requires an existing {BASELINE_PATH}"
                    )
        else:
            print("model load skipped")
        """
    ),
    code(
        """
        # Optional preflight benchmark. It is deliberately outside the main log.
        if RUN_GENERATION and RUN_BENCHMARK:
            pair = active_pairs[0]
            prompt_row = active_prompts[0]
            ca, cb = concepts[pair["a"]], concepts[pair["b"]]
            guide = Concept.average([ca, cb])
            benchmark_rows = []
            for benchmark_steps in (24, 50, 100):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                text, _ = fur.generate_with_topk_multi_guide(
                    [prompts_by_id[prompt_row["id"]]],
                    guides=[guide],
                    k=K,
                    steps=benchmark_steps,
                    fluency_weight=0.4,
                    batch_size=1,
                )
                elapsed = time.perf_counter() - started
                continuation = harness.continuation(
                    text[0], prompts_by_id[prompt_row["id"]]
                )
                meta = generation_metadata(
                    fur.model.tokenizer, continuation, benchmark_steps, elapsed
                )
                benchmark_rows.append(
                    {
                        "steps": benchmark_steps,
                        "elapsed_seconds": elapsed,
                        "seconds_per_output_token": elapsed / max(meta["token_count"], 1),
                        "continuation_tokens": meta["token_count"],
                        "peak_cuda_gib": (
                            torch.cuda.max_memory_allocated() / 2**30
                            if torch.cuda.is_available()
                            else float("nan")
                        ),
                    }
                )
            benchmark = pd.DataFrame(benchmark_rows)
            benchmark.to_csv(TIMING_PATH, index=False)
            display(benchmark.round(3))
        """
    ),
    markdown(
        """
        ## Guided run

        Each cell is generated and logged independently (`batch_size=1`). This is
        conservative for the long context and makes interruption/resume exact. The
        paper's greedy survivor rule is unchanged; only guide, score, or active
        concept differs across the three methods.
        """
    ),
    code(
        """
        FAMILY = {"blend": "frame", "vote": "score", "schedule": "time"}

        def make_methods(prompt, pair, lam):
            ca, cb = concepts[pair["a"]], concepts[pair["b"]]
            blend_guide = Concept.average([ca, cb])
            return {
                "blend": lambda: fur.generate_with_topk_multi_guide(
                    [prompt],
                    guides=[blend_guide],
                    k=K,
                    steps=run_steps,
                    fluency_weight=lam,
                    batch_size=1,
                ),
                "vote": lambda: fur.generate_with_topk_multi_guide(
                    [prompt],
                    guides=[ca, cb],
                    k=K,
                    steps=run_steps,
                    normalize="zscore",
                    fluency_weight=lam,
                    batch_size=1,
                ),
                "schedule": lambda: fur.generate_with_topk_multi_guide(
                    [prompt],
                    guides=[ca, cb],
                    k=K,
                    steps=run_steps,
                    scorer=RoundRobin(),
                    fluency_weight=lam,
                    batch_size=1,
                ),
            }


        if RUN_GENERATION and not BENCHMARK_ONLY:
            for pair in active_pairs:
                ca, cb = concepts[pair["a"]], concepts[pair["b"]]
                measured_rho = float(ca.rho(cb))
                for prompt_row in active_prompts:
                    prompt_id = prompt_row["id"]
                    prompt = prompts_by_id[prompt_id]
                    for lam in active_lambdas:
                        methods = make_methods(prompt, pair, lam)
                        for method in METHODS:
                            key = (pair["id"], prompt_id, method, float(lam), run_steps)
                            if key in completed:
                                continue
                            torch.cuda.empty_cache()
                            started = time.perf_counter()
                            texts, probe = methods[method]()
                            elapsed = time.perf_counter() - started
                            continuation = harness.continuation(texts[0], prompt)
                            metadata = generation_metadata(
                                fur.model.tokenizer, continuation, run_steps, elapsed
                            )
                            harness.evaluate(
                                [prompt],
                                texts,
                                [pair["a"], pair["b"]],
                                config={
                                    "experiment": "53",
                                    "model": MODEL_KEY,
                                    "pair_id": pair["id"],
                                    "prompt_id": prompt_id,
                                    "method": method,
                                    "family": FAMILY[method],
                                    "lam": float(lam),
                                    "a": pair["a"],
                                    "b": pair["b"],
                                    "rho_catalog": float(pair["rho"]),
                                    "rho_model": measured_rho,
                                    "k": K,
                                    "steps": run_steps,
                                    "survivor_rule": "greedy",
                                },
                                probe=probe,
                                baseline_texts=[baseline_texts_by_id[prompt_id]],
                                concepts={pair["a"]: ca, pair["b"]: cb},
                                angles=True,
                                metadata=[metadata],
                            )
                            completed.add(key)
                            print(
                                f"{len(completed):>3}/{active_cells} "
                                f"{pair['id']} / {prompt_id} / {method} / "
                                f"lambda={lam}: {metadata['token_count']} tokens, "
                                f"{elapsed:.1f}s"
                            )
        """
    ),
    markdown(
        """
        ## Whole-continuation summary

        Summary rows retain pair identity and aggregate over prompts. Positive joint
        values mean both concepts improved relative to the prompt-matched baseline.
        Length fields use actual re-tokenized continuations, exposing the historical
        one-seed-token offset and early EOS rather than assuming it away.
        """
    ),
    code(
        """
        parsed = read_records(LOG_PATH)
        result_rows = []
        for record in parsed:
            config = record["config"]
            expression = record["expression"]
            angles = record["angles"]
            da = expression[config["a"]]["delta"]
            db = expression[config["b"]]["delta"]
            ca = angles[config["a"]]["delta"]["token_cos"]
            cb = angles[config["b"]]["delta"]["token_cos"]
            metadata = record.get("metadata", {})
            result_rows.append(
                {
                    "pair_id": config["pair_id"],
                    "prompt_id": config["prompt_id"],
                    "a": config["a"],
                    "b": config["b"],
                    "rho": config["rho_model"],
                    "method": config["method"],
                    "lam": config["lam"],
                    "joint_expression": min(da, db),
                    "both_positive_expression": min(da, db) > 0,
                    "joint_token_cos": min(ca, cb),
                    "both_positive_token_cos": min(ca, cb) > 0,
                    "ppl_ratio": record["ppl_ratio"],
                    "continuation_tokens": metadata.get("token_count"),
                    "termination": metadata.get("termination"),
                    "distinct_token_ratio": metadata.get("distinct_token_ratio"),
                    "repeated_4gram_rate": metadata.get("repeated_ngram_rate"),
                    "max_repeated_span_tokens": metadata.get(
                        "max_repeated_span_tokens"
                    ),
                    "elapsed_seconds": metadata.get("elapsed_seconds"),
                }
            )

        results = pd.DataFrame(result_rows)
        summary = pd.DataFrame()
        if not results.empty:
            summary = (
                results.groupby(
                    ["pair_id", "a", "b", "rho", "method", "lam"],
                    dropna=False,
                )
                .agg(
                    joint_expression=("joint_expression", "mean"),
                    both_positive_expression=("both_positive_expression", "mean"),
                    joint_token_cos=("joint_token_cos", "mean"),
                    both_positive_token_cos=("both_positive_token_cos", "mean"),
                    ppl_ratio=("ppl_ratio", "mean"),
                    continuation_tokens=("continuation_tokens", "mean"),
                    distinct_token_ratio=("distinct_token_ratio", "mean"),
                    repeated_4gram_rate=("repeated_4gram_rate", "mean"),
                    max_repeated_span_tokens=("max_repeated_span_tokens", "mean"),
                    elapsed_seconds=("elapsed_seconds", "mean"),
                    n_prompts=("prompt_id", "nunique"),
                )
                .reset_index()
            )
            summary["observed_records"] = len(results)
            summary["expected_records"] = active_cells
            summary["complete"] = len(results) == active_cells
            summary.to_csv(SUMMARY_PATH, index=False)
            display(
                results.groupby(["method", "lam"])
                .agg(
                    joint_cos=("joint_token_cos", "mean"),
                    both_positive=("both_positive_token_cos", "mean"),
                    ppl_ratio=("ppl_ratio", "mean"),
                    repeated_4gram=("repeated_4gram_rate", "mean"),
                    tokens=("continuation_tokens", "mean"),
                    n=("pair_id", "size"),
                )
                .round(4)
            )
            print(f"wrote {SUMMARY_PATH}; completeness {len(results)}/{active_cells}")
        else:
            print(f"no records at {LOG_PATH}; generation remains disabled")
        """
    ),
    markdown(
        """
        ## Optional 20-token trajectories

        `RUN_TRAJECTORIES` computes bin-level FRH expression and token-to-subspace
        cosine from continuation hidden states. PPL v2 is computed from one aligned
        token-loss pass: every bin remains conditioned on its continuation prefix,
        loss is masked to the shared span, and tails shorter than
        `MIN_PPL_BIN_TOKENS` receive no PPL value. Group curves use the geometric
        mean ratio (mean log-ratio), not the unstable arithmetic mean of raw ratios,
        and require PPL coverage from at least `MIN_PPL_GROUP_FRACTION` of cells.
        This is separate from generation so the expensive analysis can be resumed.
        """
    ),
    code(
        """
        def token_analysis(text):
            tokens = fur.model.tokenize(text)
            with torch.inference_mode():
                hidden = fur.model.forward_last_hiden_state(tokens).squeeze(0)
            ids = tokens.squeeze(0).tolist()
            bos_id = fur.model.tokenizer.bos_token_id
            if ids and ids[0] == bos_id:
                ids = ids[1:]
                hidden = hidden[1:]
            nll_ids, token_nll = harness.continuation_token_nll(text)
            if ids != nll_ids:
                raise RuntimeError("hidden-state and token-NLL positions are misaligned")
            return ids, hidden, token_nll.tolist()


        def binned_values(ids, hidden, concept_map):
            bins = token_bins(ids, BIN_WIDTH)
            if not bins:
                return {}
            values = {part.index: {"bin": part} for part in bins}
            for name, concept in concept_map.items():
                projected = fur._project_hidden_states(
                    hidden.unsqueeze(0), concept
                ).reshape(len(ids), -1).float().mean(-1)
                cosine = vector_subspace_cosine(
                    hidden, concept.tensor.squeeze(0)
                ).float()
                for part in bins:
                    values[part.index][name] = {
                        "expression": float(projected[part.start : part.stop].mean()),
                        "token_cos": float(cosine[part.start : part.stop].mean()),
                    }
            return values


        trajectory_rows = []
        if RUN_TRAJECTORIES:
            baseline_analysis = {}
            for record_index, record in enumerate(parsed, start=1):
                config = record["config"]
                prompt_id = config["prompt_id"]
                concept_map = {
                    config["a"]: concepts[config["a"]],
                    config["b"]: concepts[config["b"]],
                }
                steered_ids, steered_hidden, steered_nll = token_analysis(
                    record["continuation"]
                )
                steered = binned_values(steered_ids, steered_hidden, concept_map)
                if prompt_id not in baseline_analysis:
                    baseline_continuation = harness.continuation(
                        baseline_texts_by_id[prompt_id], prompts_by_id[prompt_id]
                    )
                    baseline_analysis[prompt_id] = token_analysis(baseline_continuation)
                base_ids, base_hidden, baseline_nll = baseline_analysis[prompt_id]
                baseline = binned_values(base_ids, base_hidden, concept_map)

                for bin_index in sorted(set(steered) & set(baseline)):
                    sbin = steered[bin_index]["bin"]
                    bbin = baseline[bin_index]["bin"]
                    fluency = matched_segment_fluency(
                        steered_nll,
                        baseline_nll,
                        start=sbin.start,
                        stop=min(sbin.stop, bbin.stop),
                        min_shared_tokens=MIN_PPL_BIN_TOKENS,
                    )
                    row = {
                        "pair_id": config["pair_id"],
                        "prompt_id": prompt_id,
                        "method": config["method"],
                        "lam": config["lam"],
                        "bin": bin_index,
                        "start_token": sbin.start,
                        "stop_token": sbin.stop,
                        "baseline_stop_token": bbin.stop,
                        "shared_stop_token": sbin.start + fluency["shared_tokens"],
                        "coverage": fluency["shared_tokens"] / BIN_WIDTH,
                        "ppl_metric": PPL_TRAJECTORY_METRIC,
                        "min_ppl_bin_tokens": MIN_PPL_BIN_TOKENS,
                        **fluency,
                    }
                    for label, name in (("a", config["a"]), ("b", config["b"])):
                        row[f"{label}_expression_steered"] = steered[bin_index][name][
                            "expression"
                        ]
                        row[f"{label}_expression_baseline"] = baseline[bin_index][name][
                            "expression"
                        ]
                        row[f"{label}_expression_delta"] = (
                            row[f"{label}_expression_steered"]
                            - row[f"{label}_expression_baseline"]
                        )
                        row[f"{label}_token_cos_steered"] = steered[bin_index][name][
                            "token_cos"
                        ]
                        row[f"{label}_token_cos_baseline"] = baseline[bin_index][name][
                            "token_cos"
                        ]
                        row[f"{label}_token_cos_delta"] = (
                            row[f"{label}_token_cos_steered"]
                            - row[f"{label}_token_cos_baseline"]
                        )
                    row["joint_expression_delta"] = min(
                        row["a_expression_delta"], row["b_expression_delta"]
                    )
                    row["joint_token_cos_delta"] = min(
                        row["a_token_cos_delta"], row["b_token_cos_delta"]
                    )
                    if config["method"] == "schedule":
                        scored_positions = range(max(1, sbin.start), sbin.stop)
                        active = [(position - 1) % 2 for position in scored_positions]
                        row["schedule_a_fraction"] = (
                            active.count(0) / len(active) if active else float("nan")
                        )
                    else:
                        row["schedule_a_fraction"] = float("nan")
                    trajectory_rows.append(row)
                del steered_hidden
                if record_index % 10 == 0:
                    torch.cuda.empty_cache()
                    print(f"trajectory {record_index}/{len(parsed)}")

            trajectories = pd.DataFrame(trajectory_rows)
            trajectories.to_csv(TRAJECTORY_PATH, index=False)
            print(f"wrote {len(trajectories)} rows to {TRAJECTORY_PATH}")
            if not trajectories.empty:
                trajectory_summary = (
                    trajectories.groupby(["method", "lam", "bin"])
                    .agg(
                        joint_token_cos=("joint_token_cos_delta", "mean"),
                        joint_expression=("joint_expression_delta", "mean"),
                        segment_log_ppl_ratio=("segment_log_ppl_ratio", "mean"),
                        median_segment_ppl_ratio=("segment_ppl_ratio", "median"),
                        valid_ppl_cells=("segment_log_ppl_ratio", "count"),
                        trajectory_cells=("segment_log_ppl_ratio", "size"),
                        coverage=("coverage", "mean"),
                    )
                    .reset_index()
                )
                trajectory_summary["segment_ppl_ratio"] = np.exp(
                    trajectory_summary["segment_log_ppl_ratio"]
                )
                trajectory_summary["valid_ppl_fraction"] = (
                    trajectory_summary["valid_ppl_cells"]
                    / trajectory_summary["trajectory_cells"]
                )
                insufficient_ppl = (
                    trajectory_summary["valid_ppl_fraction"]
                    < MIN_PPL_GROUP_FRACTION
                )
                trajectory_summary.loc[
                    insufficient_ppl,
                    [
                        "segment_log_ppl_ratio",
                        "median_segment_ppl_ratio",
                        "segment_ppl_ratio",
                    ],
                ] = np.nan
                fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
                for axis, metric, ylabel in zip(
                    axes,
                    ("joint_token_cos", "joint_expression", "segment_ppl_ratio"),
                    (
                        "joint token-cosine delta",
                        "joint expression delta",
                        "geometric segment PPL ratio (prefix-conditioned)",
                    ),
                ):
                    for (method, lam), cell in trajectory_summary.groupby(
                        ["method", "lam"]
                    ):
                        axis.plot(
                            cell["bin"] * BIN_WIDTH,
                            cell[metric],
                            marker="o",
                            label=f"{method}, lambda={lam}",
                        )
                    axis.set_xlabel("continuation token position")
                    axis.set_ylabel(ylabel)
                axes[0].axhline(0, color="gray", linewidth=0.8)
                axes[1].axhline(0, color="gray", linewidth=0.8)
                axes[2].axhline(1, color="gray", linewidth=0.8)
                axes[0].legend(fontsize=7)
                plt.tight_layout()
                plt.savefig(TRAJECTORY_FIGURE_PATH, dpi=150)
                plt.show()

                final_bins = (
                    trajectories.sort_values("bin")
                    .groupby(["pair_id", "prompt_id", "method", "lam"])
                    .tail(1)
                    .groupby(["pair_id", "method", "lam"])
                    .agg(
                        final_joint_expression=("joint_expression_delta", "mean"),
                        final_joint_token_cos=("joint_token_cos_delta", "mean"),
                        final_bin_coverage=("coverage", "mean"),
                    )
                    .reset_index()
                )
                final_ppl_bins = (
                    trajectories.dropna(subset=["segment_log_ppl_ratio"])
                    .sort_values("bin")
                    .groupby(["pair_id", "prompt_id", "method", "lam"])
                    .tail(1)
                    .groupby(["pair_id", "method", "lam"])
                    .agg(
                        final_segment_log_ppl_ratio=(
                            "segment_log_ppl_ratio",
                            "mean",
                        ),
                        final_ppl_bin=("bin", "mean"),
                        final_ppl_bin_coverage=("coverage", "mean"),
                        final_valid_ppl_cells=("segment_log_ppl_ratio", "count"),
                    )
                    .reset_index()
                )
                final_ppl_bins["final_segment_ppl_ratio"] = np.exp(
                    final_ppl_bins["final_segment_log_ppl_ratio"]
                )
                summary = summary.merge(
                    final_bins, on=["pair_id", "method", "lam"], how="left"
                )
                summary = summary.merge(
                    final_ppl_bins, on=["pair_id", "method", "lam"], how="left"
                )
                summary["ppl_trajectory_metric"] = PPL_TRAJECTORY_METRIC
                summary["min_ppl_bin_tokens"] = MIN_PPL_BIN_TOKENS
                summary["min_ppl_group_fraction"] = MIN_PPL_GROUP_FRACTION
                summary.to_csv(SUMMARY_PATH, index=False)
        """
    ),
    code(
        """
        if not results.empty:
            plot_summary = (
                results.groupby(["method", "lam"])
                .agg(
                    joint_token_cos=("joint_token_cos", "mean"),
                    ppl_ratio=("ppl_ratio", "mean"),
                    repeated_4gram_rate=("repeated_4gram_rate", "mean"),
                )
                .reset_index()
            )
            methods = list(METHODS)
            colors = {"blend": "tab:blue", "vote": "tab:orange", "schedule": "tab:green"}
            fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
            for axis, metric, ylabel in zip(
                axes,
                ("joint_token_cos", "ppl_ratio", "repeated_4gram_rate"),
                ("joint token-cosine delta", "continuation PPL ratio", "repeated 4-gram rate"),
            ):
                for method in methods:
                    cell = plot_summary[plot_summary["method"] == method].sort_values(
                        "lam"
                    )
                    axis.plot(
                        cell["lam"],
                        cell[metric],
                        marker="o",
                        label=method,
                        color=colors[method],
                    )
                axis.set_xlabel("fluency weight lambda")
                axis.set_ylabel(ylabel)
            axes[0].axhline(0, color="gray", linewidth=0.8)
            axes[1].axhline(2.5, color="gray", linestyle=":", linewidth=0.8)
            axes[0].legend()
            plt.tight_layout()
            plt.savefig(FIGURE_PATH, dpi=150)
            plt.show()

            qualitative = []
            for record in parsed:
                config = record["config"]
                qualitative.append(
                    {
                        "pair": config["pair_id"],
                        "prompt": config["prompt_id"],
                        "method": config["method"],
                        "lambda": config["lam"],
                        "continuation": record["continuation"],
                    }
                )
            display(pd.DataFrame(qualitative))
        """
    ),
]


NB54 = [
    markdown(
        """
        # 54 — Woman/man differential steering (greedy tier)

        Evaluates both FRH gender directions (`woman − man` and `man − woman`)
        over every prompt in `long-generation-inputs.yaml` at the established
        24-step horizon. The positive-only control separates attraction from
        repulsion; frame difference is the paper operation, and normalized
        signed score voting is the score-space control.

        This is a descriptive steering audit, not a normative comparison of
        genders. Qualitative outputs require explicit stereotype review.
        Generation is disabled by default and existing logs resume by cell key.
        """
    ),
    code(
        """
        # Papermill parameters
        RUN_GENERATION = False
        SMOKE = False
        SMOKE_STEPS = 8
        OVERWRITE = False
        CONFIRM_FULL_RUN = ""
        RUN_TRAJECTORIES = False
        """,
        tags=["parameters"],
    ),
    code(
        """
        import json
        import os
        import time
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import display

        from frames.evaluation.harness import EvaluationHarness
        from frames.experiments.long_generation import (
            generation_metadata,
            load_long_generation_experiment,
            matched_segment_fluency,
            token_bins,
        )
        from frames.experiments.notebook_helpers import llama_chat
        from frames.linalg.angles import vector_subspace_cosine
        from frames.representations import FrameUnembeddingRepresentation
        from frames.representations.concept import Concept

        CONFIG_PATH = Path("long-generation-inputs.yaml")
        EXPERIMENT = load_long_generation_experiment(CONFIG_PATH, "experiment_54")

        MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
        MODEL_KEY = "llama-3.1-8b-bf16"
        MIN_LEMMAS = 11
        MAX_TOKENS = 3
        K = 4
        METHODS = ("attract-only", "frame-diff", "score-neg")
        BIN_WIDTH = 20
        MIN_PPL_BIN_TOKENS = 10
        MIN_PPL_GROUP_FRACTION = 0.5
        PPL_TRAJECTORY_METRIC = "continuation-prefix-masked-v2"

        model_max_memory = None
        if raw_max_memory := os.getenv("FRH_MODEL_MAX_MEMORY"):
            model_max_memory = {
                int(device): limit
                for device, limit in json.loads(raw_max_memory).items()
            }
            print(f"model memory caps: {model_max_memory}")
        model_device_map = os.getenv("FRH_MODEL_DEVICE_MAP", "auto")
        print(f"model device map: {model_device_map}")

        prompts_by_id = {
            row["id"]: llama_chat(row["user"], row["assistant_prefix"])
            for row in EXPERIMENT.prompts
        }
        active_prompts = list(EXPERIMENT.prompts)
        active_directions = list(EXPERIMENT.negative_directions)
        active_lambdas = list(EXPERIMENT.lambdas)
        run_steps = EXPERIMENT.steps
        if SMOKE:
            active_prompts = active_prompts[:1]
            active_directions = active_directions[:1]
            run_steps = min(SMOKE_STEPS, EXPERIMENT.steps)

        full_cells = EXPERIMENT.guided_run_count(len(METHODS))
        active_cells = (
            len(active_prompts)
            * len(active_directions)
            * len(METHODS)
            * len(active_lambdas)
        )
        confirmation = f"RUN_54_{full_cells}_CELLS"
        print(
            f"full matrix: {len(EXPERIMENT.negative_directions)} directions x "
            f"{len(EXPERIMENT.prompts)} prompts x {len(METHODS)} guided methods x "
            f"{len(EXPERIMENT.lambdas)} lambdas = {full_cells} guided continuations"
        )
        print(
            f"nominal full token budget: "
            f"{EXPERIMENT.output_token_budget(len(METHODS)):,} (early EOS excluded)"
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
        """
        suffix = "_smoke" if SMOKE else ""
        LOG_PATH = Path(f"resources/54_long_negative{suffix}.jsonl")
        BASELINE_PATH = Path(f"resources/54_long_negative_baselines{suffix}.json")
        SUMMARY_PATH = Path(f"resources/54_long_negative_summary{suffix}.csv")
        TRAJECTORY_PATH = Path(f"resources/54_long_negative_trajectories{suffix}.csv")
        TRAJECTORY_FIGURE_PATH = Path(
            f"resources/54_long_negative_trajectories{suffix}.png"
        )
        FIGURE_PATH = Path(f"resources/54_long_negative_quadrants{suffix}.png")

        def read_records(path):
            if not path.exists() or not path.read_text().strip():
                return []
            return [json.loads(line) for line in path.read_text().splitlines() if line]


        def record_key(config):
            return (
                config["direction_id"],
                config["prompt_id"],
                config["method"],
                float(config["lam"]),
                int(config["steps"]),
            )


        if RUN_GENERATION and OVERWRITE:
            for path in (LOG_PATH, BASELINE_PATH, SUMMARY_PATH, TRAJECTORY_PATH):
                path.unlink(missing_ok=True)

        planned = {
            (direction["id"], prompt["id"], method, float(lam), run_steps)
            for direction in active_directions
            for prompt in active_prompts
            for method in METHODS
            for lam in active_lambdas
        }
        completed = {record_key(row["config"]) for row in read_records(LOG_PATH)}
        unexpected = completed - planned
        if unexpected:
            raise RuntimeError(
                f"{LOG_PATH} contains {len(unexpected)} cells outside the resolved "
                "matrix; preserve it and start a new artifact or use OVERWRITE=True"
            )
        print(f"resume state: {len(completed)}/{active_cells} active cells already logged")
        """
    ),
    markdown(
        """
        ## Model, concepts, and shared baselines

        `prompt_ids: all` is resolved in catalog order before this cell. Baselines
        are generated once per prompt and shared across both steering directions.
        """
    ),
    code(
        """
        need_model = RUN_GENERATION or RUN_TRAJECTORIES
        fur = harness = None
        concepts = {}
        baseline_texts_by_id = {}

        if need_model:
            fur = FrameUnembeddingRepresentation.from_model_id(
                id=MODEL_ID,
                device_map=model_device_map,
                torch_dtype=torch.bfloat16,
                max_memory=model_max_memory,
            )
            harness = EvaluationHarness(
                fur,
                LOG_PATH,
                min_lemmas_per_synset=MIN_LEMMAS,
                max_token_count=MAX_TOKENS,
            )

            used_synsets = sorted(
                {
                    name
                    for direction in active_directions
                    for name in (direction["attract"], direction["repel"])
                }
            )
            for name in used_synsets:
                cached = fur.get_concept_cached(name, MIN_LEMMAS, MAX_TOKENS)
                concepts[name] = Concept(
                    synset=cached.synset, tensor=cached.tensor.to(fur.model.device)
                )
            print(f"loaded {len(concepts)} cached concepts")

            baseline_spec = {
                "model": MODEL_KEY,
                "steps": run_steps,
                "max_new_tokens": run_steps + 1,
                "prompts": {row["id"]: prompts_by_id[row["id"]] for row in active_prompts},
            }
            if BASELINE_PATH.exists():
                baseline_payload = json.loads(BASELINE_PATH.read_text())
                if baseline_payload.get("spec") != baseline_spec:
                    raise RuntimeError(
                        f"{BASELINE_PATH} does not match this run; use OVERWRITE=True"
                    )
                baseline_texts_by_id = baseline_payload["texts"]
                print(f"reused {len(baseline_texts_by_id)} baselines")
            elif RUN_GENERATION:
                for prompt_row in active_prompts:
                    prompt_id = prompt_row["id"]
                    baseline_texts_by_id[prompt_id] = harness.generate_baseline(
                        [prompts_by_id[prompt_id]], max_new_tokens=run_steps + 1
                    )[0]
                BASELINE_PATH.write_text(
                    json.dumps(
                        {"spec": baseline_spec, "texts": baseline_texts_by_id},
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\\n"
                )
                print(f"wrote {BASELINE_PATH}")
            else:
                raise FileNotFoundError(
                    f"RUN_TRAJECTORIES requires an existing {BASELINE_PATH}"
                )
        else:
            print("model load skipped")
        """
    ),
    markdown(
        """
        ## Guided run

        All three guided conditions use the same lambda from YAML. `frame-diff`
        applies the FRH Procrustes projection of attract minus repel. `score-neg`
        retains separate guides and normalizes each concept over the candidate pool
        before applying weights `[+1, -1]`.
        """
    ),
    code(
        """
        FAMILY = {
            "attract-only": "positive-control",
            "frame-diff": "frame",
            "score-neg": "score",
        }

        def make_methods(prompt, direction, lam):
            attract = concepts[direction["attract"]]
            repel = concepts[direction["repel"]]
            differential = attract - repel
            return {
                "attract-only": lambda: fur.generate_with_topk_multi_guide(
                    [prompt],
                    guides=[attract],
                    k=K,
                    steps=run_steps,
                    fluency_weight=lam,
                    batch_size=1,
                ),
                "frame-diff": lambda: fur.generate_with_topk_multi_guide(
                    [prompt],
                    guides=[differential],
                    k=K,
                    steps=run_steps,
                    fluency_weight=lam,
                    batch_size=1,
                ),
                "score-neg": lambda: fur.generate_with_topk_multi_guide(
                    [prompt],
                    guides=[attract, repel],
                    weights=[1.0, -1.0],
                    k=K,
                    steps=run_steps,
                    normalize="zscore",
                    fluency_weight=lam,
                    batch_size=1,
                ),
            }


        if RUN_GENERATION:
            for direction in active_directions:
                attract = concepts[direction["attract"]]
                repel = concepts[direction["repel"]]
                measured_rho = float(attract.rho(repel))
                for prompt_row in active_prompts:
                    prompt_id = prompt_row["id"]
                    prompt = prompts_by_id[prompt_id]
                    for lam in active_lambdas:
                        methods = make_methods(prompt, direction, lam)
                        for method in METHODS:
                            key = (
                                direction["id"],
                                prompt_id,
                                method,
                                float(lam),
                                run_steps,
                            )
                            if key in completed:
                                continue
                            torch.cuda.empty_cache()
                            started = time.perf_counter()
                            texts, probe = methods[method]()
                            elapsed = time.perf_counter() - started
                            continuation = harness.continuation(texts[0], prompt)
                            metadata = generation_metadata(
                                fur.model.tokenizer, continuation, run_steps, elapsed
                            )
                            harness.evaluate(
                                [prompt],
                                texts,
                                [],  # presence is not an E54 endpoint; avoids cuDF here
                                config={
                                    "experiment": "54",
                                    "model": MODEL_KEY,
                                    "direction_id": direction["id"],
                                    "prompt_id": prompt_id,
                                    "method": method,
                                    "family": FAMILY[method],
                                    "lam": float(lam),
                                    "attract": direction["attract"],
                                    "repel": direction["repel"],
                                    "rho_model": measured_rho,
                                    "k": K,
                                    "steps": run_steps,
                                    "survivor_rule": "greedy",
                                },
                                probe=probe,
                                baseline_texts=[baseline_texts_by_id[prompt_id]],
                                concepts={
                                    direction["attract"]: attract,
                                    direction["repel"]: repel,
                                },
                                angles=True,
                                metadata=[metadata],
                            )
                            completed.add(key)
                            print(
                                f"{len(completed):>2}/{active_cells} "
                                f"{direction['id']} / {prompt_id} / {method}: "
                                f"{metadata['token_count']} tokens, {elapsed:.1f}s"
                            )
        """
    ),
    markdown(
        """
        ## Whole-continuation results and controls

        Clean success is deliberately strict: the attracted concept must rise and
        the repelled concept must fall relative to the same unsteered baseline.
        Synthetic `unsteered` rows (zero deltas, PPL ratio 1) are included in the
        aggregate table; the raw JSONL remains one record per guided cell.
        """
    ),
    code(
        """
        parsed = read_records(LOG_PATH)
        result_rows = []
        for record in parsed:
            config = record["config"]
            expression = record["expression"]
            angles = record["angles"]
            delta_attract = expression[config["attract"]]["delta"]
            delta_repel = expression[config["repel"]]["delta"]
            cos_attract = angles[config["attract"]]["delta"]["token_cos"]
            cos_repel = angles[config["repel"]]["delta"]["token_cos"]
            metadata = record.get("metadata", {})
            result_rows.append(
                {
                    "direction_id": config["direction_id"],
                    "prompt_id": config["prompt_id"],
                    "method": config["method"],
                    "lam": config["lam"],
                    "attract": config["attract"],
                    "repel": config["repel"],
                    "delta_attract": delta_attract,
                    "delta_repel": delta_repel,
                    "net_expression": delta_attract - delta_repel,
                    "clean_expression": delta_attract > 0 and delta_repel < 0,
                    "cos_attract": cos_attract,
                    "cos_repel": cos_repel,
                    "net_token_cos": cos_attract - cos_repel,
                    "clean_token_cos": cos_attract > 0 and cos_repel < 0,
                    "ppl_ratio": record["ppl_ratio"],
                    "continuation_tokens": metadata.get("token_count"),
                    "termination": metadata.get("termination"),
                    "distinct_token_ratio": metadata.get("distinct_token_ratio"),
                    "repeated_4gram_rate": metadata.get("repeated_ngram_rate"),
                    "max_repeated_span_tokens": metadata.get(
                        "max_repeated_span_tokens"
                    ),
                    "elapsed_seconds": metadata.get("elapsed_seconds"),
                }
            )

        results = pd.DataFrame(result_rows)
        summary = pd.DataFrame()
        analysis_results = results.copy()
        if not results.empty:
            baseline_rows = []
            for direction in active_directions:
                for prompt_row in active_prompts:
                    baseline_rows.append(
                        {
                            "direction_id": direction["id"],
                            "prompt_id": prompt_row["id"],
                            "method": "unsteered",
                            "lam": active_lambdas[0],
                            "attract": direction["attract"],
                            "repel": direction["repel"],
                            "delta_attract": 0.0,
                            "delta_repel": 0.0,
                            "net_expression": 0.0,
                            "clean_expression": False,
                            "cos_attract": 0.0,
                            "cos_repel": 0.0,
                            "net_token_cos": 0.0,
                            "clean_token_cos": False,
                            "ppl_ratio": 1.0,
                        }
                    )
            analysis_results = pd.concat(
                [results, pd.DataFrame(baseline_rows)], ignore_index=True
            )
            summary = (
                analysis_results.groupby(
                    ["direction_id", "attract", "repel", "method", "lam"],
                    dropna=False,
                )
                .agg(
                    delta_attract=("delta_attract", "mean"),
                    delta_repel=("delta_repel", "mean"),
                    net_expression=("net_expression", "mean"),
                    clean_expression=("clean_expression", "mean"),
                    cos_attract=("cos_attract", "mean"),
                    cos_repel=("cos_repel", "mean"),
                    net_token_cos=("net_token_cos", "mean"),
                    clean_token_cos=("clean_token_cos", "mean"),
                    ppl_ratio=("ppl_ratio", "mean"),
                    n_prompts=("prompt_id", "nunique"),
                )
                .reset_index()
            )
            summary["observed_guided_records"] = len(results)
            summary["expected_guided_records"] = active_cells
            summary["complete"] = len(results) == active_cells
            summary.to_csv(SUMMARY_PATH, index=False)
            display(summary.round(4))
            print(f"wrote {SUMMARY_PATH}; completeness {len(results)}/{active_cells}")
        else:
            print(f"no records at {LOG_PATH}; generation remains disabled")
        """
    ),
    code(
        """
        if not results.empty:
            paired = results.pivot_table(
                index=["direction_id", "prompt_id"],
                columns="method",
                values="delta_repel",
                aggfunc="first",
            )
            suppression_rows = []
            for method in ("frame-diff", "score-neg"):
                comparison = paired[method] - paired["attract-only"]
                for direction_id, values in comparison.groupby(level="direction_id"):
                    suppression_rows.append(
                        {
                            "direction_id": direction_id,
                            "method": method,
                            "prompts_more_suppressed": int((values < 0).sum()),
                            "n_prompts": int(values.notna().sum()),
                            "mean_delta_repel_vs_attract_only": values.mean(),
                        }
                    )
            suppression = pd.DataFrame(suppression_rows)
            print("Paired suppression relative to the positive-only control:")
            display(suppression.round(4))

            symmetry = summary[summary["method"] != "unsteered"].pivot_table(
                index="method",
                columns="direction_id",
                values=["net_expression", "clean_expression", "net_token_cos"],
            )
            print("Direction asymmetry (directions remain separate):")
            display(symmetry.round(4))
        """
    ),
    markdown(
        """
        ## Optional 20-token trajectories

        At 24 steps this normally yields one full 20-token bin and one short final
        bin. Absolute steered/baseline values, coverage, attraction, repulsion, and
        clean success are retained. PPL v2 remains conditioned on the continuation
        prefix, masks loss to the shared bin, omits tails shorter than
        `MIN_PPL_BIN_TOKENS`, and is aggregated geometrically. No missing tokens are
        padded into either continuation. Group PPL points require valid values from
        at least `MIN_PPL_GROUP_FRACTION` of trajectory cells.
        """
    ),
    code(
        """
        def token_analysis(text):
            tokens = fur.model.tokenize(text)
            with torch.inference_mode():
                hidden = fur.model.forward_last_hiden_state(tokens).squeeze(0)
            ids = tokens.squeeze(0).tolist()
            bos_id = fur.model.tokenizer.bos_token_id
            if ids and ids[0] == bos_id:
                ids = ids[1:]
                hidden = hidden[1:]
            nll_ids, token_nll = harness.continuation_token_nll(text)
            if ids != nll_ids:
                raise RuntimeError("hidden-state and token-NLL positions are misaligned")
            return ids, hidden, token_nll.tolist()


        def binned_values(ids, hidden, concept_map):
            bins = token_bins(ids, BIN_WIDTH)
            if not bins:
                return {}
            values = {part.index: {"bin": part} for part in bins}
            for name, concept in concept_map.items():
                projected = fur._project_hidden_states(
                    hidden.unsqueeze(0), concept
                ).reshape(len(ids), -1).float().mean(-1)
                cosine = vector_subspace_cosine(
                    hidden, concept.tensor.squeeze(0)
                ).float()
                for part in bins:
                    values[part.index][name] = {
                        "expression": float(projected[part.start : part.stop].mean()),
                        "token_cos": float(cosine[part.start : part.stop].mean()),
                    }
            return values


        trajectory_rows = []
        if RUN_TRAJECTORIES:
            baseline_analysis = {}
            for record_index, record in enumerate(parsed, start=1):
                config = record["config"]
                prompt_id = config["prompt_id"]
                concept_map = {
                    config["attract"]: concepts[config["attract"]],
                    config["repel"]: concepts[config["repel"]],
                }
                steered_ids, steered_hidden, steered_nll = token_analysis(
                    record["continuation"]
                )
                steered = binned_values(steered_ids, steered_hidden, concept_map)
                if prompt_id not in baseline_analysis:
                    baseline_continuation = harness.continuation(
                        baseline_texts_by_id[prompt_id], prompts_by_id[prompt_id]
                    )
                    baseline_analysis[prompt_id] = token_analysis(baseline_continuation)
                base_ids, base_hidden, baseline_nll = baseline_analysis[prompt_id]
                baseline = binned_values(base_ids, base_hidden, concept_map)

                for bin_index in sorted(set(steered) & set(baseline)):
                    sbin = steered[bin_index]["bin"]
                    bbin = baseline[bin_index]["bin"]
                    fluency = matched_segment_fluency(
                        steered_nll,
                        baseline_nll,
                        start=sbin.start,
                        stop=min(sbin.stop, bbin.stop),
                        min_shared_tokens=MIN_PPL_BIN_TOKENS,
                    )
                    attract = config["attract"]
                    repel = config["repel"]
                    attr_expr = (
                        steered[bin_index][attract]["expression"]
                        - baseline[bin_index][attract]["expression"]
                    )
                    repel_expr = (
                        steered[bin_index][repel]["expression"]
                        - baseline[bin_index][repel]["expression"]
                    )
                    attr_cos = (
                        steered[bin_index][attract]["token_cos"]
                        - baseline[bin_index][attract]["token_cos"]
                    )
                    repel_cos = (
                        steered[bin_index][repel]["token_cos"]
                        - baseline[bin_index][repel]["token_cos"]
                    )
                    trajectory_rows.append(
                        {
                            "direction_id": config["direction_id"],
                            "prompt_id": prompt_id,
                            "method": config["method"],
                            "lam": config["lam"],
                            "bin": bin_index,
                            "start_token": sbin.start,
                            "stop_token": sbin.stop,
                            "baseline_stop_token": bbin.stop,
                            "shared_stop_token": sbin.start + fluency["shared_tokens"],
                            "coverage": fluency["shared_tokens"] / BIN_WIDTH,
                            "attract_expression_steered": steered[bin_index][attract][
                                "expression"
                            ],
                            "attract_expression_baseline": baseline[bin_index][attract][
                                "expression"
                            ],
                            "attract_expression_delta": attr_expr,
                            "repel_expression_steered": steered[bin_index][repel][
                                "expression"
                            ],
                            "repel_expression_baseline": baseline[bin_index][repel][
                                "expression"
                            ],
                            "repel_expression_delta": repel_expr,
                            "net_expression": attr_expr - repel_expr,
                            "clean_expression": attr_expr > 0 and repel_expr < 0,
                            "attract_token_cos_delta": attr_cos,
                            "repel_token_cos_delta": repel_cos,
                            "net_token_cos": attr_cos - repel_cos,
                            "clean_token_cos": attr_cos > 0 and repel_cos < 0,
                            "ppl_metric": PPL_TRAJECTORY_METRIC,
                            "min_ppl_bin_tokens": MIN_PPL_BIN_TOKENS,
                            **fluency,
                        }
                    )
                del steered_hidden
                if record_index % 10 == 0:
                    torch.cuda.empty_cache()
                    print(f"trajectory {record_index}/{len(parsed)}")

            trajectories = pd.DataFrame(trajectory_rows)
            trajectories.to_csv(TRAJECTORY_PATH, index=False)
            print(f"wrote {len(trajectories)} rows to {TRAJECTORY_PATH}")
            if not trajectories.empty:
                trajectory_summary = (
                    trajectories.groupby(["direction_id", "method", "bin"])
                    .agg(
                        net_expression=("net_expression", "mean"),
                        clean_expression=("clean_expression", "mean"),
                        net_token_cos=("net_token_cos", "mean"),
                        segment_log_ppl_ratio=("segment_log_ppl_ratio", "mean"),
                        median_segment_ppl_ratio=("segment_ppl_ratio", "median"),
                        valid_ppl_cells=("segment_log_ppl_ratio", "count"),
                        trajectory_cells=("segment_log_ppl_ratio", "size"),
                        coverage=("coverage", "mean"),
                    )
                    .reset_index()
                )
                trajectory_summary["segment_ppl_ratio"] = np.exp(
                    trajectory_summary["segment_log_ppl_ratio"]
                )
                trajectory_summary["valid_ppl_fraction"] = (
                    trajectory_summary["valid_ppl_cells"]
                    / trajectory_summary["trajectory_cells"]
                )
                insufficient_ppl = (
                    trajectory_summary["valid_ppl_fraction"]
                    < MIN_PPL_GROUP_FRACTION
                )
                trajectory_summary.loc[
                    insufficient_ppl,
                    [
                        "segment_log_ppl_ratio",
                        "median_segment_ppl_ratio",
                        "segment_ppl_ratio",
                    ],
                ] = np.nan
                fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
                for axis, metric, ylabel in zip(
                    axes,
                    ("net_expression", "clean_expression", "segment_ppl_ratio"),
                    (
                        "net expression",
                        "clean-success rate",
                        "geometric segment PPL ratio (prefix-conditioned)",
                    ),
                ):
                    for (direction_id, method), cell in trajectory_summary.groupby(
                        ["direction_id", "method"]
                    ):
                        axis.plot(
                            cell["bin"] * BIN_WIDTH,
                            cell[metric],
                            marker="o",
                            label=f"{direction_id}: {method}",
                        )
                    axis.set_xlabel("continuation token position")
                    axis.set_ylabel(ylabel)
                axes[0].axhline(0, color="gray", linewidth=0.8)
                axes[2].axhline(1, color="gray", linewidth=0.8)
                axes[0].legend(fontsize=7)
                plt.tight_layout()
                plt.savefig(TRAJECTORY_FIGURE_PATH, dpi=150)
                plt.show()

                final_bins = (
                    trajectories.sort_values("bin")
                    .groupby(["direction_id", "prompt_id", "method", "lam"])
                    .tail(1)
                    .groupby(["direction_id", "method", "lam"])
                    .agg(
                        final_net_expression=("net_expression", "mean"),
                        final_clean_expression=("clean_expression", "mean"),
                        final_net_token_cos=("net_token_cos", "mean"),
                        final_bin_coverage=("coverage", "mean"),
                    )
                    .reset_index()
                )
                final_ppl_bins = (
                    trajectories.dropna(subset=["segment_log_ppl_ratio"])
                    .sort_values("bin")
                    .groupby(["direction_id", "prompt_id", "method", "lam"])
                    .tail(1)
                    .groupby(["direction_id", "method", "lam"])
                    .agg(
                        final_segment_log_ppl_ratio=(
                            "segment_log_ppl_ratio",
                            "mean",
                        ),
                        final_ppl_bin=("bin", "mean"),
                        final_ppl_bin_coverage=("coverage", "mean"),
                        final_valid_ppl_cells=("segment_log_ppl_ratio", "count"),
                    )
                    .reset_index()
                )
                final_ppl_bins["final_segment_ppl_ratio"] = np.exp(
                    final_ppl_bins["final_segment_log_ppl_ratio"]
                )
                summary = summary.merge(
                    final_bins, on=["direction_id", "method", "lam"], how="left"
                )
                summary = summary.merge(
                    final_ppl_bins,
                    on=["direction_id", "method", "lam"],
                    how="left",
                )
                summary["ppl_trajectory_metric"] = PPL_TRAJECTORY_METRIC
                summary["min_ppl_bin_tokens"] = MIN_PPL_BIN_TOKENS
                summary["min_ppl_group_fraction"] = MIN_PPL_GROUP_FRACTION
                summary.to_csv(SUMMARY_PATH, index=False)
        """
    ),
    code(
        """
        if not results.empty:
            plot_rows = summary[summary["method"] != "unsteered"]
            directions = list(plot_rows["direction_id"].unique())
            fig, axes = plt.subplots(1, len(directions), figsize=(6 * len(directions), 4.8))
            if len(directions) == 1:
                axes = [axes]
            colors = {
                "attract-only": "tab:gray",
                "frame-diff": "tab:blue",
                "score-neg": "tab:orange",
            }
            for axis, direction_id in zip(axes, directions):
                cell = plot_rows[plot_rows["direction_id"] == direction_id]
                for _, row in cell.iterrows():
                    axis.scatter(
                        row["delta_repel"],
                        row["delta_attract"],
                        s=90,
                        color=colors[row["method"]],
                        label=row["method"],
                    )
                    axis.annotate(
                        row["method"],
                        (row["delta_repel"], row["delta_attract"]),
                        xytext=(5, 3),
                        textcoords="offset points",
                        fontsize=8,
                    )
                axis.axhline(0, color="gray", linewidth=0.8)
                axis.axvline(0, color="gray", linewidth=0.8)
                axis.set_title(direction_id)
                axis.set_xlabel("repelled-concept expression delta (lower is better)")
            axes[0].set_ylabel("attracted-concept expression delta (higher is better)")
            plt.tight_layout()
            plt.savefig(FIGURE_PATH, dpi=150)
            plt.show()

            qualitative = []
            for record in parsed:
                config = record["config"]
                qualitative.append(
                    {
                        "direction": config["direction_id"],
                        "prompt": config["prompt_id"],
                        "method": config["method"],
                        "continuation": record["continuation"],
                        "stereotype_review": "",
                    }
                )
            print(
                "Review outputs for stereotypes and asymmetric social claims; "
                "leave the review column descriptive."
            )
            display(pd.DataFrame(qualitative))
        """
    ),
]


if __name__ == "__main__":
    write_notebook("53_long_generation_composition.ipynb", NB53)
    write_notebook("54_long_generation_negative.ipynb", NB54)
