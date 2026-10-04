"""NB68: matched FRH-blend/subspace battery, with immutable resume identity.

The manifest/dry-run path is CPU-only. Model imports are delayed until execution.
Run ``python -m frames.experiments.steering_comparison --dry-run`` to inspect it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .comparison_manifest import build_manifest, stable_hash

QUESTIONS = [
    ("Tell me a short story.", "Once"),
    ("Describe your favorite place.", "My favorite place is"),
    ("What do you like to do on weekends?", "I like"),
    ("Describe the weather today.", "Today the weather is"),
    ("What should I cook for dinner?", "You should"),
    ("Write a paragraph about the city.", "The city"),
    ("Continue this thought.", "Here is what happened next:"),
    ("Describe a person you met.", "The person I met was"),
]
METHODS = ("frh_blend", "subspace")
SOURCES = ("wordnet", "custom")


def make_experiment(concepts_dir="concepts", *, smoke=False, normalize_score=False):
    """Freeze words, configurations, prompts, numerical recipe and source code.

    ``normalize_score`` switches subspace steering to the norm-normalized
    alignment score. The key is only added to the settings when enabled, so
    manifests frozen before the option existed still validate unchanged.
    """
    manifest = build_manifest(concepts_dir)
    settings = {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "quantization": "bf16",
        "survivor_rule": "greedy",
        "k": 4,
        "steps": 24,
        "baseline_max_new_tokens": 25,  # initial candidate + 24 expansions
        "max_word_tokens": 4,
        "subspace_rank": 8,
        "min_lemmas": 2,
        "wordnet_languages": "Llama",
        "lambdas": [0.0, 0.4],
        "seed": 0,
        "smoke": smoke,
        "metric_version": "shared-span-cosine-v1",
    }
    if normalize_score:
        settings["subspace_normalize_score"] = True
    configs = manifest["configs"]
    if smoke:
        # Retain all four counts, both signed and positive construction, both directions.
        chosen = []
        for count in range(1, 5):
            group = [c for c in configs if c["unit_count"] == count]
            chosen.append(next(c for c in group if not c["negative"]))
            signed = next(c for c in group if c["negative"])
            chosen.extend(
                c for c in group if c["experiment_id"] == signed["experiment_id"]
            )
        configs = chosen
    prompts = []
    for i, (user, prefix) in enumerate(QUESTIONS[:2] if smoke else QUESTIONS):
        text = (
            f"<|start_header_id|>user<|end_header_id|>{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>{prefix}"
        )
        prompts.append(
            {"id": f"p{i + 1:02d}", "user": user, "prefix": prefix, "text": text}
        )
    root = Path(__file__).resolve().parents[2]
    code_files = [
        "frames/experiments/comparison_manifest.py",
        "frames/experiments/comparison_metrics.py",
        "frames/experiments/steering_comparison.py",
        "frames/representations/concept.py",
        "frames/representations/frame.py",
        "frames/representations/unembedding.py",
        "frames/models/hf/base.py",
        "frames/models/hf/llm.py",
        "frames/nlp/synsets.py",
        "frames/evaluation/harness.py",
        "subspace/concept.py",
        "subspace/representation.py",
        "subspace/generation.py",
    ]
    manifest.update(
        settings=settings,
        configs=configs,
        prompts=prompts,
        methods=list(METHODS),
        sources=list(SOURCES),
        implementation_hashes={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in code_files
        },
    )
    return manifest


def matrix_size(manifest):
    return (
        len(manifest["configs"])
        * len(manifest["prompts"])
        * len(manifest["methods"])
        * len(manifest["sources"])
        * len(manifest["settings"]["lambdas"])
    )


def write_json(path, value):
    """Atomically replace a derived JSON artifact; reject non-JSON floats."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(
                f"Incompatible existing artifact: {path}; use a new output prefix"
            )
    else:
        write_json(path, value)


def read_records(path):
    """Fail on corruption/duplicate IDs rather than silently accepting lost runs."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    seen = set()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["run_id"] in seen:
            raise ValueError(f"Duplicate run ID in {path}")
        seen.add(record["run_id"])
        records.append(record)
    return records


def run_battery(manifest, backend, output_prefix="resources/68_comparison"):
    """Execute/resume a frozen matrix through a model or test backend.

    Backend contracts: provenance dict; baseline(prompts)->continuations;
    generate(config, source, method, lambda, prompts)->continuations;
    measure(continuation, baseline, config)->JSON-compatible metric dictionary.
    Fixed whole-prompt batches are regenerated after an interrupted partial batch.
    """
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    freeze_json(f"{prefix}_manifest.json", manifest)
    freeze_json(f"{prefix}_construction.json", backend.provenance)
    identity = stable_hash({"manifest": manifest, "construction": backend.provenance})
    log_path = Path(f"{prefix}.jsonl")
    records = read_records(log_path)
    if any(r["identity"] != identity for r in records):
        raise ValueError("Generation log belongs to a different experiment")
    prompts = manifest["prompts"]
    baseline_path = Path(f"{prefix}_baselines.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        if baseline["identity"] != identity or baseline["prompts"] != prompts:
            raise ValueError("Incompatible baseline identity or prompts")
    else:
        if records:
            raise ValueError(
                "Cannot resume generations without their original baselines"
            )
        texts = backend.baseline(prompts)
        if len(texts) != len(prompts):
            raise ValueError("Baseline output count differs from prompt count")
        baseline = {"identity": identity, "prompts": prompts, "continuations": texts}
        write_json(baseline_path, baseline)
    if len(baseline["continuations"]) != len(prompts):
        raise ValueError("Baseline output count differs from prompt count")
    baseline_hash = stable_hash(baseline)
    if any(r["baseline_hash"] != baseline_hash for r in records):
        raise ValueError("Baseline content changed after generation")
    done = {r["run_id"] for r in records}
    expected = {
        stable_hash([identity, source, method, config["id"], lam, prompt["id"]])
        for source in manifest["sources"]
        for method in manifest["methods"]
        for config in manifest["configs"]
        for lam in manifest["settings"]["lambdas"]
        for prompt in prompts
    }
    if not done <= expected:
        raise ValueError("Generation log contains runs outside the frozen matrix")
    for source in manifest["sources"]:
        for method in manifest["methods"]:
            for config in manifest["configs"]:
                for lam in manifest["settings"]["lambdas"]:
                    keys = [
                        stable_hash(
                            [identity, source, method, config["id"], lam, p["id"]]
                        )
                        for p in prompts
                    ]
                    if all(key in done for key in keys):
                        continue
                    texts = backend.generate(config, source, method, lam, prompts)
                    if len(texts) != len(prompts):
                        raise ValueError(
                            "Steered output count differs from prompt count"
                        )
                    for i, (prompt, text, key) in enumerate(zip(prompts, texts, keys)):
                        if key in done:
                            continue
                        record = {
                            "run_id": key,
                            "identity": identity,
                            "baseline_hash": baseline_hash,
                            "source": source,
                            "method": method,
                            "lambda": lam,
                            "config": config,
                            "prompt": prompt,
                            "continuation": text,
                            "metrics": backend.measure(
                                text, baseline["continuations"][i], config
                            ),
                        }
                        with log_path.open("a") as stream:
                            stream.write(
                                json.dumps(record, ensure_ascii=False, allow_nan=False)
                                + "\n"
                            )
                        records.append(record)
                        done.add(key)
                    print(
                        f"{source} {method} {config['id']} λ={lam}: {len(done)}/{matrix_size(manifest)}",
                        flush=True,
                    )
    return records


def custom_token_rows(tokenizer, words, max_tokens):
    """Use NB65's four surface forms, preserving their token-frequency weighting."""
    rows, dropped = [], []
    for word in words:
        accepted = 0
        for variant in (word, f" {word}", f"{word} ", f" {word} "):
            ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if ids and ids[0] == tokenizer.bos_token_id:
                ids = ids[1:]
            if 0 < len(ids) <= max_tokens:
                rows.append(ids + [tokenizer.pad_token_id] * (max_tokens - len(ids)))
                accepted += 1
        if not accepted:
            dropped.append(word)
    if not rows:
        raise ValueError("No custom words fit the token limit")
    return rows, dropped


def clean_continuation(text, prompt, special_tokens=()):
    """Remove the assistant prefix as well as prompt markup from decoded output."""
    marker = "assistant<|end_header_id|>"
    if marker not in text:
        raise ValueError("Decoded generation is missing its assistant boundary")
    continuation = text.split(marker, 1)[1]
    if not continuation.startswith(prompt["prefix"]):
        raise ValueError("Decoded generation does not preserve the assistant prefix")
    continuation = continuation[len(prompt["prefix"]) :]
    for token in special_tokens:
        continuation = continuation.replace(token, "")
    return continuation.strip()


def frame_guide(config, concepts):
    """A differential frame is one equal-weight unit in the outer blend."""
    from frames.representations.concept import Concept

    units = [concepts[name] for name in config["positive"]]
    if config["negative"]:
        units[0] = units[0] - concepts[config["negative"][0]]
    return units[0] if len(units) == 1 else Concept.average(units, method="extrinsic")


def resolve_synsets(concepts, wordnet):
    """Resolve aliases in the actual loader corpus; retain requested IDs in provenance."""
    return {
        entry["concept"]: wordnet.synset(entry["synset"]).name() for entry in concepts
    }


class ModelBackend:
    """One model and common token rows for both concept representations."""

    def __init__(self, manifest):
        import numpy as np
        import pandas as pd
        import torch
        import transformers

        if not torch.cuda.is_available():
            raise RuntimeError("NB68 requires CUDA; no usable CUDA device is available")
        from frames.evaluation.harness import EvaluationHarness
        from frames.representations import FrameUnembeddingRepresentation
        from subspace import SubspaceUnembeddingRepresentation

        self.settings = s = manifest["settings"]
        torch.manual_seed(s["seed"])
        self.fur = FrameUnembeddingRepresentation.from_model_id(
            s["model_id"],
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.fur.model.model.eval()
        self.sur = SubspaceUnembeddingRepresentation(
            model=self.fur.model,
            unembedding=self.fur,
            data=self.fur.data,
            rank=s["subspace_rank"],
            method="pooled",
        )
        self.harness = EvaluationHarness(
            self.fur,
            Path("resources/68_unused.jsonl"),
            s["min_lemmas"],
            s["max_word_tokens"],
        )
        self.frames, self.spaces, self.references = {}, {}, {}
        self.provenance = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model_revision": getattr(
                self.fur.model.model.config, "_commit_hash", None
            ),
            "languages": self.fur.data.language_codes.name,
            "construction": [],
        }
        if self.provenance["languages"] != s["wordnet_languages"]:
            raise ValueError("WordNet language set differs from the manifest")
        canonical = resolve_synsets(manifest["concepts"], self.fur.data.wordnet)
        self.provenance["wordnet_version"] = self.fur.data.wordnet.get_version()
        all_synsets = self.fur.data.get_all_synsets(
            self.fur.model.tokenizer, s["min_lemmas"], s["max_word_tokens"]
        )
        available = set(all_synsets["synset"])
        missing = [
            e["synset"]
            for e in manifest["concepts"]
            if canonical[e["concept"]] not in available
        ]
        if missing:
            raise ValueError(f"Required WordNet synsets are unavailable: {missing}")
        for source in SOURCES:
            self.frames[source], self.spaces[source] = {}, {}
            for entry in manifest["concepts"]:
                name = entry["concept"]
                dropped = []
                if source == "custom":
                    rows, dropped = custom_token_rows(
                        self.fur.model.tokenizer, entry["words"], s["max_word_tokens"]
                    )
                    df = pd.DataFrame(
                        {
                            "synset": [entry["synset"]],
                            "tokens": [np.array(rows, dtype=np.int64)],
                        }
                    )
                else:
                    df = all_synsets[all_synsets["synset"] == canonical[name]]
                    rows = np.asarray(df.iloc[0]["tokens"]).tolist()
                frame = self.fur.compute_concept(df)
                space = self.sur.compute_concept(df)
                self.frames[source][name], self.spaces[source][name] = frame, space
                f_basis, s_basis = frame.tensor.squeeze(0), space.basis
                ranks = [
                    int(torch.linalg.matrix_rank(b.float()).item())
                    for b in (f_basis, s_basis)
                ]
                if not all(ranks):
                    raise ValueError(f"Empty concept representation: {source}/{name}")
                self.provenance["construction"].append(
                    {
                        "source": source,
                        "concept": name,
                        "synset": entry["synset"],
                        "canonical_synset": canonical[name],
                        "token_rows": rows,
                        "token_rows_hash": stable_hash(rows),
                        "excluded_words": dropped,
                        "frame_rank": ranks[0],
                        "subspace_rank": ranks[1],
                    }
                )
            self.references[f"{source}_frame"] = {
                n: c.tensor.squeeze(0) for n, c in self.frames[source].items()
            }
            self.references[f"{source}_subspace"] = {
                n: c.basis for n, c in self.spaces[source].items()
            }
        self._ppl_cache = {}
        self._hidden_cache = {}

    def _clean(self, texts, prompts):
        if len(texts) != len(prompts):
            raise ValueError("Model returned an unexpected batch length")
        return [
            clean_continuation(t, p, self.fur.model.tokenizer.all_special_tokens)
            for t, p in zip(texts, prompts)
        ]

    def baseline(self, prompts):
        texts = self.fur.generate(
            [p["text"] for p in prompts],
            max_new_tokens=self.settings["baseline_max_new_tokens"],
            do_sample=False,
        )
        return self._clean(texts, prompts)

    def generate(self, config, source, method, lam, prompts):
        from subspace import ConceptSubspace

        inputs = [p["text"] for p in prompts]
        kwargs = {
            "k": self.settings["k"],
            "steps": self.settings["steps"],
            "fluency_weight": lam,
        }
        if method == "frh_blend":
            guide = frame_guide(config, self.frames[source])
            # A single precomposed guide through the existing lambda-capable wrapper.
            texts, _ = self.fur.generate_with_topk_multi_guide(
                inputs, guides=[guide], weights=[1.0], normalize=None, **kwargs
            )
        else:
            guides = [self.spaces[source][n] for n in config["positive"]]
            guide = (
                guides[0]
                if len(guides) == 1
                else ConceptSubspace.blend(guides, rank=self.settings["subspace_rank"])
            )
            repel = (
                self.spaces[source][config["negative"][0]]
                if config["negative"]
                else None
            )
            texts, _ = self.sur.generate_with_topk_guide(
                inputs,
                guide=guide,
                repel_guide=repel,
                normalize_score=self.settings.get("subspace_normalize_score", False),
                **kwargs,
            )
        return self._clean(texts, prompts)

    def _ppl(self, text):
        if text not in self._ppl_cache:
            ids = self.fur.model.tokenizer(text, add_special_tokens=False)["input_ids"]
            value = (
                self.harness.perplexity(text)
                if text.strip() and len(ids) >= 2
                else None
            )
            self._ppl_cache[text] = (
                value if value is not None and math.isfinite(value) else None
            )
        return self._ppl_cache[text]

    def measure(self, text, baseline, config):
        from .comparison_metrics import evaluate_continuation, ppl_metrics

        cosine = evaluate_continuation(
            self.fur,
            text,
            baseline,
            self.references,
            config["positive"],
            config["negative"],
            hidden_cache=self._hidden_cache,
        )
        # Retain only the few shared baselines on the GPU, not thousands of outputs.
        if text != baseline:
            self._hidden_cache.pop(text, None)
        return {
            "cosine": cosine,
            "fluency": ppl_metrics(self._ppl(text), self._ppl(baseline)),
            "continuation_tokens": len(
                self.fur.model.tokenizer(text, add_special_tokens=False)["input_ids"]
            ),
            "baseline_tokens": len(
                self.fur.model.tokenizer(baseline, add_special_tokens=False)[
                    "input_ids"
                ]
            ),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts-dir", default="concepts")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the frozen word/configuration manifest without loading a model",
    )
    parser.add_argument("--output-prefix")
    args = parser.parse_args()
    manifest = make_experiment(args.concepts_dir, smoke=args.smoke)
    print(
        f"{len(manifest['concepts'])} concepts; {len(manifest['configs'])} configurations; {matrix_size(manifest)} steered continuations"
    )
    if args.dry_run:
        return
    prefix = args.output_prefix or (
        "resources/68_comparison_smoke" if args.smoke else "resources/68_comparison"
    )
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    freeze_json(f"{prefix}_manifest.json", manifest)
    if args.prepare_only:
        print(f"Prepared {prefix}_manifest.json")
        return
    backend = ModelBackend(manifest)
    run_battery(manifest, backend, prefix)
    from .comparison_report import write_report

    write_report(prefix)


if __name__ == "__main__":
    main()
