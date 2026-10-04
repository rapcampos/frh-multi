"""NB69: NB68's battery plus a lemma-count-matched custom concept source.

``custom_top_c`` keeps, for each concept, the first ``c`` consensus-ranked custom
words, where ``c`` is the number of WordNet member lemmas (all eight languages,
after the loader's token filter) behind that concept's NB68 WordNet frame.

Only the new source is generated. NB68's WordNet and full-custom runs are reused
unchanged, and their continuations are rescored on the two new reference views,
so every run is measured on the same six views. Generation keeps NB68's exact
eight-prompt batches (bf16 greedy decoding depends on batch composition); prompt
p07 is excluded only at analysis, since its unsteered baseline is empty.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .comparison_manifest import stable_hash
from .steering_comparison import (
    ModelBackend,
    custom_token_rows,
    make_experiment as make_nb68_experiment,
    read_records,
    resolve_synsets,
)

NB68_PREFIX = "resources/68_comparison"
NEW_SOURCE = "custom_top_c"
NEW_VIEWS = (f"{NEW_SOURCE}_frame", f"{NEW_SOURCE}_subspace")
EXCLUDED_PROMPTS = ("p07",)


def make_experiment(nb68_prefix=NB68_PREFIX, *, smoke=False):
    """NB68's frozen concepts/configs/prompts, restricted to the new source."""
    nb68 = json.loads(Path(f"{nb68_prefix}_manifest.json").read_text())
    manifest = json.loads(json.dumps(nb68))
    configs = nb68["configs"]
    if smoke:
        keep = {c["id"] for c in make_nb68_experiment(smoke=True)["configs"]}
        configs = [c for c in configs if c["id"] in keep]
    root = Path(__file__).resolve().parents[2]
    manifest.update(
        configs=configs,
        sources=[NEW_SOURCE],
        settings={
            **nb68["settings"],
            "smoke": smoke,
            "top_c_rule": "first c ranked custom words that fit the token limit; "
            "c = WordNet member lemmas of the synset (eight languages, lowercased)",
            "excluded_prompts": list(EXCLUDED_PROMPTS),
            "nb68_manifest_hash": stable_hash(nb68),
        },
        implementation_hashes={
            **nb68["implementation_hashes"],
            "frames/experiments/topc_comparison.py": hashlib.sha256(
                (root / "frames/experiments/topc_comparison.py").read_bytes()
            ).hexdigest(),
        },
    )
    return manifest


def top_c_token_rows(tokenizer, words, c, max_tokens):
    """Rows for the first ``c`` ranked words that survive the token limit."""
    rows, kept, skipped = [], [], []
    for word in words:
        if len(kept) == c:
            break
        try:
            word_rows, _ = custom_token_rows(tokenizer, [word], max_tokens)
        except ValueError:  # no surface form fits the token limit
            skipped.append(word)
            continue
        rows.extend(word_rows)
        kept.append(word)
    if len(kept) < c:
        raise ValueError(f"Only {len(kept)} custom words fit; {c} requested")
    return rows, kept, skipped


class TopCBackend(ModelBackend):
    """NB68's backend, plus the ``custom_top_c`` frames, subspaces and views."""

    def __init__(self, manifest, nb68_prefix=NB68_PREFIX):
        import numpy as np
        import pandas as pd
        import torch

        super().__init__(manifest)
        s = self.settings
        canonical = resolve_synsets(manifest["concepts"], self.fur.data.wordnet)
        nb68_rows = {
            (r["source"], r["concept"]): r["token_rows_hash"]
            for r in json.loads(Path(f"{nb68_prefix}_construction.json").read_text())[
                "construction"
            ]
        }
        drift = [
            (r["source"], r["concept"])
            for r in self.provenance["construction"]
            if nb68_rows[(r["source"], r["concept"])] != r["token_rows_hash"]
        ]
        if drift:
            raise ValueError(f"NB68 token rows no longer reproduce: {drift}")
        self.frames[NEW_SOURCE], self.spaces[NEW_SOURCE] = {}, {}
        for entry in manifest["concepts"]:
            name = entry["concept"]
            # Also loads the per-lemma parquet before generation (cuDF OOM).
            lemmas = sorted(self.harness.member_lemmas(canonical[name]))
            rows, kept, skipped = top_c_token_rows(
                self.fur.model.tokenizer,
                entry["words"],
                len(lemmas),
                s["max_word_tokens"],
            )
            df = pd.DataFrame(
                {
                    "synset": [entry["synset"]],
                    "tokens": [np.array(rows, dtype=np.int64)],
                }
            )
            frame = self.fur.compute_concept(df)
            space = self.sur.compute_concept(df)
            self.frames[NEW_SOURCE][name], self.spaces[NEW_SOURCE][name] = frame, space
            ranks = [
                int(torch.linalg.matrix_rank(b.float()).item())
                for b in (frame.tensor.squeeze(0), space.basis)
            ]
            if not all(ranks):
                raise ValueError(f"Empty concept representation: {NEW_SOURCE}/{name}")
            self.provenance["construction"].append(
                {
                    "source": NEW_SOURCE,
                    "concept": name,
                    "synset": entry["synset"],
                    "canonical_synset": canonical[name],
                    "wordnet_lemmas": lemmas,
                    "c": len(lemmas),
                    "words": kept,
                    "skipped_words": skipped,
                    "token_rows": rows,
                    "token_rows_hash": stable_hash(rows),
                    "frame_rank": ranks[0],
                    "subspace_rank": ranks[1],
                }
            )
        self.references[NEW_VIEWS[0]] = {
            n: c.tensor.squeeze(0) for n, c in self.frames[NEW_SOURCE].items()
        }
        self.references[NEW_VIEWS[1]] = {
            n: c.basis for n, c in self.spaces[NEW_SOURCE].items()
        }
        baseline = json.loads(Path(f"{nb68_prefix}_baselines.json").read_text())
        self._nb68_baselines = dict(
            zip((p["id"] for p in baseline["prompts"]), baseline["continuations"])
        )

    def baseline(self, prompts):
        """NB68's unsteered continuations, so all deltas share one baseline."""
        return [self._nb68_baselines[p["id"]] for p in prompts]


def rescore_nb68(backend, manifest, output_path, nb68_prefix=NB68_PREFIX):
    """Score NB68's continuations on the new views; resumable, keyed by run ID."""
    from .comparison_metrics import evaluate_continuation

    output_path = Path(output_path)
    done = {r["run_id"] for r in read_records(output_path)}
    configs = {c["id"] for c in manifest["configs"]}
    prompts = {p["id"] for p in manifest["prompts"]} - set(EXCLUDED_PROMPTS)
    references = {v: backend.references[v] for v in NEW_VIEWS}
    views_hash = stable_hash(
        [
            r["token_rows_hash"]
            for r in backend.provenance["construction"]
            if r["source"] == NEW_SOURCE
        ]
    )
    count = 0
    with output_path.open("a") as stream:
        for r in read_records(f"{nb68_prefix}.jsonl"):
            if r["config"]["id"] not in configs or r["prompt"]["id"] not in prompts:
                continue
            count += 1
            if r["run_id"] in done:
                continue
            baseline = backend._nb68_baselines[r["prompt"]["id"]]
            cosine = evaluate_continuation(
                backend.fur,
                r["continuation"],
                baseline,
                references,
                r["config"]["positive"],
                r["config"]["negative"],
                hidden_cache=backend._hidden_cache,
            )
            if r["continuation"] != baseline:
                backend._hidden_cache.pop(r["continuation"], None)
            record = {"run_id": r["run_id"], "views_hash": views_hash, "cosine": cosine}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return count


def reproduction_check(backend, manifest, nb68_prefix=NB68_PREFIX, n_configs=2):
    """Regenerate a few NB68 runs with today's code; they must match exactly.

    Reusing NB68's WordNet/custom rows is only valid if the current code still
    generates the same text and cosine values for them.
    """
    from .comparison_metrics import evaluate_continuation

    records = read_records(f"{nb68_prefix}.jsonl")
    index = {
        (r["source"], r["method"], r["lambda"], r["config"]["id"], r["prompt"]["id"]): r
        for r in records
    }
    prompts = manifest["prompts"]  # NB68's exact batch
    old_views = {
        v: refs for v, refs in backend.references.items() if v not in NEW_VIEWS
    }
    rows = []
    for config in manifest["configs"][:n_configs]:
        for source in ("wordnet", "custom"):
            for method in manifest["methods"]:
                for lam in manifest["settings"]["lambdas"]:
                    texts = backend.generate(config, source, method, lam, prompts)
                    for prompt, text in zip(prompts, texts):
                        old = index[(source, method, lam, config["id"], prompt["id"])]
                        cosine = evaluate_continuation(
                            backend.fur,
                            old["continuation"],
                            backend._nb68_baselines[prompt["id"]],
                            old_views,
                            config["positive"],
                            config["negative"],
                        )
                        gaps = [
                            abs(
                                cosine[v]["summary"]["signed_mean_delta"]
                                - old["metrics"]["cosine"][v]["summary"][
                                    "signed_mean_delta"
                                ]
                            )
                            for v in old_views
                            if cosine[v]["valid"]
                        ]
                        rows.append(
                            {
                                "source": source,
                                "method": method,
                                "lambda": lam,
                                "config": config["id"],
                                "prompt": prompt["id"],
                                "same_text": text == old["continuation"],
                                "shared_prefix_chars": len(
                                    os.path.commonprefix([text, old["continuation"]])
                                ),
                                "chars": len(old["continuation"]),
                                "max_cosine_gap": max(gaps, default=0.0),
                            }
                        )
    return rows
