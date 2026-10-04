"""NB69 rank variant: subspace steering with 16 or 32 directions.

Only the subspace *steering* rank changes: each concept subspace and every blend
use ``steering_subspace_rank`` directions. Measurement stays on NB69's fixed
rank-8 reference views, so ranks are compared on one yardstick. Only the subspace
arm is regenerated (three sources, both λ, raw score); FRH and rank-8 subspace
runs are reused from NB68/NB69 after a reproduction check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .comparison_manifest import stable_hash
from .topc_comparison import TopCBackend, reproduction_check
from .topc_comparison import make_experiment as make_nb69_experiment
from .topc_normalized import NB69_PREFIX, SOURCES

RANK_KEY = "steering_subspace_rank"


def make_experiment(rank, nb69_prefix=NB69_PREFIX, *, smoke=False):
    """NB69's frozen matrix, subspace only, all three sources, steering rank ``rank``."""
    nb69 = json.loads(Path(f"{nb69_prefix}_manifest.json").read_text())
    manifest = make_nb69_experiment(smoke=smoke)
    root = Path(__file__).resolve().parents[2]
    manifest.update(
        methods=["subspace"],
        sources=list(SOURCES),
        settings={
            **manifest["settings"],
            RANK_KEY: rank,
            "nb69_manifest_hash": stable_hash(nb69),
        },
        implementation_hashes={
            **manifest["implementation_hashes"],
            "frames/experiments/topc_rank.py": hashlib.sha256(
                (root / "frames/experiments/topc_rank.py").read_bytes()
            ).hexdigest(),
        },
    )
    return manifest


class RankBackend(TopCBackend):
    """NB69 backend whose subspace guides use ``steering_subspace_rank`` directions."""

    def __init__(self, manifest):
        import numpy as np
        import torch

        from subspace import SubspaceUnembeddingRepresentation

        super().__init__(manifest)
        rank = self.settings[RANK_KEY]
        steer = SubspaceUnembeddingRepresentation(
            model=self.fur.model,
            unembedding=self.fur,
            data=self.fur.data,
            rank=rank,
            method="pooled",
        )
        self.steer_spaces = {source: {} for source in SOURCES}
        self.provenance["steering"] = []
        for r in self.provenance["construction"]:
            # Same token pool as the rank-8 concept; pooled SVD drops pad ids.
            space = steer.build_concept_subspace(np.asarray(r["token_rows"]))
            self.steer_spaces[r["source"]][r["concept"]] = space
            self.provenance["steering"].append(
                {
                    "source": r["source"],
                    "concept": r["concept"],
                    "rank": rank,
                    "realized_rank": int(torch.linalg.matrix_rank(space.float())),
                }
            )

    def generate(self, config, source, method, lam, prompts):
        rank = self.settings.get(RANK_KEY)
        if method != "subspace" or rank is None:
            return super().generate(config, source, method, lam, prompts)
        from subspace import ConceptSubspace

        spaces = {
            n: ConceptSubspace(synset=n, basis=b)
            for n, b in self.steer_spaces[source].items()
        }
        guides = [spaces[n] for n in config["positive"]]
        guide = (
            guides[0] if len(guides) == 1 else ConceptSubspace.blend(guides, rank=rank)
        )
        repel = spaces[config["negative"][0]] if config["negative"] else None
        texts, _ = self.sur.generate_with_topk_guide(
            [p["text"] for p in prompts],
            guide=guide,
            repel_guide=repel,
            k=self.settings["k"],
            steps=self.settings["steps"],
            fluency_weight=lam,
        )
        return self._clean(texts, prompts)


def reference_reproduction_check(backend, manifest):
    """Regenerate NB68 runs at NB69's rank-8 settings before reusing its rows."""
    reference = json.loads(json.dumps(manifest))
    reference["methods"] = ["frh_blend", "subspace"]
    del reference["settings"][RANK_KEY]
    settings = backend.settings
    backend.settings = reference["settings"]
    try:
        return reproduction_check(backend, reference)
    finally:
        backend.settings = settings


def load_runs(prefixes, nb69_prefix=NB69_PREFIX):
    """NB68/NB69 runs plus each rank run, its method relabelled ``subspace_r{rank}``."""
    from .steering_comparison import read_records
    from .topc_comparison import EXCLUDED_PROMPTS
    from .topc_normalized import load_runs as load_reference

    runs = [
        r
        for r in load_reference(prefixes[0], nb69_prefix)
        if r["method"] != "subspace_norm"
    ]
    for prefix in prefixes:
        rank = json.loads(Path(f"{prefix}_manifest.json").read_text())["settings"][
            RANK_KEY
        ]
        for r in read_records(f"{prefix}.jsonl"):
            if r["prompt"]["id"] not in EXCLUDED_PROMPTS:
                runs.append({**r, "method": f"subspace_r{rank}"})
    return runs
