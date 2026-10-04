"""NB70: NB69's three-source battery with norm-normalized subspace scoring.

Normalization (``‖Bᵀh‖ / ‖h‖``) only changes subspace steering, so only the
subspace arm is regenerated, for all three sources and both λ. FRH runs are
reused from NB68/NB69, and the unnormalized subspace runs stay available as the
reference arm. Generation keeps NB68's eight-prompt batches and baselines.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .comparison_manifest import stable_hash
from .topc_comparison import NB68_PREFIX, NEW_SOURCE
from .topc_comparison import make_experiment as make_nb69_experiment

NB69_PREFIX = "resources/69_topc"
SOURCES = ("wordnet", NEW_SOURCE, "custom")


def make_experiment(nb69_prefix=NB69_PREFIX, *, smoke=False):
    """NB69's frozen matrix, subspace only, normalized score, all three sources."""
    nb69 = json.loads(Path(f"{nb69_prefix}_manifest.json").read_text())
    manifest = make_nb69_experiment(smoke=smoke)
    root = Path(__file__).resolve().parents[2]
    manifest.update(
        methods=["subspace"],
        sources=list(SOURCES),
        settings={
            **manifest["settings"],
            "subspace_normalize_score": True,
            "nb69_manifest_hash": stable_hash(nb69),
        },
        implementation_hashes={
            **manifest["implementation_hashes"],
            "frames/experiments/topc_normalized.py": hashlib.sha256(
                (root / "frames/experiments/topc_normalized.py").read_bytes()
            ).hexdigest(),
            "subspace/generation.py": hashlib.sha256(
                (root / "subspace/generation.py").read_bytes()
            ).hexdigest(),
        },
    )
    return manifest


def unnormalized(manifest):
    """The same matrix as it was generated for NB68/NB69 (both methods, raw score)."""
    reference = json.loads(json.dumps(manifest))
    reference["methods"] = ["frh_blend", "subspace"]
    del reference["settings"]["subspace_normalize_score"]
    return reference


def load_runs(prefix, nb69_prefix=NB69_PREFIX, nb68_prefix=NB68_PREFIX):
    """All runs on six views, labelled ``frh_blend``/``subspace``/``subspace_norm``."""
    from .steering_comparison import read_records
    from .topc_comparison import EXCLUDED_PROMPTS

    rescore = {
        r["run_id"]: r["cosine"]
        for r in read_records(f"{nb69_prefix}_nb68_rescore.jsonl")
    }
    old = read_records(f"{nb68_prefix}.jsonl") + read_records(f"{nb69_prefix}.jsonl")
    new = read_records(f"{prefix}.jsonl")
    runs = []
    for arm, records in (("old", old), ("new", new)):
        for r in records:
            if r["prompt"]["id"] in EXCLUDED_PROMPTS:
                continue
            r = {**r, "metrics": {**r["metrics"]}}
            r["metrics"]["cosine"] = {
                **r["metrics"]["cosine"],
                **rescore.get(r["run_id"], {}),
            }
            if arm == "new":
                r["method"] = "subspace_norm"
            runs.append(r)
    return runs


def raw_reproduction_check(backend, manifest):
    """Regenerate NB68 runs with the raw score, proving the environment reproduces.

    Only then is it valid to set the reused FRH/raw-subspace rows beside new ones.
    """
    from .topc_comparison import reproduction_check

    reference = unnormalized(manifest)
    settings = backend.settings
    backend.settings = reference["settings"]
    try:
        return reproduction_check(backend, reference)
    finally:
        backend.settings = settings
