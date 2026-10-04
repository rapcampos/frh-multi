"""Probe: does the unnormalized subspace score ``‖Bᵀh‖`` prefer EOT candidates?

Hypothesis (NB68 follow-up): subspace steering truncates early (23-31% of runs
produce <=8 words, vs 0.4-2.8% for frame blending) because the score is an
unnormalized projection magnitude that rewards large-norm hidden states, and
EOT-adjacent hidden states have distinctive norms.

This instruments the real selection loop from
``subspace/representation.py::generate_with_topk_guide`` and records, for every
candidate at every step, the raw score, the hidden-state norm, and the
norm-normalized score. It then asks whether EOT candidates win under the raw
score and whether normalization removes that advantage.

Writes .scratch/subspace-normalization/probe_eot.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from frames.experiments.steering_comparison import ModelBackend, frame_guide  # noqa: F401
from subspace.generation import subspace_alignment

OUT = Path(__file__).parent / "probe_eot.csv"
MANIFEST = Path("resources/68_comparison_manifest.json")
# One 1-unit and one 4-unit config: truncation worsens with unit count.
CONFIG_IDS = ["n1-joy", "n1-man"]
N_UNIT4 = 2


@torch.inference_mode()
def probe(backend, config, prompts, source, k, steps):
    """Replay the subspace loop, recording per-candidate diagnostics."""
    from subspace import ConceptSubspace

    sur = backend.sur
    eos_id = sur.model.tokenizer.eos_token_id
    guides = [backend.spaces[source][n] for n in config["positive"]]
    guide = (
        guides[0]
        if len(guides) == 1
        else ConceptSubspace.blend(guides, rank=backend.settings["subspace_rank"])
    )
    basis = guide.basis.to(sur.model.device)

    inputs = sur.model.make_input([p["text"] for p in prompts], padding=True)
    first, _, _ = sur._candidates(inputs["input_ids"], k)
    tokens = first.flatten(0, 1)
    n = len(prompts)

    rows = []
    for step in range(steps):
        sequences, last_hs, _ = sur._candidates(tokens, k)
        # Score exactly as the live loop does (no repel guide here).
        scores = subspace_alignment(last_hs, basis)  # (n*k, T')
        raw_last = scores[:, -1]  # score at the newest position
        hnorm = last_hs[:, -1, :].float().norm(dim=-1)  # (n*k,)
        normed = raw_last / hnorm

        # The token that produced each row's newest position.
        newest = tokens[:, -1]
        for i in range(tokens.shape[0]):
            rows.append(
                dict(
                    step=step,
                    prompt=prompts[i // k]["id"],
                    config=config["id"],
                    source=source,
                    token_id=int(newest[i]),
                    is_eos=bool(int(newest[i]) == eos_id),
                    raw_score=float(raw_last[i]),
                    h_norm=float(hnorm[i]),
                    normed_score=float(normed[i]),
                )
            )

        # Advance the loop the same way the real implementation does.
        cum = scores.unsqueeze(-1).cumsum(-2).reshape(n, k, -1)
        col_idx = cum.max(-2).indices[..., -1]
        candidate_tokens = sequences.reshape(n, k, k, -1)
        tokens = candidate_tokens[np.arange(n), col_idx.cpu()].flatten(0, 1)
        if sur._is_generation_complete(tokens):
            break
    return rows


def main():
    manifest = json.loads(MANIFEST.read_text())
    s = manifest["settings"]
    by_id = {c["id"]: c for c in manifest["configs"]}
    unit4 = [c for c in manifest["configs"] if c["unit_count"] == 4][:N_UNIT4]
    configs = [by_id[c] for c in CONFIG_IDS if c in by_id] + unit4

    backend = ModelBackend(manifest)
    rows = []
    for source in ("wordnet", "custom"):
        for config in configs:
            rows += probe(
                backend, config, manifest["prompts"], source, s["k"], s["steps"]
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} candidate observations)")

    print("\n=== EOT vs non-EOT candidates ===")
    print(df.groupby("is_eos")[["raw_score", "h_norm", "normed_score"]].agg(
        ["mean", "median", "count"]
    ).round(3))


if __name__ == "__main__":
    main()
