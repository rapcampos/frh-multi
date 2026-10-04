"""Probe v3: does the subspace path lose tokens in the decode/re-encode round trip?

Established so far:
  v1 -- no EOT candidate in 6,144 observations; the loop never breaks early.
  v2 -- short runs reproduce with zero EOT, and come in two flavours: real token
        loss (p04: 3 generated ids after 24 steps) and subword garbling
        (p06: 'kaleidesopecopeptiomeof', '24 x   a day') at full length.

Both ``subspace/representation.py::_candidates`` and the frame pipeline's
``frames/representations/frame.py::_generate_candidates`` rebuild their inputs
with ``make_input(decode(tokens))`` every step, so the lossy round trip is shared.
The difference must therefore be in *which* tokens each score selects: tokens
that do not survive re-tokenization cost length, and the loss compounds over 24
steps.

This measures sequence length after each step for both methods on the same
prompts. A step that adds one token should grow the sequence by exactly 1;
anything less is round-trip loss.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from frames.experiments.steering_comparison import ModelBackend, frame_guide
from subspace.generation import subspace_alignment

MANIFEST = Path("resources/68_comparison_manifest.json")
OUT = Path(__file__).parent / "probe_roundtrip.csv"
CONFIG_ID = "n1-man"
SOURCE = "custom"


def real_lengths(model, tokens, k):
    """Non-pad token count of each surviving beam (one row per prompt).

    ``tokens`` is padded to a common batch width every step, so ``shape[-1]``
    measures padding, not content. Decode each kept row and re-tokenize without
    special tokens to get the length that actually survives the round trip.
    """
    kept = tokens[::k]
    texts = model.decode(kept)
    return [
        len(model.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts
    ]


@torch.inference_mode()
def trace_subspace(backend, guide, prompts, k, steps, normalize=False, label="subspace"):
    sur = backend.sur
    basis = guide.basis.to(sur.model.device)
    inputs = sur.model.make_input([p["text"] for p in prompts], padding=True)
    first, _, _ = sur._candidates(inputs["input_ids"], k)
    tokens = first.flatten(0, 1)
    n = len(prompts)
    out = [(label, 0, pid, L) for pid, L in zip(
        [p["id"] for p in prompts], real_lengths(sur.model, tokens, k))]
    for step in range(steps):
        sequences, last_hs, _ = sur._candidates(tokens, k)
        scores = subspace_alignment(last_hs, basis, normalize=normalize)
        cum = scores.unsqueeze(-1).cumsum(-2).reshape(n, k, -1)
        col_idx = cum.max(-2).indices[..., -1]
        tokens = sequences.reshape(n, k, k, -1)[np.arange(n), col_idx.cpu()].flatten(
            0, 1
        )
        out += [(label, step + 1, pid, L) for pid, L in zip(
            [p["id"] for p in prompts], real_lengths(sur.model, tokens, k))]
        if sur._is_generation_complete(tokens):
            break
    return out


@torch.inference_mode()
def trace_frame(backend, guide, prompts, k, steps):
    """Mirror of the frame loop, tracking only sequence length."""
    fur = backend.fur
    inputs = fur.model.make_input([p["text"] for p in prompts], padding=True)
    first, _, _ = fur._generate_candidates(inputs["input_ids"], k)
    tokens = first.flatten(0, 1)
    n = len(prompts)
    out = [("frh_blend", 0, pid, L) for pid, L in zip(
        [p["id"] for p in prompts], real_lengths(fur.model, tokens, k))]
    for step in range(steps):
        sequences, hs, _ = fur._generate_candidates(tokens, k)
        # The real frame scoring, same call the guided loop makes.
        scores = fur._score_hidden_states(hs[-1], [guide], [1.0], fur._weighted_sum_score, n, None)
        cum = scores.cumsum(-2).reshape(n, k, -1)
        col_idx = cum.max(-2).indices[..., -1].cpu()
        tokens = sequences.reshape(n, k, k, -1)[np.arange(n), col_idx.cpu()].flatten(
            0, 1
        )
        out += [("frh_blend", step + 1, pid, L) for pid, L in zip(
            [p["id"] for p in prompts], real_lengths(fur.model, tokens, k))]
    return out


def main():
    manifest = json.loads(MANIFEST.read_text())
    s = manifest["settings"]
    config = next(c for c in manifest["configs"] if c["id"] == CONFIG_ID)
    prompts = manifest["prompts"]

    backend = ModelBackend(manifest)
    rows = []
    guide = backend.spaces[SOURCE][config["positive"][0]]
    rows += trace_subspace(
        backend, guide, prompts, s["k"], s["steps"], False, "subspace_raw"
    )
    rows += trace_subspace(
        backend, guide, prompts, s["k"], s["steps"], True, "subspace_normalized"
    )
    rows += trace_frame(
        backend, frame_guide(config, backend.frames[SOURCE]), prompts, s["k"], s["steps"]
    )
    df = pd.DataFrame(rows, columns=["method", "step", "prompt", "seq_len"])
    df["growth"] = df.groupby(["method", "prompt"])["seq_len"].diff()
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}\n")
    print("=== mean growth per step (1.0 = no round-trip loss) ===")
    print(df.groupby("method")["growth"].agg(["mean", "min", "count"]).round(3))
    print()
    print("=== per prompt: final real length and lossy-step count ===")
    piv = df.groupby(["method", "prompt"]).agg(
        final_len=("seq_len", "last"),
        lossy_steps=("growth", lambda g: int((g.dropna() < 1).sum())),
    )
    print(piv.unstack(0).to_string())


if __name__ == "__main__":
    main()
