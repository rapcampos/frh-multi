"""Smoke experiment: current subspace loop vs Approach A (token-space + real terminator).

Same concepts, prompts, k, steps and scoring as NB68 — only the generation loop
differs, so any change is attributable to the loop.

Reports the truncation symptom (generated token count, short-run rate), how often
a run legitimately terminates, and how far the output sits from the unsteered
baseline, which is the check that a "fix" has not simply switched steering off
(the trap normalization fell into: 50% of its outputs were byte-identical to the
greedy baseline).
"""

import json
from pathlib import Path

import pandas as pd
import torch

from frames.experiments.steering_comparison import ModelBackend, clean_continuation
from subspace.representation import SubspaceUnembeddingRepresentation

import sys

sys.path.insert(0, str(Path(__file__).parent))
from approach_a import TokenSpaceMixin  # noqa: E402

MANIFEST = Path("resources/68_comparison_manifest.json")
OUT = Path(__file__).parent / "smoke_approach_a.csv"
SOURCE = "wordnet"
LAMBDAS = [0.0, 0.4]
N_CONFIGS = 6


class PatchedSubspace(TokenSpaceMixin, SubspaceUnembeddingRepresentation):
    """The real class with only the generation loop replaced."""


def prefix_overlap(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


@torch.inference_mode()
def main():
    manifest = json.loads(MANIFEST.read_text())
    s = manifest["settings"]
    prompts = manifest["prompts"]
    # A spread of unit counts; truncation worsened with unit count in NB68.
    configs = [
        c
        for u in (1, 2, 3, 4)
        for c in [next(x for x in manifest["configs"] if x["unit_count"] == u)]
    ][:N_CONFIGS]

    backend = ModelBackend(manifest)
    tok = backend.fur.model.tokenizer
    baselines = dict(zip([p["id"] for p in prompts], backend.baseline(prompts)))

    patched = PatchedSubspace(
        model=backend.fur.model,
        unembedding=backend.fur,
        data=backend.fur.data,
        rank=s["subspace_rank"],
        method="pooled",
    )
    patched.__dict__.update(
        {k: v for k, v in backend.sur.__dict__.items() if k not in patched.__dict__}
    )

    from subspace import ConceptSubspace

    rows = []
    for config in configs:
        guides = [backend.spaces[SOURCE][n] for n in config["positive"]]
        guide = (
            guides[0]
            if len(guides) == 1
            else ConceptSubspace.blend(guides, rank=s["subspace_rank"])
        )
        repel = (
            backend.spaces[SOURCE][config["negative"][0]]
            if config["negative"]
            else None
        )
        for lam in LAMBDAS:
            current, _ = backend.sur.generate_with_topk_guide(
                [p["text"] for p in prompts],
                guide=guide,
                repel_guide=repel,
                k=s["k"],
                steps=s["steps"],
                fluency_weight=lam,
            )
            fixed, _, stats = patched.generate_token_space(
                [p["text"] for p in prompts],
                guide=guide,
                repel_guide=repel,
                k=s["k"],
                steps=s["steps"],
                fluency_weight=lam,
            )
            for i, p in enumerate(prompts):
                for loop, raw, st in (
                    ("current", current[i], None),
                    ("approach_a", fixed[i], stats[i]),
                ):
                    cleaned = clean_continuation(raw, p, tok.all_special_tokens)
                    rows.append(
                        dict(
                            loop=loop,
                            config=config["id"],
                            unit_count=config["unit_count"],
                            lam=lam,
                            prompt=p["id"],
                            n_words=len(cleaned.split()),
                            gen_tokens=st["generated_tokens"] if st else None,
                            terminated=st["terminated"] if st else None,
                            baseline_prefix=prefix_overlap(cleaned, baselines[p["id"]]),
                            identical_to_baseline=cleaned.strip()
                            == baselines[p["id"]].strip(),
                            text=cleaned,
                        )
                    )

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} runs)\n")

    df["short"] = df.n_words <= 8
    print("=== truncation and baseline divergence ===")
    summary = df.groupby(["loop", "lam"]).agg(
        mean_words=("n_words", "mean"),
        short_rate=("short", "mean"),
        baseline_prefix_words=("baseline_prefix", "mean"),
        identical_to_baseline=("identical_to_baseline", "mean"),
    )
    print(summary.round(2).to_string())
    print()
    print("=== approach_a: generated tokens (24 steps => 25 expected) ===")
    a = df[df.loop == "approach_a"]
    print(a.groupby("lam")["gen_tokens"].describe(percentiles=[0.5]).round(1).to_string())
    print(f"\nterminated early (emitted <|eot_id|>): {a.terminated.mean():.1%}")
    print()
    print("=== short-run rate by unit count ===")
    print(df.pivot_table(index="unit_count", columns="loop", values="short").round(3).to_string())
    print()
    for loop in ("current", "approach_a"):
        print(f"--- {loop} samples (lambda=0) ---")
        g = df[(df.loop == loop) & (df.lam == 0.0)]
        for _, r in g.head(4).iterrows():
            print(f"  {r['prompt']} ({r['n_words']:2d}w) {r['text'][:74]!r}")


if __name__ == "__main__":
    main()
