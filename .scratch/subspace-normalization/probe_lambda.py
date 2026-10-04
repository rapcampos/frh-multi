"""Probe v4: normalization x fluency weight.

At lambda=0 normalization is a near no-op for selection: within a candidate set
``‖h‖`` varies by ~1.5% (CV) while the raw score varies by ~28.5%, so dividing by
a near-constant preserves the argmax 97.5% of the time, and the shared prefix
makes selection depend only on the last position anyway. Probe v3 confirmed it:
raw and normalized produced byte-identical output.

Where normalization must matter is lambda > 0. Selection maximizes
``cumulative_alignment + lambda * cumulative_logprob``. Raw alignment runs ~7.9
per position; normalized runs ~0.055 — a ~144x rescale of the steering term
against an unchanged fluency term. The same lambda therefore buys vastly more
fluency pressure under normalization.

This runs the real generation call across both scorings and several lambdas,
reporting output length (the truncation symptom) and the text itself.
"""

import json
from pathlib import Path

import pandas as pd
import torch

from frames.experiments.steering_comparison import ModelBackend, clean_continuation

MANIFEST = Path("resources/68_comparison_manifest.json")
OUT = Path(__file__).parent / "probe_lambda.csv"
CONFIG_ID = "n1-man"
SOURCE = "custom"
LAMBDAS = [0.0, 0.4]


@torch.inference_mode()
def main():
    manifest = json.loads(MANIFEST.read_text())
    s = manifest["settings"]
    config = next(c for c in manifest["configs"] if c["id"] == CONFIG_ID)
    prompts = manifest["prompts"]

    backend = ModelBackend(manifest)
    sur, tok = backend.sur, backend.fur.model.tokenizer
    guide = backend.spaces[SOURCE][config["positive"][0]]

    rows = []
    for normalize in (False, True):
        for lam in LAMBDAS:
            texts, _ = sur.generate_with_topk_guide(
                [p["text"] for p in prompts],
                guide=guide,
                k=s["k"],
                steps=s["steps"],
                fluency_weight=lam,
                normalize_score=normalize,
            )
            for p, raw in zip(prompts, texts):
                cleaned = clean_continuation(raw, p, tok.all_special_tokens)
                rows.append(
                    dict(
                        scoring="normalized" if normalize else "raw",
                        lam=lam,
                        prompt=p["id"],
                        n_words=len(cleaned.split()),
                        text=cleaned,
                    )
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}\n")
    print("=== mean words (24 steps should give ~20; <=8 is the truncation symptom) ===")
    print(df.pivot_table(index="lam", columns="scoring", values="n_words",
                         aggfunc=["mean", lambda x: (x <= 8).mean()]).round(2))
    print()
    for (sc, lam), g in df.groupby(["scoring", "lam"]):
        print(f"--- {sc} lambda={lam} ---")
        for _, r in g.head(3).iterrows():
            print(f"  {r['prompt']} ({r['n_words']:2d}w) {r['text'][:80]!r}")


if __name__ == "__main__":
    main()
