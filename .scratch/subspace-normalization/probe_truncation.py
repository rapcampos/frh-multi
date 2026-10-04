"""Probe v2: where does the text go in a short subspace run?

Probe v1 instrumented the selection loop and found no EOT candidate in 6,144
observations, with every run going the full 24 steps -- yet the same configs
truncate in the NB68 battery (n1-man/custom: 62.5% short at lambda=0). So v1 was
not faithful: it never decoded, and the loop is not what shortens the text.

This calls the real ``generate_with_topk_guide`` on a config/prompt known to
truncate, then dumps the raw token ids and the decode/clean stages so the token
loss can be located.
"""

import json
from pathlib import Path

import torch

from frames.experiments.steering_comparison import ModelBackend, clean_continuation

MANIFEST = Path("resources/68_comparison_manifest.json")
OUT = Path(__file__).parent / "probe_truncation.json"
# n1-man/custom truncates 62.5% of the time at lambda=0 in the battery.
CONFIG_ID = "n1-man"
SOURCE = "custom"


@torch.inference_mode()
def main():
    manifest = json.loads(MANIFEST.read_text())
    s = manifest["settings"]
    config = next(c for c in manifest["configs"] if c["id"] == CONFIG_ID)
    prompts = manifest["prompts"]

    backend = ModelBackend(manifest)
    sur = backend.sur
    tok = sur.model.tokenizer
    guide = backend.spaces[SOURCE][config["positive"][0]]

    # The real call, exactly as the battery makes it.
    texts, _ = sur.generate_with_topk_guide(
        [p["text"] for p in prompts], guide=guide, k=s["k"], steps=s["steps"]
    )

    report = []
    for p, raw in zip(prompts, texts):
        cleaned = clean_continuation(raw, p, tok.all_special_tokens)
        ids = tok(raw, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(p["text"], add_special_tokens=False)["input_ids"]
        report.append(
            dict(
                prompt=p["id"],
                n_words_clean=len(cleaned.split()),
                n_ids_full=len(ids),
                n_ids_prompt=len(prompt_ids),
                n_ids_generated=len(ids) - len(prompt_ids),
                has_eot=("<|eot_id|>" in raw),
                eot_count=raw.count("<|eot_id|>"),
                tail_after_prefix=raw.split("assistant<|end_header_id|>", 1)[1][:300],
                cleaned=cleaned,
            )
        )

    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}\n")
    for r in report:
        flag = "SHORT" if r["n_words_clean"] <= 8 else "     "
        print(
            f"{flag} {r['prompt']} words={r['n_words_clean']:2d} "
            f"gen_ids={r['n_ids_generated']:3d} eot={r['eot_count']} "
            f"| {r['cleaned'][:70]!r}"
        )
    print("\n=== raw tail of first SHORT prompt ===")
    short = [r for r in report if r["n_words_clean"] <= 8]
    if short:
        print(repr(short[0]["tail_after_prefix"]))
    else:
        print("no short runs reproduced")


if __name__ == "__main__":
    main()
