"""NB68 by concept source: expression, fluency and token health side by side.

Deliberately writes to .scratch/, not resources/: repo convention is that
resources/ artifact names embed the producing notebook's number, and this is a
scratch analysis with no notebook behind it.

Token health is plotted alongside the other two panels because the subspace
numbers cannot be read without it — those runs drop ~half their tokens.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "nb68_by_source.png"


def build():
    rec = []
    for line in open("resources/68_comparison.jsonl"):
        r = json.loads(line)
        fl = r["metrics"]["fluency"]
        rec.append(
            dict(
                source=r["source"],
                method=r["method"],
                lam=r["lambda"],
                nword=len(r["continuation"].split()),
                ratio=fl["ppl_ratio"] if fl["valid"] else np.nan,
            )
        )
    raw = pd.DataFrame(rec)
    raw["short"] = raw.nword <= 8
    tok = raw.groupby(["source", "method", "lam"]).agg(
        short_rate=("short", "mean"), med_ppl_ratio=("ratio", "median")
    )
    s = pd.read_csv("resources/68_comparison_summary.csv")
    exp = s.groupby(["source", "method", "lambda"])[["all_requested_improved"]].mean()
    exp.index.names = ["source", "method", "lam"]
    return exp.join(tok).reset_index()


def main():
    d = build()
    panels = [
        ("all_requested_improved", "concept expression\n(all requested improved)", None),
        ("med_ppl_ratio", "fluency cost\n(median perplexity ratio)", 1.0),
        ("short_rate", "token loss\n(runs $\\leq$ 8 words)", 0.0),
    ]
    keys = [
        ("wordnet", "frh_blend"),
        ("wordnet", "subspace"),
        ("custom", "frh_blend"),
        ("custom", "subspace"),
    ]
    labels = ["wordnet\nMFRH", "wordnet\nsubspace", "custom\nMFRH", "custom\nsubspace"]
    colors = ["#4C72B0", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(keys))
    width = 0.38

    for ax, (col, title, ref) in zip(axes, panels):
        for i, lam in enumerate([0.0, 0.4]):
            vals = [
                d[(d.source == s) & (d.method == m) & (d.lam == lam)][col].iloc[0]
                for s, m in keys
            ]
            bars = ax.bar(
                x + (i - 0.5) * width, vals, width,
                label=f"$\\lambda$={lam}", color=colors[i], edgecolor="black", linewidth=0.5,
            )
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
        if ref is not None:
            ax.axhline(ref, ls=":", c="grey", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].text(
        0.5, 0.93,
        "subspace bars are unreliable:\nthese runs lose ~half their tokens",
        transform=axes[2].transAxes, ha="center", fontsize=7.5, style="italic", color="#8B0000",
    )
    fig.suptitle(
        "NB68: MFRH vs subspace steering by concept source (Llama-3.1-8B-Instruct, bf16)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
