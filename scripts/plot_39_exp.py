"""Plot experiment 39's expression-based scaling curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "resources" / "39_scaling4_summary.csv"
OUTPUT_PATH = ROOT / "resources" / "39_exp.png"

METHODS = ("blend", "common-blend", "vote", "schedule")
COLORS = {
    "blend": "tab:blue",
    "common-blend": "tab:red",
    "vote": "tab:orange",
    "schedule": "tab:green",
}


def main() -> None:
    summary = pd.read_csv(SUMMARY_PATH)
    required = {"method", "n", "joint", "all_positive_cos"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    panels = (
        (axes[0], "joint", "Mean joint concept expression gain"),
        (
            axes[1],
            "all_positive_cos",
            "Share of runs with all concepts improved (cosine)",
        ),
    )

    for ax, metric, ylabel in panels:
        for method in METHODS:
            subset = summary[summary["method"] == method].sort_values("n")
            ax.plot(
                subset["n"],
                subset[metric],
                marker="o",
                color=COLORS[method],
                label=method,
                linestyle="-",
                linewidth=2,
                markersize=6.5,
            )

        ax.set_xticks([2, 3, 4])
        ax.set_xlabel("Concurrent concepts (n)")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="0.55", linewidth=0.8)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.spines[["bottom", "left"]].set_visible(False)

    axes[1].axhline(0.5, color="0.55", linestyle=":", linewidth=0.8)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Scaling curves, n=2–4 (bf16 pool, nested groups)")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
