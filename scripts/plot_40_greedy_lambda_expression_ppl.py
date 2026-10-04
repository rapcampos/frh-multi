"""Plot notebook 40's joint expression and perplexity results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "resources" / "40_greedy_lambda_summary.csv"
OUTPUT_PATH = ROOT / "resources" / "40_greedy_lambda_expression_ppl.png"
VERTICAL_OUTPUT_PATH = (
    ROOT / "resources" / "40_greedy_lambda_expression_ppl_vertical.png"
)

METHODS = ("blend", "vote", "schedule")
COLORS = {
    "blend": "tab:blue",
    "vote": "tab:orange",
    "schedule": "tab:green",
}
LABELS = {
    "blend": "Frame blending",
    "vote": "Score voting",
    "schedule": "Round-robin scheduling",
}
LINESTYLES = {0.0: "-", 0.4: "--"}


def main() -> None:
    summary = pd.read_csv(SUMMARY_PATH)
    required = {"method", "n", "lam", "joint", "ppl_ratio"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    panels = (
        (
            axes[0],
            "joint",
            "Mean joint concept expression gain",
        ),
        (
            axes[1],
            "ppl_ratio",
            "Mean perplexity ratio",
        ),
    )

    for ax, metric, ylabel in panels:
        for method in METHODS:
            for lam, linestyle in LINESTYLES.items():
                subset = summary[
                    (summary["method"] == method) & (summary["lam"] == lam)
                ].sort_values("n")
                ax.plot(
                    subset["n"],
                    subset[metric],
                    color=COLORS[method],
                    linestyle=linestyle,
                    marker="o",
                    linewidth=2.2,
                    markersize=6.5,
                )

        ax.set_xlabel("Concurrent concepts (n)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([2, 3, 4])
        ax.grid(axis="y", color="0.88", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].axhline(0, color="0.45", linewidth=1)
    axes[0].set_ylim(bottom=0)
    axes[1].axhline(1, color="0.55", linestyle=":", linewidth=1)
    axes[1].axhline(2.5, color="0.35", linestyle=":", linewidth=1.2)
    axes[1].annotate(
        "2.5× fluency guardrail",
        xy=(4.02, 2.5),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="0.35",
        fontsize=9,
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker="o",
            linewidth=2.2,
            label=LABELS[method],
        )
        for method in METHODS
    ]
    lambda_handles = [
        Line2D(
            [0],
            [0],
            color="0.2",
            linestyle=linestyle,
            linewidth=2.2,
            label=f"λ = {lam:g}",
        )
        for lam, linestyle in LINESTYLES.items()
    ]
    fig.legend(
        handles=method_handles + lambda_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.94), w_pad=3)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")

    vertical_fig, vertical_axes = plt.subplots(2, 1, figsize=(7.2, 9), sharex=True)
    vertical_panels = (
        (vertical_axes[0], "joint", "Mean joint concept expression gain"),
        (vertical_axes[1], "ppl_ratio", "Mean perplexity ratio"),
    )
    for ax, metric, ylabel in vertical_panels:
        for method in METHODS:
            for lam, linestyle in LINESTYLES.items():
                subset = summary[
                    (summary["method"] == method) & (summary["lam"] == lam)
                ].sort_values("n")
                ax.plot(
                    subset["n"],
                    subset[metric],
                    color=COLORS[method],
                    linestyle=linestyle,
                    marker="o",
                    linewidth=2.2,
                    markersize=6.5,
                )

        ax.set_ylabel(ylabel)
        ax.set_xticks([2, 3, 4])
        ax.grid(axis="y", color="0.88", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    vertical_axes[0].axhline(0, color="0.45", linewidth=1)
    vertical_axes[0].set_ylim(bottom=0)
    vertical_axes[1].axhline(1, color="0.55", linestyle=":", linewidth=1)
    vertical_axes[1].axhline(2.5, color="0.35", linestyle=":", linewidth=1.2)
    vertical_axes[1].annotate(
        "2.5× fluency guardrail",
        xy=(4.02, 2.5),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="0.35",
        fontsize=9,
    )
    vertical_axes[1].set_xlabel("Concurrent concepts (n)")

    vertical_fig.legend(
        handles=method_handles + lambda_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    vertical_fig.tight_layout(rect=(0, 0.12, 1, 1), h_pad=3)
    vertical_fig.savefig(VERTICAL_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plt.close(vertical_fig)
    print(OUTPUT_PATH)
    print(VERTICAL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
