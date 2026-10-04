"""Plot the decomposition triptych in expression units through n=4."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "resources" / "34b_joint_expression_summary.csv"
OUTPUT_PATH = ROOT / "resources" / "36_triptych_expression_n2_n4.png"

METHODS = ("gds-blend", "blend", "common-blend")
NS = (2, 3, 4)
LABELS = {
    "gds-blend": "difference only",
    "blend": "full blend",
    "common-blend": "common only",
}
COLORS = {
    "gds-blend": "navy",
    "blend": "tab:blue",
    "common-blend": "tab:red",
}


def main() -> None:
    summary = pd.read_csv(SUMMARY_PATH)
    required = {"method", "n", "joint", "all_positive"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    indexed = summary.set_index(["method", "n"])
    expected = pd.MultiIndex.from_product([METHODS, NS])
    missing_cells = expected.difference(indexed.index)
    if not missing_cells.empty:
        raise ValueError(f"Missing method/n cells: {list(missing_cells)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    panels = (
        (axes[0], "joint", "mean joint concept expression gain"),
        (axes[1], "all_positive", "share with ALL concepts improved (expression)"),
    )
    width = 0.24

    for ax, metric, ylabel in panels:
        for offset, method in zip((-width, 0, width), METHODS):
            values = [indexed.loc[(method, n), metric] for n in NS]
            ax.bar(
                [i + offset for i in range(len(NS))],
                values,
                width=width,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set_xticks(range(len(NS)), [f"n={n}" for n in NS])
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="gray", linewidth=0.8)

    axes[0].legend(fontsize=8)
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    fig.suptitle("decomposition triptych in the concept-expression track, n=2–4")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
