#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Values used in the previously approved adapter-ablation figure.
SKIP_WER = {
    0: 0.229,
    1: 0.197,
    2: 0.229,
    3: 0.206,
    4: 0.233,
    5: 0.194,
    6: 0.215,
    7: 0.199,
    8: 0.220,
    9: 0.204,
    10: 0.233,
    11: 0.242,
    12: 0.245,
    13: 0.275,
    14: 0.212,
    15: 0.237,
    16: 0.228,
    17: 0.212,
    18: 0.203,
    19: 0.198,
    20: 0.197,
    21: 0.358,
}

OUT_PATH = Path("/yangliusha02/Evaluate/LibriSpeech/v66_test2/wer_results/wer_scatter.pdf")

# The ten largest WER increases relative to the baseline.
IMPORTANT_LAYERS = {0, 2, 4, 10, 11, 12, 13, 15, 16, 21}


def main():
    fig, ax = plt.subplots(figsize=(12, 7))

    for layer, wer in SKIP_WER.items():
        selected = layer in IMPORTANT_LAYERS
        ax.scatter(
            [layer],
            [wer],
            s=145 if selected else 62,
            facecolors="black" if selected else "white",
            edgecolors="black",
            linewidths=1.6 if selected else 1.2,
            zorder=3,
        )

    # Show selected layer indices without labeling every layer.
    tick_positions = list(range(0, 22, 2)) + [21]
    tick_labels = [str(layer) for layer in range(0, 22, 2)] + ["21"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=14)
    ax.set_yticks([0.18, 0.22, 0.26, 0.30, 0.34, 0.38])
    ax.tick_params(axis="y", labelsize=14)

    ax.set_xlabel("Adapter layer skipped", fontsize=17, labelpad=12)
    ax.set_ylabel("WER", fontsize=17, labelpad=12)

    # Leave one empty column on both sides of the plotted points.
    ax.set_xlim(-1, 22)
    ax.set_ylim(0.17, 0.375)

    # Minimal black-and-white style with arrow axes.
    ax.set_frame_on(False)
    ax.grid(axis="y", color="#bdbdbd", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="black",
                markersize=10,
                linestyle="None",
                label="Top 10 important layers",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="white",
                markersize=7,
                linestyle="None",
                label="Other layers",
            ),
        ],
        loc="upper left",
        frameon=False,
        fontsize=17,
    )
    ax.annotate(
        "",
        xy=(22, 0.17),
        xytext=(-1, 0.17),
        arrowprops={"arrowstyle": "-|>", "linewidth": 1.4, "color": "black"},
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(-1, 0.375),
        xytext=(-1, 0.17),
        arrowprops={"arrowstyle": "-|>", "linewidth": 1.4, "color": "black"},
        annotation_clip=False,
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
