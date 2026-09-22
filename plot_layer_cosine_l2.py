#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path("/yangliusha02/tests/qwen")
COSINE_CSV = BASE_DIR / "layer_cosine_summary.csv"
L2_CSV = BASE_DIR / "layer_l2_summary.csv"
OUT_PATH = BASE_DIR / "layer_cosine_l2_curve.png"


cosine_df = pd.read_csv(COSINE_CSV)
l2_df = pd.read_csv(L2_CSV)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharex=True)

# Cosine similarity subplot.
ax = axes[0]
layers = cosine_df["layer"]
cosine_mean = cosine_df["frame_cosine_mean"]
cosine_std = cosine_df["frame_cosine_std"]
ax.plot(
    layers,
    cosine_mean,
    marker="o",
    linewidth=2.2,
    label="mean",
)
ax.fill_between(
    layers,
    cosine_mean - cosine_std,
    cosine_mean + cosine_std,
    alpha=0.18,
    label="+/- 1 std",
)
ax.set_xlabel("Qwen Audio Encoder layer", fontsize=16)
ax.set_ylabel("Cosine similarity", fontsize=16)
ax.set_title("Representation cosine similarity", fontsize=21, pad=12)
ax.set_ylim(0.70, 1.02)
ax.legend(loc="upper left", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.35)

# Absolute L2 subplot.
ax = axes[1]
layers = l2_df["layer"]
l2_mean = l2_df["absolute_l2_mean"]
l2_std = l2_df["absolute_l2_std"]
ax.plot(
    layers,
    l2_mean,
    marker="o",
    linewidth=2.2,
    label="mean",
)
ax.fill_between(
    layers,
    l2_mean - l2_std,
    l2_mean + l2_std,
    alpha=0.18,
    label="+/- 1 std",
)
ax.set_xlabel("Qwen Audio Encoder layer", fontsize=16)
ax.set_ylabel("L2 distance", fontsize=16)
ax.set_title("Representation absolute L2 distance", fontsize=21, pad=12)
ax.legend(loc="upper left", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.35)

for ax in axes:
    ax.set_xticks(range(1, 25))
    ax.set_xlim(0, 25)
    ax.tick_params(axis="both", labelsize=13)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=400, bbox_inches="tight")
print(f"Saved plot to {OUT_PATH}")
