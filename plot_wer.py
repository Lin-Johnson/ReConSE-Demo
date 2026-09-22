#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def condition_sort_key(path: Path):
    if path.name == "baseline":
        return (-1, -1)

    match = re.fullmatch(r"skip_layer_(\d+)", path.name)
    if match:
        return (0, int(match.group(1)))

    return (1, path.name)


def read_wer_results(results_dir: Path):
    records = []

    for condition_dir in sorted(results_dir.iterdir(), key=condition_sort_key):
        if not condition_dir.is_dir():
            continue

        json_path = condition_dir / "results.json"
        if not json_path.exists():
            print(f"Skip: {json_path} does not exist")
            continue

        try:
            with json_path.open("r", encoding="utf-8") as file:
                result = json.load(file)
            wer = float(result["WER"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"Skip: failed to read {json_path}: {error}")
            continue

        records.append((condition_dir.name, wer))

    return records


def plot_wer(records, output_path: Path):
    if not records:
        raise RuntimeError("No valid results.json files were found.")

    labels = [name for name, _ in records]
    wers = [wer for _, wer in records]
    x_values = list(range(len(records)))

    baseline_indices = [i for i, name in enumerate(labels) if name == "baseline"]
    ablation_indices = [i for i, name in enumerate(labels) if name != "baseline"]

    fig, ax = plt.subplots(figsize=(15, 7))

    if ablation_indices:
        ax.scatter(
            [x_values[i] for i in ablation_indices],
            [wers[i] for i in ablation_indices],
            color="#377eb8",
            s=70,
            label="Skip one adapter layer",
            zorder=3,
        )

    if baseline_indices:
        ax.scatter(
            [x_values[i] for i in baseline_indices],
            [wers[i] for i in baseline_indices],
            color="#e41a1c",
            marker="*",
            s=180,
            label="Baseline",
            zorder=4,
        )

    for x, (label, wer) in zip(x_values, records):
        ax.annotate(
            f"{wer:.3f}",
            (x, wer),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels(
        ["baseline" if label == "baseline" else label.replace("skip_layer_", "skip ") for label in labels],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("WER")
    ax.set_xlabel("Inference condition")
    ax.set_title("WER by Adapter-Layer Ablation")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    y_min = min(wers)
    y_max = max(wers)
    margin = max((y_max - y_min) * 0.25, 0.01)
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path(
            "/yangliusha02/Evaluate/LibriSpeech/v66_test2/wer_results"
        ),
        help="Directory containing baseline/ and skip_layer_XX/ folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to results_dir/wer_scatter.png.",
    )
    args = parser.parse_args()

    output_path = args.output or args.results_dir / "wer_scatter.png"
    records = read_wer_results(args.results_dir)

    print("WER results:")
    for name, wer in records:
        print(f"  {name}: {wer:.3f}")

    plot_wer(records, output_path)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
