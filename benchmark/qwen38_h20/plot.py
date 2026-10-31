"""Plot the full matrix from checked baseline and optimized CSV exports."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--optimized", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--baseline-label", default="Baseline")
parser.add_argument("--optimized-label", default="Optimized")
args = parser.parse_args()
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for path, label, color in [
    (args.baseline, args.baseline_label, "#737373"),
    (args.optimized, args.optimized_label, "#1269b0"),
]:
    with path.open() as stream:
        rows = [
            row for row in csv.DictReader(stream) if row["attempt"].endswith("-matrix")
        ]
    assert len(rows) == 10
    assert all(row["timing"] == "requests" and row["gpus"] == "32" for row in rows)
    for index, (isl, osl) in enumerate([(1024, 256), (8192, 1024)]):
        group = sorted(
            [row for row in rows if int(row["input_len"]) == isl],
            key=lambda row: int(row["concurrency"]),
        )
        assert [int(row["concurrency"]) for row in group] == [1, 16, 64, 128, 256]
        assert all(
            int(row["output_len"]) == osl
            and int(row["num_prompts"]) == max(8, 2 * int(row["concurrency"]))
            for row in group
        )
        x = [int(row["concurrency"]) for row in group]
        for col, metric in enumerate(["output_throughput", "mean_ttft_ms"]):
            axis = axes[index, col]
            axis.plot(
                x,
                [float(row[metric]) for row in group],
                "o-",
                color=color,
                label=label,
            )
            axis.set_title(f"{isl} input / {osl} output tokens")
            axis.set_xlabel("Concurrency")
            axis.set_xticks(x)
            axis.grid(alpha=0.2)
axes[0, 0].set_ylabel("Output tokens/s")
axes[1, 0].set_ylabel("Output tokens/s")
axes[0, 1].set_ylabel("Mean TTFT (ms)")
axes[1, 1].set_ylabel("Mean TTFT (ms)")
axes[0, 0].legend()
fig.suptitle("Qwen3.8-Flash-Next, 32 NVIDIA H20 GPUs")
args.output.parent.mkdir(parents=True, exist_ok=True)
for suffix in [".png", ".svg"]:
    fig.savefig(args.output.with_suffix(suffix), dpi=180)
