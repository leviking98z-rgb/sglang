"""Compare matched 16K/32K measurements on the same 32-GPU cohort."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

SHAPES = [(16384, 1024), (32768, 1024)]
CONCURRENCY = [1, 16, 64, 128, 256]
EXPECTED = Counter(
    [("matrix", i, o, c, max(8, 2 * c), 0) for i, o in SHAPES for c in CONCURRENCY]
    + [("peak", i, o, 256, 512, r) for i, o in SHAPES for r in range(3)]
)


def key(row):
    phase = row["attempt"].rsplit("-", 1)[1]
    return (
        phase,
        int(row["input_len"]),
        int(row["output_len"]),
        int(row["concurrency"]),
        int(row["num_prompts"]),
        int(row["run"].rsplit("-r", 1)[1]),
    )


def read(path, label, tp, ep, replicas):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert Counter(key(row) for row in rows) == EXPECTED
    assert len(rows) == 16
    assert sum(int(row["completed"]) for row in rows) == 4944
    for row in rows:
        assert row["configuration"] == label
        assert row["timing"] == "requests"
        assert int(row["gpus"]) == 32 == tp * replicas
        assert (int(row["tp"]), int(row["ep"]), int(row["replicas"])) == (
            tp, ep, replicas
        )
        count = int(row["num_prompts"])
        assert int(row["completed"]) == count
        assert int(row["total_input_tokens"]) == count * int(row["input_len"])
        assert int(row["total_output_tokens"]) == count * int(row["output_len"])
        assert float(row["output_throughput"]) > 0
    return rows


def summarize(rows):
    rates = [float(row["output_throughput"]) for row in rows]
    return {
        "repetitions": len(rows),
        "output_tokens_per_second": rates,
        "median_output_tokens_per_second": statistics.median(rates),
        "observed_min": min(rates),
        "observed_max": max(rates),
        **{
            "median_" + metric: statistics.median(float(row[metric]) for row in rows)
            for metric in [
                "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms",
                "mean_e2e_latency_ms", "p99_e2e_latency_ms",
            ]
        },
    }


def gain(a, b):
    return 100 * (b["median_output_tokens_per_second"] / a["median_output_tokens_per_second"] - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs = {"baseline": (8, 1, 4), "tp4": (4, 4, 8), "mtp3": (4, 4, 8)}
    paths = {name: args.results / name / "aggregate.csv" for name in configs}
    measurements = {name: read(paths[name], name, *values) for name, values in configs.items()}
    shas = {row["benchmark_sha256"] for rows in measurements.values() for row in rows}
    assert len(shas) == 1 and len(next(iter(shas))) == 64
    result = {
        "gpus": 32,
        "shapes": SHAPES,
        "concurrency": CONCURRENCY,
        "measurements_per_configuration": 16,
        "completed_requests_per_configuration": 4944,
        "benchmark_sha256": next(iter(shas)),
        "csv_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "timing": "requests; excludes the post-run server_info query",
        "range_note": "Observed min/max across three repetitions, not a confidence interval.",
        "primary": [],
        "matrix": [],
    }
    for i, o in SHAPES:
        summaries = {
            name: summarize([r for r in rows if key(r)[:3] == ("peak", i, o)])
            for name, rows in measurements.items()
        }
        entry = {
            "input_tokens": i, "output_tokens": o, "concurrency": 256,
            "requests_per_repetition": 512, "configurations": summaries,
            "tp4_vs_baseline_percent": gain(summaries["baseline"], summaries["tp4"]),
            "mtp3_vs_tp4_percent": gain(summaries["tp4"], summaries["mtp3"]),
            "mtp3_vs_baseline_percent": gain(summaries["baseline"], summaries["mtp3"]),
        }
        result["primary"].append(entry)
        print(json.dumps(entry))
        for c in CONCURRENCY:
            result["matrix"].append({
                "input_tokens": i, "output_tokens": o, "concurrency": c,
                "configurations": {
                    name: summarize([r for r in rows if key(r)[:4] == ("matrix", i, o, c)])
                    for name, rows in measurements.items()
                },
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
