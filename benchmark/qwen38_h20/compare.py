"""Compare the fixed 19-measurement, 32-GPU protocol using matching samples."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

SHAPES = [(1024, 256), (8192, 1024)]
CONCURRENCY = [1, 16, 64, 128, 256]


def phase(row):
    if row["attempt"].endswith("-matrix"):
        return "matrix"
    if row["attempt"].endswith("-peak"):
        return "peak"
    assert int(row["num_prompts"]) == 2048
    return "steady_short"


def key(row):
    return (
        phase(row),
        int(row["input_len"]),
        int(row["output_len"]),
        int(row["concurrency"]),
        int(row["num_prompts"]),
        int(row["run"].rsplit("-r", 1)[1]),
    )


EXPECTED = Counter(
    [("matrix", i, o, c, max(8, 2 * c), 0) for i, o in SHAPES for c in CONCURRENCY]
    + [("peak", i, o, 256, 512, r) for i, o in SHAPES for r in range(3)]
    + [("steady_short", 1024, 256, 256, 2048, r) for r in range(3)]
)


def read(path):
    rows = list(csv.DictReader(path.open()))
    assert Counter(key(row) for row in rows) == EXPECTED
    assert {row["timing"] for row in rows} == {"requests"}
    assert len({row["benchmark_sha256"] for row in rows}) == 1
    assert len(rows[0]["benchmark_sha256"]) == 64
    for row in rows:
        n = int(row["num_prompts"])
        assert int(row["gpus"]) == 32 == int(row["tp"]) * int(row["replicas"])
        assert int(row["completed"]) == n
        assert int(row["total_input_tokens"]) == n * int(row["input_len"])
        assert int(row["total_output_tokens"]) == n * int(row["output_len"])
        assert float(row["output_throughput"]) > 0
    assert sum(int(row["completed"]) for row in rows) == 11088
    config = {
        name: rows[0][name]
        for name in ["configuration", "gpus", "tp", "ep", "replicas"]
    }
    assert all(
        all(row[name] == value for name, value in config.items()) for row in rows
    )
    return rows, config


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
            for metric in ["mean_ttft_ms", "mean_tpot_ms", "p99_ttft_ms", "p99_tpot_ms"]
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, base_config = read(args.baseline)
    opt, opt_config = read(args.optimized)
    assert base[0]["benchmark_sha256"] == opt[0]["benchmark_sha256"]
    result = {
        "baseline_configuration": base_config,
        "optimized_configuration": opt_config,
        "benchmark_sha256": base[0]["benchmark_sha256"],
        "csv_sha256": {
            "baseline": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            "optimized": hashlib.sha256(args.optimized.read_bytes()).hexdigest(),
        },
        "timing": "requests; excludes the post-run server_info query",
        "measurements_per_configuration": 19,
        "completed_requests_per_configuration": 11088,
        "range_note": "Observed min/max across three repetitions, not a confidence interval.",
        "primary": [],
        "matrix": [],
    }
    print(
        "| Input/output tokens | Requests/repetition | Baseline output token/s | Optimized output token/s | Change |"
    )
    print("|---|---:|---:|---:|---:|")
    for isl, osl in SHAPES:
        selected_phase = "steady_short" if isl == 1024 else "peak"
        b = summarize(
            [
                r
                for r in base
                if phase(r) == selected_phase and int(r["input_len"]) == isl
            ]
        )
        o = summarize(
            [
                r
                for r in opt
                if phase(r) == selected_phase and int(r["input_len"]) == isl
            ]
        )
        gain = 100 * (
            o["median_output_tokens_per_second"] / b["median_output_tokens_per_second"]
            - 1
        )
        count = 2048 if isl == 1024 else 512
        result["primary"].append(
            {
                "input_tokens": isl,
                "output_tokens": osl,
                "concurrency": 256,
                "requests_per_repetition": count,
                "baseline": b,
                "optimized": o,
                "throughput_change_percent": gain,
            }
        )
        print(
            f"| {isl}/{osl} | {count} | {b['median_output_tokens_per_second']:.1f} | {o['median_output_tokens_per_second']:.1f} | {gain:+.1f}% |"
        )
    opt_matrix = {key(r)[1:]: r for r in opt if phase(r) == "matrix"}
    for row in base:
        if phase(row) != "matrix":
            continue
        other = opt_matrix[key(row)[1:]]
        result["matrix"].append(
            {
                "input_tokens": int(row["input_len"]),
                "output_tokens": int(row["output_len"]),
                "concurrency": int(row["concurrency"]),
                "baseline_output_tps": float(row["output_throughput"]),
                "optimized_output_tps": float(other["output_throughput"]),
                "throughput_change_percent": 100
                * (
                    float(other["output_throughput"]) / float(row["output_throughput"])
                    - 1
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
