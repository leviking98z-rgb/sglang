"""Export checked aggregate and per-request metrics without deployment addresses."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--attempts", nargs="+", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--gpus", type=int, required=True)
parser.add_argument("--tp", type=int, required=True)
parser.add_argument("--ep", type=int, required=True)
parser.add_argument("--replicas", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert args.gpus == args.tp * args.replicas
metrics = [
    "duration",
    "completed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_e2e_latency_ms",
    "p99_e2e_latency_ms",
]
rows, requests = [], []
for attempt in args.attempts:
    directory = args.results / attempt
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("benchmark attempt is not complete: " + attempt)
    for run in manifest["runs"]:
        if run.get("returncode") != 0:
            raise ValueError("failed benchmark run: " + run["id"])
        raw = (directory / (run["id"] + ".jsonl")).read_bytes()
        data = json.loads(raw.decode().splitlines()[-1])
        count = run["num_prompts"]
        assert data["completed"] == count
        assert data["total_input_tokens"] == count * run["input_len"]
        assert data["total_output_tokens"] == count * run["output_len"]
        assert not any(data["errors"])
        row = {
            "configuration": args.label,
            "gpus": args.gpus,
            "tp": args.tp,
            "ep": args.ep,
            "replicas": args.replicas,
            "attempt": attempt,
            "run": run["id"],
            "input_len": run["input_len"],
            "output_len": run["output_len"],
            "concurrency": run["concurrency"],
            "num_prompts": count,
            "timing": manifest.get("timing", "upstream"),
            "benchmark_sha256": manifest.get("benchmark_provenance", {}).get(
                "generated_sha256", ""
            ),
            **{key: data[key] for key in metrics},
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
        }
        rows.append(row)
        assert all(
            len(data[key]) == count
            for key in ["input_lens", "output_lens", "ttfts", "itls"]
        )
        for index in range(count):
            assert data["input_lens"][index] == run["input_len"]
            assert data["output_lens"][index] == run["output_len"]
            requests.append(
                {
                    "configuration": args.label,
                    "attempt": attempt,
                    "run": run["id"],
                    "request": index,
                    "input_len": data["input_lens"][index],
                    "output_len": data["output_lens"][index],
                    "ttft_ms": data["ttfts"][index] * 1000,
                    "first_to_last_token_ms": sum(data["itls"][index]) * 1000,
                    "stream_tpot_ms": sum(data["itls"][index])
                    * 1000
                    / (run["output_len"] - 1),
                }
            )
if not rows:
    raise ValueError("no completed measurements")
args.output.mkdir(parents=True, exist_ok=True)
for name, values in [("aggregate.csv", rows), ("requests.csv", requests)]:
    with (args.output / name).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
print(json.dumps({"measurements": len(rows), "requests": len(requests)}))
