import argparse
import concurrent.futures
import json
import subprocess
import time
import urllib.request

from prepare_benchmark import prepare
from runtime import MODEL, MODEL_NAME, PROJECT, PYTHON, environment

parser = argparse.ArgumentParser()
parser.add_argument("--attempt", required=True)
parser.add_argument("--url", required=True)
parser.add_argument("--workers", nargs="+", required=True)
parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 16, 64, 128, 256])
parser.add_argument("--shapes", nargs="+", default=["1024:256", "8192:1024"])
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--timing", choices=["upstream", "requests"], default="requests")
args = parser.parse_args()
if args.timing == "requests":
    benchmark_path, benchmark_provenance = prepare()
    benchmark_entry = [str(benchmark_path)]
else:
    benchmark_entry = ["-m", "sglang.benchmark.serving"]
    benchmark_provenance = {"change": "unmodified upstream timing"}
result_dir = PROJECT / "results" / args.attempt
result_dir.mkdir(exist_ok=False)
manifest = {
    "attempt": args.attempt,
    "url": args.url,
    "workers": args.workers,
    "concurrency": args.concurrency,
    "shapes": args.shapes,
    "repeats": args.repeats,
    "started_unix": time.time(),
    "runs": [],
    "warmups": [],
    "timing": args.timing,
    "benchmark_provenance": benchmark_provenance,
    "cache_policy": "warm every worker, then flush every worker before each timed run",
}
(result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def flush(url):
    request = urllib.request.Request(
        url + "/flush_cache?timeout=60", data=b"", method="POST"
    )
    with urllib.request.urlopen(request, timeout=70) as response:
        assert response.status == 200


def warmup(url, input_len, output_len):
    payload = {
        "input_ids": [1000 + (i * 7919 + 20261031) % 200000 for i in range(input_len)],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
    }
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=1200) as response:
        data = json.load(response)
    assert data["meta_info"]["completion_tokens"] == output_len
    return {
        "url": url,
        "input_len": input_len,
        "output_len": output_len,
        "started_unix": started,
        "seconds": time.time() - started,
        "meta_info": data["meta_info"],
    }


try:
    for shape in args.shapes:
        input_len, output_len = map(int, shape.split(":"))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(args.workers)
        ) as pool:
            manifest["warmups"].extend(
                pool.map(
                    lambda url, isl=input_len, osl=output_len: warmup(url, isl, osl),
                    args.workers,
                )
            )
        (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for concurrency in args.concurrency:
            for repeat in range(args.repeats):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(args.workers)
                ) as pool:
                    list(pool.map(flush, args.workers))
                num_prompts = max(8, 2 * concurrency)
                run_id = f"i{input_len}-o{output_len}-c{concurrency}-r{repeat}"
                command = [
                    PYTHON,
                    *benchmark_entry,
                    "--backend",
                    "sglang",
                    "--base-url",
                    args.url,
                    "--model",
                    MODEL_NAME,
                    "--tokenizer",
                    MODEL,
                    "--dataset-name",
                    "random",
                    "--random-input-len",
                    str(input_len),
                    "--random-output-len",
                    str(output_len),
                    "--random-range-ratio",
                    "1",
                    "--num-prompts",
                    str(num_prompts),
                    "--max-concurrency",
                    str(concurrency),
                    "--request-rate",
                    "inf",
                    "--seed",
                    "20261031",
                    "--tokenize-prompt",
                    "--warmup-requests",
                    "0",
                    "--disable-tqdm",
                    "--output-details",
                    "--output-file",
                    str(result_dir / (run_id + ".jsonl")),
                ]
                run = {
                    "id": run_id,
                    "input_len": input_len,
                    "output_len": output_len,
                    "concurrency": concurrency,
                    "num_prompts": num_prompts,
                    "command": command,
                    "started_unix": time.time(),
                }
                print(json.dumps({"starting": run}), flush=True)
                manifest["running"] = run
                (result_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2)
                )
                with open(result_dir / (run_id + ".log"), "x") as logfile:
                    bench_env = environment()
                    bench_env["SGLANG_IS_IN_CI"] = "0"
                    process = subprocess.run(
                        command,
                        stdout=logfile,
                        stderr=subprocess.STDOUT,
                        env=bench_env,
                        timeout=2400,
                        check=False,
                    )
                run.update(returncode=process.returncode, ended_unix=time.time())
                manifest["runs"].append(run)
                manifest.pop("running", None)
                (result_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2)
                )
                if process.returncode:
                    raise RuntimeError(f"{run_id} exited {process.returncode}")
                lines = (result_dir / (run_id + ".jsonl")).read_text().splitlines()
                data = json.loads(lines[-1])
                if data.get("completed") != num_prompts:
                    raise RuntimeError(
                        f"{run_id} incomplete: {data.get('completed')} / {num_prompts}"
                    )
                if data.get("total_input_tokens") != input_len * num_prompts:
                    raise RuntimeError(f"{run_id} unexpected input token count")
                if data.get("total_output_tokens") != output_len * num_prompts:
                    raise RuntimeError(f"{run_id} unexpected output token count")
                print(
                    json.dumps(
                        {
                            "completed": run_id,
                            "metrics": {
                                k: v
                                for k, v in data.items()
                                if k
                                in [
                                    "completed",
                                    "output_throughput",
                                    "mean_ttft_ms",
                                    "p99_ttft_ms",
                                    "mean_tpot_ms",
                                    "p99_tpot_ms",
                                    "duration",
                                ]
                            },
                        }
                    ),
                    flush=True,
                )
    manifest["status"] = "complete"
except BaseException as exc:
    manifest.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
    raise
finally:
    manifest["ended_unix"] = time.time()
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
