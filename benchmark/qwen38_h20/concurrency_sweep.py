"""Run a bounded, cache-isolated serving sweep from an allocated CPU client.

This does not change the engines, flush caches, or retry failed requests. The
existing SGLang benchmark supplies SSE parsing and latency/token accounting.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


SOURCE_SHA256 = "852c7d3076f58fc1e96f65021ce26b3a1dfd2b5839f1487f20fa18a8019b9935"
SOURCE_COMMIT = "a4a6eddd25343905245db7a8b54bcf24d9483c8b"


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def child(source, nonce, argv):
    from sglang.benchmark.datasets.random import RandomDataset

    original_load = RandomDataset.load

    def isolated_load(self, tokenizer, model_id=None):
        rows = original_load(self, tokenizer, model_id=model_id)
        for i, row in enumerate(rows):
            assert isinstance(row.prompt, list), "Use --tokenize-prompt"
            salt = hashlib.sha256(f"{nonce}:{i}".encode()).hexdigest()
            prefix = tokenizer.encode(salt + "\n", add_special_tokens=False)
            assert len(prefix) < row.prompt_len
            row.prompt = prefix + row.prompt[: row.prompt_len - len(prefix)]
            assert len(row.prompt) == row.prompt_len
        return rows

    RandomDataset.load = isolated_load
    original = source.read_bytes()
    assert hashlib.sha256(original).hexdigest() == SOURCE_SHA256
    text = original.decode()
    timer = "    benchmark_duration = time.perf_counter() - benchmark_start_time\n"
    anchor = "    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)\n"
    assert text.count(timer) == text.count(anchor) == 1
    # Exclude the post-run server_info request from measured request duration.
    text = text.replace(timer, "").replace(anchor, anchor + timer)
    active = 0
    peak = 0
    events = []

    def track_request(fn):
        async def tracked(*a, **kw):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            events.append([time.time(), active])
            try:
                return await fn(*a, **kw)
            finally:
                active -= 1
                events.append([time.time(), active])
        return tracked

    anchor = "ASYNC_REQUEST_FUNCS = {"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "async_request_sglang_generate = _track_request(async_request_sglang_generate)\n\n" + anchor)
    sys.argv = [str(source), *argv]
    try:
        exec(compile(text, str(source), "exec"), {"__name__": "__main__", "__file__": str(source),
                                                "_track_request": track_request})
    finally:
        if os.environ.get("SWEEP_CLIENT_TRACE"):
            write_json(Path(os.environ["SWEEP_CLIENT_TRACE"]),
                       dict(peak_client_requests=peak, final_client_requests=active, events=events))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--shapes", nargs="+", default=["16384:1024", "32768:1024"])
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    assert args.attempt.replace("-", "").replace("_", "").isalnum()
    assert all(1 <= c <= 256 for c in args.concurrency)
    assert args.num_prompts >= max(args.concurrency)
    root = args.project / "results" / args.attempt
    env = os.environ.copy()
    env.pop("CB_PANEL_CONTEXT", None)
    env.update(CUDA_VISIBLE_DEVICES="", SGLANG_IS_IN_CI="0", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", OMP_NUM_THREADS="8",
               PYTHONPATH=str(args.source_project / "workdir/sglang/python"))
    cuda_root = args.source_project / "workdir/venv/lib/python3.12/site-packages/nvidia/cu13"
    env["LD_LIBRARY_PATH"] = ":".join(["/opt/cudacompat13/usr/local/cuda-13.3/compat",
        "/opt/conda/lib", str(cuda_root / "lib"), env.get("LD_LIBRARY_PATH", "")])
    if not args.run:
        root.mkdir(exist_ok=False)
        logfile = args.project / "logs" / (args.attempt + ".log")
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--run"]
        with logfile.open("xb") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        stat = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(") ", 1)[1].split()
        record = dict(pid=proc.pid, pgid=proc.pid, start_ticks=int(stat[19]),
                      command=command, started_unix=time.time(), log=str(logfile), results=str(root))
        write_json(root / "process.json", record)
        print(json.dumps(record), flush=True)
        return

    source = args.source_project / "workdir/sglang/python/sglang/benchmark/serving.py"
    python = args.source_project / "workdir/venv/bin/python"
    model = args.source_project / "checkpoints/Qwen3.8-Flash-Next"
    manifest = dict(status="running", started_unix=time.time(), url=args.url,
        source_commit=SOURCE_COMMIT, source_sha256=SOURCE_SHA256, runs=[],
        workload="ShareGPT prompts repeated/truncated to exact length; unique SHA256 text prefix per request",
        cache_policy="No flush; independent request prefixes across runs; report actual cached tokens",
        timing="Request timer ends at asyncio.gather; post-run server_info excluded",
        generation="temperature=0, ignore_eos=true, exact fixed output budget",
        seed=20261102, num_prompts=args.num_prompts, deadline_unix=args.deadline_unix)
    try:
        assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
        for shape in args.shapes:
            isl, osl = map(int, shape.split(":"))
            assert isl + osl + 4 <= 262144
            for concurrency in args.concurrency:
                # A stop file stops new stages while allowing the current one to drain.
                if (args.project / "STOP").exists():
                    raise RuntimeError("Stop requested; no further stages admitted")
                remaining = args.deadline_unix - time.time()
                if remaining < 300:
                    raise RuntimeError("Less than 5 minutes left in service window; no new stage")
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(args.url + "/workers", timeout=15) as response:
                    workers = json.load(response)["workers"]
                assert len(workers) == 8 and all(w["is_healthy"] and w["load"] == 0 for w in workers), workers
                run_id = f"i{isl}-o{osl}-c{concurrency}"
                output = root / (run_id + ".jsonl")
                trace = root / (run_id + ".client.json")
                env["SWEEP_CLIENT_TRACE"] = str(trace)
                command = [str(python), str(Path(__file__).resolve()), "--child", str(source),
                    args.attempt + ":" + run_id,
                    "--backend", "sglang", "--base-url", args.url,
                    "--model", "Qwen/Qwen3.8-Flash-Next", "--tokenizer", str(model),
                    "--dataset-name", "random", "--random-input-len", str(isl),
                    "--random-output-len", str(osl), "--random-range-ratio", "1",
                    "--num-prompts", str(args.num_prompts), "--max-concurrency", str(concurrency),
                    "--request-rate", "inf", "--seed", "20261102", "--tokenize-prompt",
                    "--warmup-requests", "0", "--disable-tqdm", "--output-details",
                    "--cache-report", "--output-file", str(output)]
                run = dict(id=run_id, input_len=isl, output_len=osl, concurrency=concurrency,
                           started_unix=time.time(), command=command, router_preflight=workers)
                manifest["running"] = run
                write_json(root / "manifest.json", manifest)
                print(json.dumps({"starting": run_id, "unix": time.time()}), flush=True)
                with (root / (run_id + ".log")).open("x") as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=min(2400, max(1, args.deadline_unix - time.time())), check=False)
                run.update(returncode=result.returncode, ended_unix=time.time())
                manifest["runs"].append(run)
                manifest.pop("running", None)
                if result.returncode:
                    raise RuntimeError(f"{run_id} benchmark exit={result.returncode}; no retry")
                data = json.loads(output.read_text().splitlines()[-1])
                for key, expected in [("completed", args.num_prompts),
                        ("total_input_tokens", isl * args.num_prompts),
                        ("total_output_tokens", osl * args.num_prompts)]:
                    if data.get(key) != expected:
                        raise RuntimeError(f"{run_id}: {key}={data.get(key)}, expected {expected}")
                keys = ["completed", "duration", "total_input_tokens", "total_output_tokens",
                        "output_throughput", "mean_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
                        "mean_tpot_ms", "p95_tpot_ms", "p99_tpot_ms", "cache_report"]
                run["metrics"] = {k: data.get(k) for k in keys}
                client = json.loads(trace.read_text())
                run["metrics"].update(peak_client_requests=client["peak_client_requests"],
                                      mean_client_requests=data.get("concurrency"))
                assert client["final_client_requests"] == 0 and client["peak_client_requests"] <= concurrency
                cached = data.get("cache_report", {}).get("total_cached_tokens", 0)
                if cached > isl * args.num_prompts * 0.01:
                    raise RuntimeError(f"{run_id}: cache hit exceeds 1%; inspect before continuing")
                write_json(root / "manifest.json", manifest)
                print(json.dumps({"completed": run_id, "metrics": run["metrics"]}), flush=True)
        manifest["status"] = "complete"
        manifest["exit_code"] = 0
    except BaseException as exc:
        manifest.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["ended_unix"] = time.time()
        write_json(root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
