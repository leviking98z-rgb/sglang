"""Validate every replica, then benchmark a fixed serving configuration."""

import argparse
import concurrent.futures
import json
import subprocess
import time
import urllib.request

from runtime import PROJECT, PYTHON, environment

parser = argparse.ArgumentParser()
parser.add_argument("--attempt", required=True)
parser.add_argument("--url", required=True)
parser.add_argument("--workers", nargs="+", required=True)
parser.add_argument("--router-attempt")
parser.add_argument("--router-port", type=int, default=30000)
parser.add_argument("--health-wait", type=int, default=900)
parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 16, 64, 128, 256])
parser.add_argument("--shapes", nargs="+", default=["1024:256", "8192:1024"])
parser.add_argument("--peak-repeats", type=int, default=3)
parser.add_argument("--timing", choices=["upstream", "requests"], default="requests")
args = parser.parse_args()
if not args.attempt.replace("-", "").replace("_", "").isalnum():
    raise ValueError("invalid attempt")
directory = PROJECT / "results" / args.attempt
directory.mkdir(exist_ok=False)
manifest = {
    "attempt": args.attempt,
    "workers": args.workers,
    "url": args.url,
    "started_unix": time.time(),
    "status": "waiting_for_workers",
    "steps": [],
}


def save():
    (directory / "stage.json").write_text(json.dumps(manifest, indent=2))


def health(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=10) as response:
            return {"url": url, "ready": response.status == 200}
    except OSError as exc:
        return {"url": url, "ready": False, "error": str(exc)}


def router_snapshot(label):
    if args.url.rstrip("/") in {url.rstrip("/") for url in args.workers}:
        return
    expected = {url.rstrip("/") for url in args.workers}
    deadline = time.monotonic() + (120 if label == "before_matrix" else 0)
    while True:
        with urllib.request.urlopen(args.url + "/workers", timeout=15) as response:
            data = json.load(response)
        healthy = {
            worker["url"].rstrip("/")
            for worker in data["workers"]
            if worker["is_healthy"]
        }
        if healthy == expected and len(data["workers"]) == len(expected):
            break
        if time.monotonic() >= deadline:
            raise AssertionError("router worker set changed or never became ready")
        time.sleep(2)
    data["verified_unix"] = time.time()
    with (directory / ("router_workers_" + label + ".json")).open("x") as stream:
        json.dump(data, stream, indent=2)


def step(name, arguments):
    entry = {"name": name, "command": [PYTHON, *arguments], "started_unix": time.time()}
    manifest["steps"].append(entry)
    manifest["status"] = name
    save()
    with (directory / (name + ".log")).open("xb") as output:
        result = subprocess.run(
            entry["command"],
            env=environment(),
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    entry.update(returncode=result.returncode, ended_unix=time.time())
    save()
    if result.returncode:
        raise RuntimeError(name + " failed; inspect its log")


save()
try:
    deadline = time.monotonic() + args.health_wait
    while True:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(args.workers)
        ) as pool:
            checks = list(pool.map(health, args.workers))
        manifest["last_health"] = checks
        save()
        if all(x["ready"] for x in checks):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("not all requested replicas became healthy")
        time.sleep(10)
    scripts = PROJECT / "scripts"
    step(
        "short_smoke",
        [
            str(scripts / "smoke_services.py"),
            "--workers",
            *args.workers,
            "--output",
            str(directory / "short_smoke.json"),
        ],
    )
    smoke_lengths = sorted(
        {max(8000, int(shape.split(":")[0]) - 128) for shape in args.shapes}
    )
    for filler_tokens in smoke_lengths:
        name = f"long_context_smoke_{filler_tokens}"
        step(
            name,
            [
                str(scripts / "long_context_smoke.py"),
                "--workers",
                *args.workers,
                "--filler-tokens",
                str(filler_tokens),
                "--output",
                str(directory / (name + ".json")),
            ],
        )
    if args.router_attempt:
        step(
            "start_router",
            [
                str(scripts / "manage_router.py"),
                "start",
                "--attempt",
                args.router_attempt,
                "--port",
                str(args.router_port),
                "--workers",
                *args.workers,
            ],
        )
        for _ in range(60):
            if health(args.url)["ready"]:
                break
            time.sleep(2)
        else:
            raise RuntimeError("router did not become healthy")
    router_snapshot("before_matrix")
    step(
        "matrix",
        [
            str(scripts / "run_benchmarks.py"),
            "--attempt",
            args.attempt + "-matrix",
            "--url",
            args.url,
            "--workers",
            *args.workers,
            "--concurrency",
            *map(str, args.concurrency),
            "--shapes",
            *args.shapes,
            "--timing",
            args.timing,
        ],
    )
    router_snapshot("after_matrix")
    if args.peak_repeats:
        step(
            "peak_repeats",
            [
                str(scripts / "run_benchmarks.py"),
                "--attempt",
                args.attempt + "-peak",
                "--url",
                args.url,
                "--workers",
                *args.workers,
                "--concurrency",
                str(max(args.concurrency)),
                "--shapes",
                *args.shapes,
                "--repeats",
                str(args.peak_repeats),
                "--timing",
                args.timing,
            ],
        )
        router_snapshot("after_peak")
    manifest["status"] = "complete"
except Exception as exc:
    manifest["status"] = "failed"
    manifest["error"] = repr(exc)
    raise
finally:
    manifest["ended_unix"] = time.time()
    save()
