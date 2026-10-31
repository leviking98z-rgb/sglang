import argparse
import json
import subprocess
import time

from runtime import PROJECT, PYTHON, environment

parser = argparse.ArgumentParser()
parser.add_argument("--attempt", required=True)
parser.add_argument("--url", required=True)
parser.add_argument("--workers", nargs="+", required=True)
parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 16, 64, 128, 256])
parser.add_argument("--shapes", nargs="+", default=["1024:256", "8192:1024"])
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--timing", choices=["upstream", "requests"], default="requests")
args = parser.parse_args()
if not args.attempt.replace("-", "").replace("_", "").isalnum():
    raise ValueError("invalid attempt")
record = PROJECT / "logs" / (args.attempt + ".process.json")
if record.exists():
    raise RuntimeError("attempt already exists")
command = [
    PYTHON,
    str(PROJECT / "scripts/run_benchmarks.py"),
    "--attempt",
    args.attempt,
    "--url",
    args.url,
    "--workers",
    *args.workers,
    "--concurrency",
    *map(str, args.concurrency),
    "--shapes",
    *args.shapes,
    "--repeats",
    str(args.repeats),
    "--timing",
    args.timing,
]
log_path = PROJECT / "logs" / (args.attempt + ".log")
with log_path.open("xb") as logfile:
    process = subprocess.Popen(
        command,
        stdout=logfile,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=environment(),
    )
info = {
    "pid": process.pid,
    "pgid": process.pid,
    "command": command,
    "log": str(log_path),
    "started_unix": time.time(),
}
with record.open("x") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info))
