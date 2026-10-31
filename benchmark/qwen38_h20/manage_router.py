import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from runtime import PROJECT, PYTHON, environment

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["start", "stop", "status"])
parser.add_argument("--attempt", required=True)
parser.add_argument("--port", type=int, default=30000)
parser.add_argument("--prometheus-port", type=int)
parser.add_argument(
    "--policy", choices=["round_robin", "power_of_two", "cache_aware"],
    default="round_robin",
)
parser.add_argument("--workers", nargs="+")
args = parser.parse_args()
if not args.attempt.replace("-", "").replace("_", "").isalnum():
    raise ValueError("invalid attempt")
record = PROJECT / "logs" / (args.attempt + ".process.json")
if args.action == "start":
    if record.exists() or not args.workers:
        raise RuntimeError("existing attempt or missing workers")
    metrics_port = args.prometheus_port or (args.port - 1000)
    if not 1 <= metrics_port <= 65535 or metrics_port == args.port:
        raise ValueError("choose a distinct valid --prometheus-port")
    for port in [args.port, metrics_port]:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
    command = [
        PYTHON,
        "-m",
        "sglang_router.launch_router",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--prometheus-port",
        str(metrics_port),
        "--policy",
        args.policy,
        "--worker-urls",
        *args.workers,
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
        "attempt": args.attempt,
        "pid": process.pid,
        "pgid": process.pid,
        "command": command,
        "port": args.port,
        "prometheus_port": metrics_port,
        "started_unix": time.time(),
        "log": str(log_path),
    }
    info["process_start_ticks"] = int(
        (Path("/proc") / str(process.pid) / "stat")
        .read_text()
        .rsplit(")", 1)[1]
        .split()[19]
    )
    with record.open("x") as f:
        json.dump(info, f, indent=2)
else:
    info = json.loads(record.read_text())
    d = Path("/proc") / str(info["pid"])
    alive = False
    if d.exists():
        stat = (d / "stat").read_text().rsplit(")", 1)[1].split()
        ticks = int(stat[19])
        boot_unix = time.time() - float(Path("/proc/uptime").read_text().split()[0])
        same_start = (
            ticks == info["process_start_ticks"]
            if "process_start_ticks" in info
            else abs(
                boot_unix + ticks / os.sysconf("SC_CLK_TCK") - info["started_unix"]
            )
            < 10
        )
        cmdline = (d / "cmdline").read_bytes().decode(errors="replace")
        named_router = (
            "sglang_router.launch_router" in cmdline
            or cmdline.split("\0", 1)[0].strip() == "sglang::router"
        )
        alive = (
            stat[0] != "Z"
            and same_start
            and named_router
            and os.getpgid(info["pid"]) == info["pgid"]
        )
    if args.action == "stop" and alive:
        if os.getpgid(info["pid"]) != info["pgid"]:
            raise RuntimeError("process group changed")
        os.killpg(info["pid"], signal.SIGTERM)
        for _ in range(100):
            if not d.exists() or (d / "stat").read_text().split()[2] == "Z":
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("router did not stop")
        info["stopped_unix"] = time.time()
        record.write_text(json.dumps(info, indent=2))
        alive = False
    info["alive"] = alive
    info["tail"] = Path(info["log"]).read_text(errors="replace")[-5000:]
print(json.dumps(info))
