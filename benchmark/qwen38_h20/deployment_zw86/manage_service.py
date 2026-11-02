import argparse
import errno
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from runtime import MODEL, MODEL_NAME, PROJECT, PYTHON, environment

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["start", "status", "stop"])
parser.add_argument("--attempt", required=True)
parser.add_argument("--tp", type=int, default=8)
parser.add_argument("--ep", type=int, default=1)
parser.add_argument("--base-gpu", type=int, default=0)
parser.add_argument("--port", type=int, default=31000)
parser.add_argument("--max-running", type=int, default=128)
parser.add_argument("--graph-bs", type=int)
parser.add_argument("--chunk-size", type=int, default=8192)
parser.add_argument("--mem-fraction", type=float, default=0.85)
parser.add_argument("--ssm-dtype", choices=["bfloat16", "float32"], default="bfloat16")
parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
args = parser.parse_args()
if not args.attempt.replace("-", "").replace("_", "").isalnum():
    raise ValueError("invalid attempt")
record = PROJECT / "logs" / (args.attempt + ".process.json")

if args.action == "start":
    if record.exists():
        raise RuntimeError("attempt already exists; inspect its original record")
    for port in [args.port]:
        deadline = time.monotonic() + 60
        while True:
            try:
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("0.0.0.0", port))
                break
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE or time.monotonic() >= deadline:
                    raise
                time.sleep(1)
    command = [
        PYTHON,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--served-model-name",
        MODEL_NAME,
        "--tp-size",
        str(args.tp),
        "--ep-size",
        str(args.ep),
        "--base-gpu-id",
        str(args.base_gpu),
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--mamba-ssm-dtype",
        args.ssm_dtype,
        "--max-running-requests",
        str(args.max_running),
        "--cuda-graph-max-bs-decode",
        str(args.graph_bs or args.max_running),
        "--chunked-prefill-size",
        str(args.chunk_size),
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--watchdog-timeout",
        "1200",
    ] + args.extra
    log_path = PROJECT / "logs" / (args.attempt + ".log")
    run_env = environment()
    with open(log_path, "xb") as logfile:
        process = subprocess.Popen(
            command,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=run_env,
        )
    info = {
        "attempt": args.attempt,
        "pid": process.pid,
        "pgid": process.pid,
        "command": command,
        "port": args.port,
        "started_unix": time.time(),
        "log": str(log_path),
        "hostname": socket.gethostname(),
        "runtime_environment": {
            key: run_env.get(key)
            for key in [
                "CUDA_HOME",
                "LD_LIBRARY_PATH",
                "OMP_NUM_THREADS",
                "SGLANG_CACHE_DIR",
                "SGLANG_JIT_CACHE_DIR",
                "PYTHONPYCACHEPREFIX",
                "SGLANG_MOE_CONFIG_DIR",
                "SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED",
                "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE",
            ]
        },
    }
    with open(record, "x") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info))
else:
    info = json.loads(record.read_text())
    process_dir = Path("/proc") / str(info["pid"])
    alive = process_dir.exists()
    if alive:
        cmdline = (
            (process_dir / "cmdline")
            .read_bytes()
            .decode(errors="replace")
            .replace("\0", " ")
        )
        alive = "sglang.launch_server" in cmdline and MODEL in cmdline
    if args.action == "stop" and alive:
        if os.getpgid(info["pid"]) != info["pgid"]:
            raise RuntimeError("process group changed")
        os.killpg(info["pgid"], signal.SIGTERM)
        for _ in range(150):
            if not process_dir.exists():
                break
            time.sleep(0.2)
        if process_dir.exists():
            state = (process_dir / "stat").read_text().split()[2]
            if state != "Z":
                raise RuntimeError("original service has not exited after SIGTERM")
        info["stopped_unix"] = time.time()
        record.write_text(json.dumps(info, indent=2))
        alive = False
    info["alive"] = alive
    info["tail"] = Path(info["log"]).read_text(errors="replace")[-7000:]
    print(json.dumps(info))
