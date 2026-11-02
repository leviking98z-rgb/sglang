"""Run small rollback checks, then hold all eight real PLE table pairs together."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

project = Path(os.environ["QWEN38_PROJECT"])
sys.path.insert(0, str(project / "scripts"))
from runtime import PYTHON, environment

parser = argparse.ArgumentParser()
parser.add_argument("--attempt", default="ple-registered-8rank-01")
parser.add_argument("--qualifier", default="qualify_ple_registered.py")
args = parser.parse_args()
attempt = args.attempt
assert attempt.replace("-", "").isalnum()
receipt = project / "results" / (attempt + ".json")
script = project / "scripts" / args.qualifier
started = time.time()
processes = []
streams = []
try:
    small = project / "results" / (attempt + "-small-rollback.json")
    with (project / "logs" / (attempt + "-small-rollback.log")).open("xb") as log:
        result = subprocess.run(
            [
                "/usr/bin/numactl",
                "--interleave=all",
                PYTHON,
                str(script),
                "--device",
                "0",
                "--receipt",
                str(small),
            ],
            env=environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    assert result.returncode == 0, "Small allocation/rollback check failed"
    check = json.loads(small.read_text())
    assert check["passed"] and check["rollback_exact"] and check["cleanup_exact"]
    barrier = project / "results" / (attempt + "-barrier")
    barrier.mkdir()
    for device in range(8):
        destination = project / "results" / f"{attempt}-rank{device}.json"
        log = (project / "logs" / f"{attempt}-rank{device}.log").open("xb")
        streams.append(log)
        argv = [
            "/usr/bin/numactl",
            "--interleave=all",
            PYTHON,
            str(script),
            "--device",
            str(device),
            "--receipt",
            str(destination),
            "--full",
            "--barrier-dir",
            str(barrier),
            "--peers",
            "8",
        ]
        processes.append(
            subprocess.Popen(
                argv,
                env=environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
    deadline = time.monotonic() + 600
    while any(proc.poll() is None for proc in processes):
        assert time.monotonic() < deadline, "Eight-rank qualification exceeded deadline"
        assert all(
            proc.poll() in (None, 0) for proc in processes
        ), "A full-table rank failed"
        time.sleep(1)
    ranks = [
        json.loads((project / "results" / f"{attempt}-rank{i}.json").read_text())
        for i in range(8)
    ]
    assert all(
        j["passed"] and j["full"] and j["cleanup_exact"] and j["concurrent_peers"] == 8
        for j in ranks
    )
    result = dict(
        passed=True,
        small_rollback=check,
        ranks=ranks,
        registration_chunk_mb=256,
        numa_policy="interleave=all",
        simultaneously_registered_bytes=sum(
            sum(r["bytes"] for r in j["results"]) for j in ranks
        ),
        started_unix=started,
        completed_unix=time.time(),
    )
except BaseException as exc:
    result = dict(
        passed=False, error=repr(exc), started_unix=started, completed_unix=time.time()
    )
    raise
finally:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    for stream in streams:
        stream.close()
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
