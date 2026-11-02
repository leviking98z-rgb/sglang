"""Sample GPU memory on the serving node during this visual qualification."""
import json
import subprocess
import time
from pathlib import Path

project = Path(PROJECT)
node = NODE
started = time.time()
samples = []
result_path = project / "results/visual-pressure-qualification.json"
output = project / "results" / ("visual-memory-" + node + "-04.json")
errors = []
reason = "deadline"
try:
    while time.time() - started < 1800:
        now = time.time()
        run = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        if run.returncode:
            errors.append({"unix": now, "error": run.stderr[-500:]})
        else:
            gpus = [[int(value.strip()) for value in line.split(",")]
                    for line in run.stdout.splitlines() if line.strip()]
            assert len(gpus) == 8, gpus
            samples.append({"unix": now, "gpu_mib": gpus})
        if result_path.exists() and result_path.stat().st_mtime >= started:
            reason = "visual qualification result published"
            break
        time.sleep(2)
finally:
    summary = []
    for index in range(8):
        rows = [row for sample in samples for row in sample["gpu_mib"]
                if row[0] == index]
        if rows:
            summary.append({"index": index, "total_mib": rows[0][1],
                            "peak_used_mib": max(row[2] for row in rows),
                            "min_free_mib": min(row[3] for row in rows)})
    output.write_text(json.dumps({
        "node": node, "started_unix": started, "completed_unix": time.time(),
        "reason": reason, "sampling_interval_seconds": 2,
        "sample_count": len(samples), "summary": summary, "errors": errors,
        "samples": samples,
        "scope": "NVML samples of this node; sub-second peaks can be missed",
    }, indent=2) + "\n")
