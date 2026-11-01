"""Record runtime log checks once each long-context phase has completed."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--gpus", type=int, choices=[16, 24, 32], default=32)
    parser.add_argument("--phase", choices=["baseline", "tp4", "mtp3"])
    args = parser.parse_args()
    pending = {"baseline": args.gpus // 8, "tp4": args.gpus // 4, "mtp3": args.gpus // 4}
    if args.phase:
        pending = {args.phase: pending[args.phase]}
    deadline = time.monotonic() + args.timeout
    error_pattern = re.compile(
        r"Scheduler hit an exception|Traceback \(most recent call last\)|"
        r"CUDA error:|RuntimeError:"
    )
    while pending:
        for phase, count in list(pending.items()):
            results = args.project / "results"
            postflight_path = results / (phase + "-postflight.json")
            if not postflight_path.exists():
                continue
            completed = []
            for stage_path in results.glob("lc-" + phase + "*/stage.json"):
                try:
                    stage = json.loads(stage_path.read_text())
                except json.JSONDecodeError:
                    continue
                if stage.get("status") == "complete":
                    completed.append(stage)
            if not completed:
                continue
            assert len(completed) == 1, phase
            stage = completed[0]
            logs = sorted((args.project / "logs").glob(f"lc-{phase}-n*r*-01.log"))
            assert len(logs) == count, (phase, len(logs), count)
            workers = []
            for path in logs:
                raw = path.read_bytes()
                text = raw.decode(errors="replace")
                workers.append(
                    {
                        "attempt": path.stem,
                        "log_sha256": hashlib.sha256(raw).hexdigest(),
                        "log_bytes": len(raw),
                        "runtime_error_events": len(error_pattern.findall(text)),
                        "retraction_log_lines": sum(
                            "#retracted_reqs:" in line for line in text.splitlines()
                        ),
                        "compiler_data_race_warnings": text.count("Warning: Data race detected:"),
                    }
                )
            report = {
                "phase": phase,
                "stage_status": stage["status"],
                "stage_attempt": stage["attempt"],
                "observed_unix": time.time(),
                "worker_count": count,
                "workers": workers,
                "runtime_error_events": sum(w["runtime_error_events"] for w in workers),
                "retraction_log_lines": sum(w["retraction_log_lines"] for w in workers),
                "compiler_data_race_warnings": sum(w["compiler_data_race_warnings"] for w in workers),
                "scope": "Completed phase logs after worker postflight; counts of matching log events.",
                "patterns": {
                    "runtime_errors": error_pattern.pattern,
                    "retractions": "#retracted_reqs:",
                },
            }
            output = results / (phase + "-runtime-validation.json")
            with output.open("x") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
            print(json.dumps(report), flush=True)
            del pending[phase]
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Incomplete phases: {sorted(pending)}")
            time.sleep(3)


if __name__ == "__main__":
    main()
