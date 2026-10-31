"""Run a bounded node-local candidate sweep and restore its starting configuration."""

import argparse
import concurrent.futures
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from runtime import PROJECT, PYTHON, environment


def run(plan):
    attempt = plan["attempt"]
    if not attempt.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid attempt")
    directory = PROJECT / "results" / attempt
    directory.mkdir(exist_ok=False)
    scripts = PROJECT / "scripts"
    state = {"status": "waiting_for_control", "plan": plan, "candidates": []}
    active = list(plan["initial_services"])
    changed = False

    def save():
        state["updated_unix"] = time.time()
        (directory / "sweep.json").write_text(json.dumps(state, indent=2) + "\n")

    def command(label, argv, env):
        with (directory / (label + ".log")).open("xb") as output:
            subprocess.run(
                [PYTHON, *map(str, argv)], env=env, stdout=output,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True,
            )

    def wait_healthy(urls):
        deadline = time.monotonic() + plan.get("health_wait", 900)
        while time.monotonic() < deadline:
            for service in active:
                record = json.loads((PROJECT / "logs" / (service + ".process.json")).read_text())
                proc = Path("/proc") / str(record["pid"])
                try:
                    exited = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
                except FileNotFoundError:
                    exited = True
                if exited:
                    raise RuntimeError(service + " exited during startup; inspect its log")
            ready = []
            for url in urls:
                try:
                    with urllib.request.urlopen(url + "/health", timeout=5) as response:
                        ready.append(response.status == 200)
                except OSError:
                    ready.append(False)
            if all(ready):
                return
            time.sleep(5)
        raise RuntimeError("replicas did not become healthy within the startup budget")

    def stop(label):
        for index, service in enumerate(active):
            command(
                f"{label}-stop-{index}",
                [scripts / "manage_service.py", "stop", "--attempt", service],
                environment(),
            )
        active.clear()

    def start(label, config):
        env = environment()
        env.update(config.get("environment", {}))
        commands = []
        tp = int(config.get("tp", 4))
        replicas = int(config.get("replicas", 2))
        if tp * replicas != 8:
            raise ValueError("screening must use exactly eight GPUs")
        for index in range(replicas):
            service = f"{attempt}-{label}-{index}"
            active.append(service)
            argv = [
                scripts / "manage_service.py", "start", "--attempt", service,
                "--tp", str(tp), "--ep", str(config.get("ep", 4)),
                "--base-gpu", str(index * tp), "--port", str(31000 + index),
                "--max-running", str(config.get("max_running", 64)),
                "--graph-bs", str(config.get("graph_bs", 64)),
                "--mem-fraction", str(config.get("mem_fraction", 0.82)),
                "--ssm-dtype", config.get("ssm_dtype", "bfloat16"),
                "--chunk-size", str(config.get("chunk_size", 8192)),
                "--extra", "--model-loader-extra-config",
                '{"enable_multithread_load":true}',
                "--linear-attn-prefill-backend", "flashinfer",
                *config.get("extra", []),
            ]
            commands.append((f"{label}-start-{index}", argv))
        state["active_services"] = active.copy()
        save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(command, label, argv, env) for label, argv in commands]
            for future in futures:
                future.result()
        wait_healthy(
            plan.get("restore_workers", plan["workers"])
            if label == "restored" else plan["workers"]
        )

    save()
    try:
        control = PROJECT / "results" / plan["wait_for_benchmark"] / "manifest.json"
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            try:
                data = json.loads(control.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                data = {}
            if data.get("status") == "complete":
                break
            if data.get("status") == "failed":
                raise RuntimeError("control benchmark failed")
            time.sleep(5)
        else:
            raise RuntimeError("control benchmark did not finish")
        state["status"] = "screening"
        save()
        for config in plan["candidates"]:
            label = config["name"]
            if not label.replace("-", "").isalnum():
                raise ValueError("invalid candidate name")
            entry = {"name": label, "started_unix": time.time(), "config": config}
            state["candidates"].append(entry)
            state["current_candidate"] = label
            save()
            stop(label)
            changed = True
            try:
                start(label, config)
                stage = f"{attempt}-{label}"
                entry["stage"] = stage
                command(
                    label + "-stage",
                    [
                        scripts / "run_stage.py", "--attempt", stage,
                        "--url", plan["url"], "--workers", *plan["workers"],
                        "--concurrency", "64", "--peak-repeats", "1",
                        "--health-wait", "60", "--timing", "requests",
                    ], environment(),
                )
                entry["status"] = "complete"
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                entry.update(status="failed", error=repr(exc))
            entry["ended_unix"] = time.time()
            save()
        state["status"] = "screening_complete"
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        if changed:
            state["restore_status"] = "starting"
            save()
            stop("restore")
            start("restored", plan["restore_config"])
            state["restored_services"] = active.copy()
            state["restore_status"] = "healthy"
        state["ended_unix"] = time.time()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    with open(args.plan) as stream:
        plan = json.load(stream)
    if args.detach:
        attempt = plan["attempt"]
        record = PROJECT / "logs" / (attempt + ".process.json")
        if record.exists():
            raise RuntimeError("attempt already exists")
        with (PROJECT / "logs" / (attempt + ".log")).open("xb") as output:
            command = [PYTHON, __file__, "--plan", args.plan]
            process = subprocess.Popen(
                command, env=environment(), stdout=output, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        info = {"pid": process.pid, "pgid": process.pid, "command": command,
                "started_unix": time.time()}
        with record.open("x") as stream:
            json.dump(info, stream, indent=2)
        print(json.dumps(info))
    else:
        run(plan)
