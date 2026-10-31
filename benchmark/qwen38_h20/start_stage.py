import argparse
import json
import subprocess
import time

from runtime import PROJECT, PYTHON, environment

parser = argparse.ArgumentParser()
parser.add_argument("--attempt", required=True)
parser.add_argument("arguments", nargs=argparse.REMAINDER)
args = parser.parse_args()
if not args.attempt.replace("-", "").replace("_", "").isalnum():
    raise ValueError("invalid attempt")
record = PROJECT / "logs" / (args.attempt + ".process.json")
if record.exists():
    raise RuntimeError("attempt already exists")
command = [PYTHON, str(PROJECT / "scripts/run_stage.py"), "--attempt", args.attempt]
command += args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
log = PROJECT / "logs" / (args.attempt + ".log")
with log.open("xb") as stream:
    process = subprocess.Popen(
        command,
        env=environment(),
        stdout=stream,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
info = {
    "pid": process.pid,
    "pgid": process.pid,
    "command": command,
    "log": str(log),
    "started_unix": time.time(),
}
with record.open("x") as stream:
    json.dump(info, stream, indent=2)
print(json.dumps(info))
