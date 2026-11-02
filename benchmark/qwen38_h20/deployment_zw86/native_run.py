"""Persist exact native command arguments and receipts for outcome recovery."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


def execute(argv, key, timeout):
    argv = list(argv)
    separator = argv.index("--")
    argv[separator:separator] = ["--idempotency-key", key]
    record = ROOT / (key + ".argv.json")
    if record.exists():
        assert json.loads(record.read_text()) == argv, "Cannot reuse key for changed arguments"
    else:
        record.write_text(json.dumps(argv))
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        chunks = [exc.stdout or "", exc.stderr or ""]
        (ROOT / (key + ".receipt.txt")).write_text("\n".join(
            x.decode(errors="replace") if isinstance(x, bytes) else x for x in chunks
        ))
        raise RuntimeError(f"Native command outcome unknown; inspect or reuse exact key {key}") from exc
    (ROOT / (key + ".receipt.txt")).write_text(result.stdout + result.stderr)
    return result


def json_result(output):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        for line in reversed(output.splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise
