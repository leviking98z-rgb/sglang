"""Cache a verified, unchanged serving environment on one allocated node."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True)
parser.add_argument("--archive", required=True)
parser.add_argument("--sha256", required=True)
parser.add_argument("--node", required=True)
args = parser.parse_args()
project = Path(args.project)
target = Path("/root/.cache") / ("qwen38-runtime-" + project.name)
staging = target.with_name(target.name + "-staging-01")
receipt = project / "results" / ("runtime-cache-" + args.node + ".json")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


try:
    assert not target.exists() and not staging.exists()
    source = Path(args.archive)
    original = source.stat()
    assert shutil.disk_usage(target.parent).free > original.st_size * 3 + 2**30
    staging.mkdir()
    archive = staging / "dependencies.tar"
    print(json.dumps({"phase": "copying dependencies", "bytes": original.st_size}), flush=True)
    shutil.copyfile(source, archive)
    after = source.stat()
    assert (original.st_size, original.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert sha256(archive) == args.sha256
    (staging / "venv").mkdir()
    subprocess.run(["tar", "-xf", str(archive), "-C", str(staging / "venv")], check=True)
    code_archive = project / "workdir/source-62d4f89.tar.gz"
    assert sha256(code_archive) == "8b049c9be9a02332b6a3ec576eae07ccacc71c6aeb18559d1340f7643299a64c"
    (staging / "sglang").mkdir()
    subprocess.run(["tar", "-xf", str(code_archive), "-C", str(staging / "sglang")], check=True)
    info = {
        "passed": True,
        "node": args.node,
        "runtime": str(target),
        "dependencies_sha256": args.sha256,
        "dependencies_bytes": original.st_size,
        "engine_commit": "62d4f8941a0d36d009a09ae86bee6647dcb2b871",
        "scope": "Node-local cache; unchanged dependencies and pinned source",
        "created_unix": time.time(),
    }
    (staging / "cache-manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    staging.rename(target)
    receipt.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)
except BaseException as exc:
    receipt.write_text(json.dumps({"passed": False, "error": repr(exc), "created_unix": time.time()}, indent=2) + "\n")
    raise
