"""Copy pinned checkpoint bytes sequentially to a private node-local cache."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True)
parser.add_argument("--node", required=True)
args = parser.parse_args()
project = Path(args.project)
source = (project / "checkpoints/Qwen3.8-Flash-Next").resolve()
runtime = Path("/root/.cache") / ("qwen38-runtime-" + project.name)
target = runtime / "model/Qwen3.8-Flash-Next"
staging = target.with_name(target.name + "-staging-01")
receipt = project / "results" / ("model-cache-" + args.node + ".json")
started = time.time()
revision = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
records = []


def report(**extra):
    info = dict(
        node=args.node, source=str(source), target=str(target),
        revision=revision, started_unix=started, updated_unix=time.time(),
        verified_files=records, **extra,
    )
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(info, indent=2) + "\n")
    tmp.replace(receipt)
    print(json.dumps({k: v for k, v in info.items() if k != "verified_files"}), flush=True)
    return info


try:
    assert runtime.is_dir() and not target.exists() and not staging.exists()
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) == 131
    files = sorted(p for p in source.rglob("*") if p.is_file())
    assert all(not p.is_symlink() for p in files)
    required = sum(p.stat().st_size for p in files)
    assert shutil.disk_usage(runtime).free > required + 20 * 2**30
    target.parent.mkdir(exist_ok=True)
    staging.mkdir()
    for original in files:
        relative = original.relative_to(source)
        if relative.suffix in (".lock", ".incomplete"):
            continue
        before = original.stat()
        expected = None
        metadata = source / ".cache/huggingface/download" / (str(relative) + ".metadata")
        if relative.as_posix() in shards:
            lines = metadata.read_text().splitlines()
            assert lines[0] == revision, relative
            expected = lines[1].strip('"')
            assert len(expected) == 64 and all(c in "0123456789abcdef" for c in expected)
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        copied = 0
        begin = time.monotonic()
        with original.open("rb") as reader, destination.open("xb") as writer:
            while chunk := reader.read(16 * 1024 * 1024):
                writer.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        after = original.stat()
        assert (before.st_size, before.st_mtime_ns, before.st_ino) == (
            after.st_size, after.st_mtime_ns, after.st_ino
        ), relative
        actual = digest.hexdigest()
        assert copied == before.st_size and (expected is None or actual == expected), relative
        records.append(dict(
            file=relative.as_posix(), bytes=copied, sha256=actual,
            expected_lfs_sha256=expected, seconds=time.monotonic() - begin,
        ))
        if expected:
            report(passed=False, phase="copying", weight_shards_verified=sum(
                r["expected_lfs_sha256"] is not None for r in records
            ), last_file=relative.as_posix(), last_seconds=records[-1]["seconds"])
    assert sum(r["expected_lfs_sha256"] is not None for r in records) == 131
    info = report(passed=True, phase="verified", bytes=sum(r["bytes"] for r in records))
    (staging / "node-cache-manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    staging.rename(target)
    report(passed=True, phase="published", bytes=sum(r["bytes"] for r in records))
except BaseException as exc:
    report(passed=False, phase="failed", error=repr(exc))
    raise
