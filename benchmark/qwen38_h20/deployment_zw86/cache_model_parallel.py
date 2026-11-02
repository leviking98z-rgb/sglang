"""Resume verified local shards and copy the remaining checkpoint with four streams."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True)
parser.add_argument("--node", required=True)
parser.add_argument("--attempt", required=True)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()
assert args.attempt.replace("-", "").isalnum()
assert 1 <= args.workers <= 4
project = Path(args.project)
source = (project / "checkpoints/Qwen3.8-Flash-Next").resolve()
runtime = Path("/root/.cache") / ("qwen38-runtime-" + project.name)
target = runtime / "model/Qwen3.8-Flash-Next"
staging = target.with_name(target.name + "-staging-01")
receipt = project / "results" / ("model-cache-" + args.node + ".json")
revision = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
started = time.time()
records = []
guard = threading.Lock()
lockfile = (target.parent / ".model-cache-writer.lock").open("a+")
fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def report(**extra):
    info = dict(
        node=args.node,
        source=str(source),
        target=str(target),
        revision=revision,
        attempt=args.attempt,
        workers=args.workers,
        started_unix=started,
        updated_unix=time.time(),
        verified_files=sorted(records, key=lambda r: r["file"]),
        weight_shards_verified=sum(
            r["expected_lfs_sha256"] is not None for r in records
        ),
        **extra,
    )
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(info, indent=2) + "\n")
    temporary.replace(receipt)
    print(
        json.dumps({k: v for k, v in info.items() if k != "verified_files"}), flush=True
    )
    return info


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


try:
    assert runtime.is_dir() and not target.exists() and staging.is_dir()
    old = json.loads(receipt.read_text())
    assert old["source"] == str(source) and old["target"] == str(target)
    assert old["revision"] == revision and old["phase"] == "copying"
    old_records = {r["file"]: r for r in old["verified_files"]}
    # Preserve the last sequential receipt as evidence before publishing progress.
    with (
        project / "results" / f"model-cache-{args.node}-before-{args.attempt}.json"
    ).open("x") as out:
        json.dump(old, out, indent=2)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    assert len(shards) == 131
    files = sorted(
        p
        for p in source.rglob("*")
        if p.is_file() and p.suffix not in (".lock", ".incomplete")
    )
    assert all(not p.is_symlink() for p in files)
    remaining_bytes = sum(
        p.stat().st_size for p in files if str(p.relative_to(source)) not in old_records
    )
    assert shutil.disk_usage(runtime).free > remaining_bytes + 20 * 2**30

    def copy(original):
        relative = original.relative_to(source)
        name = relative.as_posix()
        before = original.stat()
        expected = None
        if name in shards:
            lines = (
                (source / ".cache/huggingface/download" / (name + ".metadata"))
                .read_text()
                .splitlines()
            )
            assert lines[0] == revision
            expected = lines[1].strip('"')
            assert len(expected) == 64
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        begin = time.monotonic()
        if name in old_records:
            previous = old_records[name]
            assert destination.is_file() and not destination.is_symlink()
            assert destination.stat().st_size == previous["bytes"] == before.st_size
            actual = digest_file(destination)
            assert actual == previous["sha256"] and (
                expected is None or actual == expected
            )
            reused = True
        else:
            if destination.exists():
                # The stopped sequential writer may have left one incomplete file.
                quarantine = target.parent / f"incomplete-before-{args.attempt}"
                quarantine.mkdir(exist_ok=True)
                preserved = quarantine / relative.name
                assert not preserved.exists()
                destination.rename(preserved)
            temporary = destination.with_name(
                destination.name + "." + args.attempt + ".partial"
            )
            digest = hashlib.sha256()
            copied = 0
            with original.open("rb") as reader, temporary.open("xb") as writer:
                while chunk := reader.read(16 * 1024 * 1024):
                    writer.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            actual = digest.hexdigest()
            assert copied == before.st_size and (expected is None or actual == expected)
            temporary.rename(destination)
            reused = False
        after = original.stat()
        assert (before.st_size, before.st_mtime_ns, before.st_ino) == (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        )
        record = dict(
            file=name,
            bytes=before.st_size,
            sha256=actual,
            expected_lfs_sha256=expected,
            seconds=time.monotonic() - begin,
            reused_verified_local_file=reused,
        )
        with guard:
            records.append(record)
            if expected:
                report(
                    passed=False,
                    phase="copying",
                    last_file=name,
                    last_seconds=record["seconds"],
                )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(copy, files))
    assert sum(r["expected_lfs_sha256"] is not None for r in records) == 131
    info = report(passed=True, phase="verified", bytes=sum(r["bytes"] for r in records))
    (staging / "node-cache-manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    staging.rename(target)
    report(passed=True, phase="published", bytes=sum(r["bytes"] for r in records))
except BaseException as exc:
    report(passed=False, phase="failed", error=repr(exc))
    raise
