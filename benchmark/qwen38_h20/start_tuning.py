"""Run main's focused MoE tuner on one otherwise-idle, owned eight-GPU node."""

import argparse
import hashlib
import json
import subprocess
import time

from runtime import MODEL, PROJECT, PYTHON, environment

parser = argparse.ArgumentParser()
parser.add_argument("--attempt", required=True)
parser.add_argument("--tp", type=int, required=True)
parser.add_argument("--ep", type=int, required=True)
parser.add_argument("--fused-swiglu", action="store_true")
parser.add_argument(
    "--batches", nargs="+", type=int, default=[1, 16, 32, 64, 128, 1024, 4096, 8192]
)
args = parser.parse_args()
if not args.attempt.replace("-", "").replace("_", "").isalnum():
    raise ValueError("invalid attempt")
record = PROJECT / "logs" / (args.attempt + ".process.json")
if record.exists():
    raise RuntimeError("attempt already exists")
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    text=True,
).strip()
if apps:
    raise RuntimeError("GPU processes remain; stop and verify owned services first")
directory = PROJECT / "results" / args.attempt
directory.mkdir(exist_ok=False)
tuner = (
    PROJECT
    / "workdir/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py"
)
tuner_directory = tuner.parent
source = tuner.read_bytes()
source_sha = hashlib.sha256(source).hexdigest()
assert source_sha == "342555c255d43528e64b45426f970599ca101edd47208f92c9596d36706077fd"
provenance = {"source_sha256": source_sha, "fused_swiglu": args.fused_swiglu}
if args.fused_swiglu:
    needle = "                block_shape=block_shape,\n            )"
    text = source.decode()
    import_needle = (
        "from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe"
    )
    assert text.count(import_needle) == 1
    text = text.replace(
        import_needle,
        "from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts as fused_moe",
    )
    assert text.count(needle) == 1
    generated = text.replace(
        needle,
        "                block_shape=block_shape,\n"
        "                fuse_swiglu_interleaved=True,\n            )",
    ).encode()
    generated_sha = hashlib.sha256(generated).hexdigest()
    tuner = PROJECT / (
        "scripts/tuning_fused_moe_interleaved_" + generated_sha[:12] + ".py"
    )
    if tuner.exists():
        assert tuner.read_bytes() == generated
    else:
        temporary = tuner.with_suffix(".tmp-" + str(time.time_ns()))
        temporary.write_bytes(generated)
        temporary.replace(tuner)
    provenance["generated_sha256"] = generated_sha
    provenance["entrypoint"] = (
        "fused_experts (supports fuse_swiglu_interleaved); wrapper fused_moe does not"
    )
(directory / "tuner_provenance.json").write_text(json.dumps(provenance, indent=2))
command = [
    PYTHON,
    str(tuner),
    "--model",
    MODEL,
    "--tp-size",
    str(args.tp),
    "--ep-size",
    str(args.ep),
    "--dtype",
    "auto",
    "--batch-sizes",
    *map(str, args.batches),
    "--search-space-file",
    str(PROJECT / "scripts/moe_search_space.json"),
    "--tune",
]
env = environment()
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
env["PYTHONPATH"] = (
    str(PROJECT / "workdir/tuning_extra")
    + ":"
    + str(tuner_directory)
    + ":"
    + env.get("PYTHONPATH", "")
)
env["RAY_ADDRESS"] = "local"
env["RAY_TMPDIR"] = "/tmp/q38-" + hashlib.sha256(args.attempt.encode()).hexdigest()[:8]
env["RAY_USAGE_STATS_ENABLED"] = "0"
env["RAY_OVERRIDE_RESOURCES"] = json.dumps({"CPU": 16})
env["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"] = str(2 * 1024**3)
env["RAY_DEDUP_LOGS"] = "0"
log = PROJECT / "logs" / (args.attempt + ".log")
with log.open("xb") as stream:
    process = subprocess.Popen(
        command,
        env=env,
        cwd=directory,
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
    "directory": str(directory),
    "started_unix": time.time(),
    "ray_address": "local",
    "ray_cpus": 16,
    "tuner_provenance": provenance,
    "ray_tmpdir": env["RAY_TMPDIR"],
    "pythonpath": env["PYTHONPATH"],
    "search_space_sha256": hashlib.sha256(
        (PROJECT / "scripts/moe_search_space.json").read_bytes()
    ).hexdigest(),
}
with record.open("x") as stream:
    json.dump(info, stream, indent=2)
print(json.dumps(info))
