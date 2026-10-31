# Qwen3.8-Flash-Next on 32 H20 GPUs

Operational scripts for four 8-GPU nodes, with a fixed source and model revision:

- SGLang main: `a4a6eddd25343905245db7a8b54bcf24d9483c8b`
- Model: `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- BF16 weights and BF16 recurrent state; no speculative decoding.
- PyTorch 2.14.1 (CUDA 13.0), FlashInfer 0.7.2, SGLang kernel 0.4.10.

## Runtime layout

Set `QWEN38_PROJECT` to a persistent project directory. Copy these Python scripts
to `$QWEN38_PROJECT/scripts/`. The scripts expect an editable installation of the
pinned SGLang source in `$QWEN38_PROJECT/workdir/venv/` and the pinned model in
`$QWEN38_PROJECT/checkpoints/Qwen3.8-Flash-Next/`. Create `logs/` and `results/`.

`runtime.py` uses this cluster's existing CUDA 13 driver compatibility libraries
at `/opt/cudacompat13/usr/local/cuda-13.3/compat` and C++ runtime at
`/opt/conda/lib`. Adjust those paths for another machine. Use a Python runtime
compatible with the installed PyTorch wheel. The tested runtime is Python 3.12.

Apply `cuda130-constraints.txt` when installing the main branch dependencies:

```bash
uv pip install --python "$QWEN38_PROJECT/workdir/venv/bin/python" \
  --prerelease=allow -c cuda130-constraints.txt -r /path/to/sglang/python/pyproject.toml
SGLANG_BUILD_RUST_EXTS=none uv pip install \
  --python "$QWEN38_PROJECT/workdir/venv/bin/python" \
  --no-deps --no-build-isolation -e /path/to/sglang/python
uv pip install --python "$QWEN38_PROJECT/workdir/venv/bin/python" sglang-router==0.3.2
```

The CUDA compiler, CRT and NVVM need matching versions. Unconstrained dependency
resolution selected CUDA 13.4 tools alongside PyTorch's CUDA 13.0 headers; this
failed FlashInfer's compiler/header compatibility check. The constraints pin the
13.0 toolchain. `runtime.py` also exposes pip's `lib/` through the conventional
`lib64/` path and creates the unversioned `libcudart.so` linker name.

Python bytecode and JIT caches live on each node's local disk. A shared-FUSE
bytecode read blocked all eight scheduler ranks on one node during startup;
moving `PYTHONPYCACHEPREFIX` to local storage allowed TileLang to import normally.

## Start and compare

Run commands on nodes owned by the current allocation, using the cluster's native
allocation command interface. `manage_service.py` starts a detached process and
records its exact command, PID, process group, start time and log. Attempt names
are unique and cannot overwrite earlier process records.

```bash
python3 "$QWEN38_PROJECT/scripts/manage_service.py" start \
  --attempt baseline-node0 --tp 8 --ep 1 \
  --extra --model-loader-extra-config '{"enable_multithread_load":true}'
```

The baseline uses one TP8 replica per node (four replicas). The candidate layouts
use FlashInfer linear attention, expert parallelism, two TP4 replicas per node,
and GPU-resident PLE embeddings where memory permits. Supply candidate flags
after `--extra`; command records preserve the resolved CLI inputs.

```bash
python3 "$QWEN38_PROJECT/scripts/manage_router.py" start \
  --attempt baseline-router --workers http://WORKER0:31000 http://WORKER1:31000 \
  http://WORKER2:31000 http://WORKER3:31000
python3 "$QWEN38_PROJECT/scripts/start_benchmark.py" \
  --attempt baseline-matrix --url http://ROUTER:30000 \
  --workers http://WORKER0:31000 http://WORKER1:31000 \
  http://WORKER2:31000 http://WORKER3:31000
```

Wait for all workers to pass health and response checks before benchmarking.
`smoke_services.py` checks model discovery and three deterministic short answers
on each worker, and saves the actual responses and server information. These
checks establish basic service behavior; they are not a model quality evaluation.

## Measurement

The benchmark invokes the pinned main branch's `sglang.benchmark.serving` module
through the round-robin router. Defaults:

- Exact 1024/256 and 8192/1024 input/output token lengths.
- Concurrency 1, 16, 64, 128 and 256; unlimited arrival rate.
- `max(8, 2 * concurrency)` requests per point; fixed seed `20261031`.
- Native token-ID prompts, fixed length range, ignored EOS, no progress display.
- One untimed full-length warmup on every worker for each shape, then explicit
  cache flushes on every worker before each timed run. The benchmark's own
  warmup is disabled so no worker retains a warmup prefix in the timed run.
- Completed request count and total input/output tokens must match the workload.

Each run retains the exact command, timings, raw per-request metrics and logs in
`results/<attempt>/`. A manifest records completion or the first failure. Report
output tokens/s together with concurrency, TTFT and TPOT, and retain repeated
measurements for the chosen operating point. Synthetic random-token workloads
measure serving performance and do not represent an application quality test.
