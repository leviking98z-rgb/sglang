# Qwen3.8-Flash-Next on 32 H20 GPUs

Operational scripts for four 8-GPU nodes, with a fixed source and model revision:

- SGLang main: `a4a6eddd25343905245db7a8b54bcf24d9483c8b`
- Model: `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Baseline: BF16 weights and BF16 recurrent state, without speculative decoding.
- PyTorch 2.14.1 (CUDA 13.0), FlashInfer 0.7.2, SGLang kernel 0.4.10.

`packages-tested.txt` records the observed package versions, with local paths
and editable references removed. It is an environment record, not a resolver lock.

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

The baseline uses one TP8 replica per node (four replicas). Candidate layouts
cover FlashInfer prefill, expert parallelism, two TP4 replicas per node,
and native NEXTN/MTP. FlashInfer decode on H20 (SM90) requires FP32 recurrent
state; use `--ssm-dtype float32` for that candidate. The other candidates keep
BF16 state and Triton decode. GPU-resident PLE embeddings and kernel tuning
can be evaluated on the best layout where memory permits. Supply candidate flags
after `--extra`; command records preserve the resolved CLI inputs.

Screening uses equal eight-GPU budgets and the geometric mean of output
tokens/s for both shapes at concurrency 64. This corresponds to concurrency 256
when scaling four such cohorts to 32 GPUs. The final comparison repeats the
complete 32-GPU protocol. Additional screens cover native interleaved SwiGLU
fusion and the FlashInfer CUTLASS BF16 MoE backend.

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
through the round-robin router. `prepare_benchmark.py` verifies the source SHA256
and generates a separate benchmark script with one timing change: stop the timer
immediately after all request tasks finish, before the post-run `/server_info`
query. The original benchmark and all engine source files remain untouched.
This removes metadata-query latency from the throughput denominator; the
unmodified upstream mode remains available with `--timing upstream`.
Comparisons must use the same timing mode on both configurations. Defaults:

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

Short requests also use a longer measurement window at concurrency 256:
`--shapes 1024:256 --concurrency 256 --prompt-multiplier 8 --repeats 3`.
Each repetition has 2,048 requests. Compare these repetitions separately from
the 512-request matrix points, and report the median and full observed range.

`run_stage.py` validates every worker with short and 8K-context prompts, verifies
the exact healthy router worker set before and after measurements, runs the full
matrix, then repeats the largest concurrency three times. The baseline and the
optimized deployment use this same workflow. `report.py` checks request and token
counts again and exports aggregate and per-request CSV files without deployment
addresses or generated text.

`profile_service.py` captures a separate six-step prefill/decode trace using
SGLang's native profiler API. Profiling requests are excluded from the serving
throughput measurements.

`start_tuning.py` runs the main branch's fused-MoE tuner on one otherwise idle
eight-GPU node, using the 74 configurations in `moe_search_space.json`. Install
Ray 2.55.1 and msgpack 1.2.4 without dependencies into `workdir/tuning_extra/`;
only the tuning process adds that directory to `PYTHONPATH`. Its Ray address is
`local`, with a unique temporary directory. The inference environment stays
unchanged. Publish a successful tuning result under
`$SGLANG_MOE_CONFIG_DIR/configs/triton_3_8_0/` and restart the candidate with that
environment variable. For EP configurations the microbenchmark's local expert
routing approximates the serving workload; accept a tuning change only after
an end-to-end serving comparison and response checks.

With `--fused-swiglu`, the helper creates a SHA-tagged derivative of the pinned
tuner that calls `fused_experts` directly with `fuse_swiglu_interleaved=True`.
The public `fused_moe` wrapper at this commit does not accept this argument.
Both source and derivative hashes are recorded; engine files remain untouched.
