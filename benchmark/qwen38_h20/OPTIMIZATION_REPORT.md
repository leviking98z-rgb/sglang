# Qwen3.8-Flash-Next: H20 throughput tuning and serving fixes

These experiments target aggregate output throughput on NVIDIA H20 while retaining BF16 weights and BF16 recurrent state. They distinguish deployment configuration gains, concurrency gains, and correctness/memory fixes. The figures below come from separate matched experiments; their percentages must not be multiplied together.

## Environment and selected configuration

- Measured main baseline: `a4a6eddd25343905245db7a8b54bcf24d9483c8b` (2026-10-31).
- Model: `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- PyTorch 2.14.1 + CUDA 13.0, FlashInfer 0.7.2, SGLang kernel 0.4.10, Triton 3.8.0, Transformers 5.19.0.
- Four eight-H20 hosts: **eight TP4/EP4 replicas**, two replicas per host, round-robin routing.
- FlashInfer linear-attention prefill, Triton decode/verification, original measured Triton MoE tuning table.
- Native NEXTN/MTP: **three steps, four draft/verification tokens, top-k one**. SGLang resolves this through its EAGLE implementation.
- BF16 weights/KV/recurrent state; 8192-token prefill chunks; static memory fraction 0.82; requested running limit and decode graph batch limit 64.
- Native 262144-token context limit retained. Request slots are not a guarantee that every slot can hold a full-length context.
- CPU PLE offload, local bytecode/JIT caches. The final serving deployment additionally uses NUMA interleaving and bounded PLE/image allocations; no independent throughput gain is attributed to these operational changes.

## 1. Layout, backends, MoE tuning and MTP

All 32-H20 configurations used the same measurement protocol. Primary values are medians of three repetitions at global concurrency 256. Short repetitions contain 2048 requests; 8K repetitions contain 512 requests.

| Input / output tokens | TP8 baseline | Tuned TP4, no MTP | Tuned TP4 + MTP3 | Final vs baseline |
|---|---:|---:|---:|---:|
| 1024 / 256 | 7967.2 | 9677.1 | **11768.0** | **+47.7%** |
| 8192 / 1024 | 5731.3 | 7785.8 | **8879.3** | **+54.9%** |

Units are aggregate output tokens/s. TP8 baseline means four TP8/EP1 replicas; tuned TP4 means eight TP4/EP4 replicas plus the stated backend, memory, graph, and MoE settings. The combined configuration gain cannot be assigned solely to EP, the MoE table, or one kernel.

An independent same-eight-GPU confirmation isolates MTP3 on the tuned layout: three 512-request repetitions per shape at C64 gave 2496.3→3078.7 output tokens/s for 1024/256 (**+23.3%**) and 1967.2→2383.6 for 8192/1024 (**+21.2%**). These eight-GPU figures are not the final 32-GPU totals.

Evidence: [round 1](README.md), [round 2 and ablations](round2/README.md), [MTP confirmation](round2/confirmation/comparison.json), [32-GPU comparison](round2/comparison-vs-initial.json). Each final configuration completed 19 measurements and 11088 timed requests with checked token counts.

## 2. Matched long-context comparison on 16 H20

The same two hosts ran all three configurations. These are medians of three C256 repetitions with 512 requests each; do not compare their absolute throughput to the separate 32-GPU deployment.

| Input / output tokens | TP8 baseline | Tuned TP4 | Tuned TP4 + MTP3 | MTP alone | Total gain |
|---|---:|---:|---:|---:|---:|
| 16384 / 1024 | 1982.3 | 2736.1 | **3055.2** | +11.7% | **+54.1%** |
| 32768 / 1024 | 1076.3 | 1510.6 | **1690.3** | +11.9% | **+57.0%** |

MTP increased mean TPOT by 0.5% at 16K and 9.2% at 32K relative to tuned TP4 despite improving aggregate throughput. Each configuration completed 16 measurements and 4944 timed requests. [Full latency matrix, repetitions and limitations](bj80-longctx/README.md).

## 3. Admission concurrency on the final 32-H20 service

This separate six-stage sweep uses the already deployed eight-replica MTP3 service. It does not change model settings. Each stage has 256 text requests, exact 1024-token output, temperature zero, and ignored EOS. ShareGPT-derived prompts are repeated/truncated to exact lengths with an independent text prefix per request; all stages measured **zero cached tokens**. No cache flush or server restart was performed. This sweep has one repetition per point and no explicit warmup; it is an operating-point measurement, not a confidence interval or proof of saturation.

| Input | HTTP concurrency | Output tokens/s | Mean TTFT (s) | p95 TTFT (s) | Mean TPOT (ms) | p95 TPOT (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 16K | 16 | 2552.5 | 1.421 | 1.924 | 4.746 | 7.006 |
| 16K | 32 | 3469.8 | 1.653 | 3.707 | 7.095 | 11.157 |
| 16K | 64 | **4426.9** | 2.680 | 7.655 | 10.831 | 17.513 |
| 32K | 16 | 1969.4 | 2.536 | 3.007 | 5.467 | 8.439 |
| 32K | 32 | 2514.2 | 3.131 | 6.821 | 9.232 | 14.709 |
| 32K | 64 | **2933.7** | 5.550 | 14.558 | 15.366 | 28.144 |

C16→C64 improved throughput **73.4% at 16K** and **49.0% at 32K**. All **1536/1536** requests succeeded with exact input/output token counts. Request-lifecycle counters reached the configured 16/32/64 peaks; average concurrency was 15.65/30.20/59.49 at 16K and 15.63/30.87/60.94 at 32K. Router `/workers` load remained zero during some streaming `/generate` requests, so it was not used as the source of actual request concurrency.

For matching text workloads, C64 is the best throughput point tested in this sweep. It has higher latency and is not a validated setting for 192K input plus 64K output or many-image requests. The final deployment had 55 effective request slots and 579008 KV tokens per replica. A frozen application benchmark's session admission count is different from concurrent model HTTP requests.

Evidence: [aggregate CSV](concurrency-20261102/aggregate.csv), [per-request CSV](concurrency-20261102/requests.csv), [method and provenance](concurrency-20261102/method.json). Timing ends after the last request response, before server-info queries; import/dataset preparation is excluded. Original artifacts were retained and independently SHA256-verified; the public export omits addresses, allocation identities and generated text.

## 4. Serving fixes found during tuning

### Empty grouped Gemma RMSNorm input

An idle attention-DP rank can pass shape `(0, 10240)` with group size 2560. The existing launcher computes a zero-block CUDA grid and fails with `CUDA error: invalid argument`. Return after input validation when the computed grid is empty. The nonempty kernel is unchanged. Original H20 regression coverage passed 33 tests including BF16/FP16 numerical checks, empty/nonempty output, supplied output tensors, and CUDA Graph capture/replay. This enabled evaluation of attention-DP candidates; **no standalone throughput gain** is claimed.

### Grayscale JPEG channel normalization

GPU JPEG decoding returned one channel for grayscale images, while the processor expects RGB tensors. Explicit `ImageReadMode.RGB` aligns the GPU path with the existing PIL RGB conversion. Checks covered grayscale and RGB JPEGs, including rectangular source images; output channel shape and pixel values matched torchvision's explicit RGB decode. This is a correctness fix.

### Bounded GPU image preprocessing with CPU feature transport

A 47-page request failed in Transformers preprocessing at the final GPU `torch.cat(processed_images)`, before vision encoding. An opt-in Qwen path transforms four images at a time, transfers completed features to CPU, and concatenates on CPU. Pixel values, grids, dtype, shape, stride and image order matched the original path **bit for bit** for 24-page, 47-page and mixed-size fixtures.

| Fixture | Original PyTorch GPU peak | Chunked peak | Original time | Chunked time |
|---|---:|---:|---:|---:|
| 24 pages | 3.28 GiB | 0.67 GiB | 0.451 s | 0.630 s |
| 47 pages | 6.43 GiB | 0.80 GiB | 0.790 s | 1.184 s |
| 47 mixed pages | 5.00 GiB | 0.63 GiB | 0.631 s | 0.985 s |

The 47-page peak falls **87.5%**, at about **0.394 s additional preprocessing time**. This avoids the measured OOM; it is not an inference speedup. It is opt-in (`SGLANG_QWEN4_IMAGE_PREPROCESS_CHUNK_SIZE=4`) and rejects GPU-resident feature transport modes. [Exact measurements and output hashes](concurrency-20261102/image-preprocessing.json).

### Bounded host registration for PLE tables

Large single pinned allocations failed on fragmented Linux hosts. The opt-in allocator uses one prefaulted private contiguous mapping and registers it in 256 MiB CUDA chunks (`SGLANG_QWEN4_PLE_HOST_REGISTER_CHUNK_MB=256`), preserving the host address used by gathers. Storage ownership keeps the registration alive through parameter/view lifetimes. Partial registration failure rolls back; failed unregistration retains the mapping rather than reusing pages still owned by the driver.

Qualification covered both approximately 25.6 GB tables per rank, concurrent eight-rank allocation, boundary-row gathers, CUDA Graph replay, view lifetime and cleanup, plus injected partial-registration rollback. A subsequent four-host qualification covered all 32 ranks. This is a startup/host-memory reliability change, with **no independent serving throughput claim**. The default pinned allocator remains selected when the environment setting is zero.

## 5. Experiments that did not justify changing the final configuration

- Three-step MTP was selected. Five steps exceeded the current QSA verification-width limit.
- 16K prefill chunks with MTP changed short/long throughput by -1.98%/+1.09% relative to 8K chunks; the equally weighted geometric mean was -0.45%, so 8K remained selected.
- Expanded MoE tuning points changed the combined score by only +0.06%, without a consistent gain; the original table remained selected.
- Power-of-two and cache-aware routing did not beat round robin on the tested fixed-length workload.
- Attention-DP alternatives did not beat the selected layout on both shapes. One DP8 candidate failed graph capture with OOM; a later low-memory attempt lost streams, with unresolved cause. Mixed prefill/decode encountered a device assertion. These failed/partial runs were excluded from successful throughput comparisons.
- Scheduler receive intervals, decode steps, lazy cache settings, graph/batch limits and FP32 verification were screened; no further setting was adopted without a supported improvement.

The code PR contains the serving fixes and focused regression tests. It does not change global MTP, parallelism, concurrency, or transport defaults based on this one hardware/workload combination. The reported synthetic performance and basic response checks are not a general model-quality evaluation.
