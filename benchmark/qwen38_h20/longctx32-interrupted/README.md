# Completed 32-H20 long-context controls; interrupted MTP phase

Baseline and tuned TP4 measurements completed on the same clean four-node,
32-H20 cohort. The subsequent MTP phase was interrupted by the user-requested
migration to BJ80. Its seven completed matrix points are preserved separately;
there is no completed MTP primary repetition or 32K high-concurrency result in
this dataset. BJ80 uses a fresh 16-H20 cohort, whose results must remain separate.

Primary output token/s values are medians of three C256 repetitions, with 512
requests per repetition:

| Input/output | TP8 baseline | TP4 tuned | Change |
|---|---:|---:|---:|
| 16384/1024 | 3745.6 | 5410.0 | +44.4% |
| 32768/1024 | 2100.7 | 2933.1 | +39.6% |

The latency statistics below are medians of each repetition's statistic, not
pooled-request percentiles:

| Input | Configuration | Mean TTFT (s) | Mean TPOT (ms) | p99 TTFT (s) |
|---|---|---:|---:|---:|
| 16K | baseline | 20.24 | 48.18 | 47.50 |
| 16K | tp4 | 15.32 | 32.02 | 29.69 |
| 32K | baseline | 50.60 | 63.06 | 118.55 |
| 32K | tp4 | 42.38 | 31.35 | 76.00 |

The TP4 gain includes the complete configuration package: four TP8/EP1 replicas
with Triton linear attention and memory fraction 0.85 changed to eight TP4/EP4
replicas, FlashInfer linear-attention prefill, the existing tuned Triton MoE table
and memory fraction 0.82. Both retain BF16 weights/recurrent state and 8192-token
prefill chunks. It is not an isolated parallel-layout or MoE kernel gain.

Each complete configuration passed 16 measurements and 4944 timed requests:
exact 16384/1024 and 32768/1024 token lengths, concurrency 1/16/64/128/256, then
three extra C256 repetitions per shape. Requests use random token IDs, seed
20261031, temperature zero and ignore-EOS. Every worker was warmed per shape;
every prefix cache was flushed before each timed run. The timer excludes the
post-run server-info query. The CSV files preserve individual requests, all
latency metrics, and raw-file SHA256 values.

Throughput is the selection priority. TP4 regresses at C1. For 32K input, the
TP4 matrix changes from 2912.4 to 2925.7 output token/s at C128 to C256, while
mean TTFT rises from 15.43s to 42.47s. C128 is therefore a useful operating point
when that latency difference matters. The three-repeat primary table uses
separate measurements from these single-run matrix points.

[Checked comparison and complete curves](comparison.json), [completed control
exports](results/), and [interrupted MTP measurements](interrupted-mtp.json)
retain the evidence. [Validation](validation.json) records the response and
runtime checks. Basic chat and near-16K/32K retrieval passed at 10%, 50%
and 90% needle positions. They are basic service checks, not application-quality
validation. These fixed-length synthetic requests do not establish performance
for multimodal, shared-prefix, mixed-length or longer-output workloads.

The pinned main base is `a4a6eddd25343905245db7a8b54bcf24d9483c8b`, with the
existing empty-RMSNorm guard `fc2ad6c` and regression coverage `4b1c9d3`. The
model revision is `de4b8e4d43b917e7706784d8bb445c9af86a3540`. No separate speedup
is attributed to the guard. The 32-H20 allocation was released through the
native cleanup operation when migration was requested.
