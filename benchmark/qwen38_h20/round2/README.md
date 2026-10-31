# Further optimization: measurements in progress

The first-round deployment and complete 32-GPU comparison remain documented in
the parent directory. This directory records a second round of experiments;
initial screening results are not a new final deployment claim.

Comparisons use equal eight-H20 budgets, concurrency 64, exact 1024/256 and
8192/1024 token lengths, and the same pinned engine, model and request timer as
the first round. Four same-node controls each have three repetitions per shape,
with 128 requests per repetition. Router-policy tests also have three
repetitions; initial service candidates have two, including their first run
after restart. Every successful timed request has checked token counts.

Fresh service processes show substantial variation in their first short run.
All samples are retained in the CSV files and
[screening-progress.json](screening-progress.json). Candidate selection requires
independent longer repetitions and a subsequent complete 32-GPU comparison.
The response checks cover basic chat and 8K retrieval, not application quality.

Independent longer confirmation on one unchanged eight-GPU cohort is complete
for MTP3. Both configurations use the empty-launch guard, TP4/EP4, two replicas,
8K chunks and the original MoE table. At concurrency 64, three repetitions of
512 requests per shape gave these output-token/s medians:

| Configuration | 1024/256 | 8192/1024 |
|---|---:|---:|
| Original control | 2496.3 | 1967.2 |
| NEXTN3 / four verification tokens | 3078.7 | 2383.6 |
| Change | +23.3% | +21.2% |

All six longer measurements and both initial 128-request measurements are
retained for each configuration in [confirmation](confirmation/comparison.json).
These are eight-GPU results; the full 32-GPU follow-up is still pending.

Results already established on the unchanged node-0 control layout:

| Router policy | Short output token/s | Long output token/s |
|---|---:|---:|
| Round robin control | 2607.31 | 1959.42 |
| Power of two | 2458.18 | 1828.07 |
| Cache aware | 2601.28 | 1946.63 |

Values are medians of three repetitions. Round robin remains selected for this
fixed-length throughput workload. The separate
[HC microbenchmark](hc-kernels.json) passed its numerical checks and supports
retaining main's existing split-kernel dispatch through batch 32; changing the
cutoff did not establish a useful improvement. Kernel timings are not serving
throughput results.

The [EP8 tuning result](moe-tp8ep8/provenance.json) uses main's official MoE tuner,
74 kernel configurations and eight batch sizes. It prepares the attention-DP
experiments; it does not by itself establish an end-to-end gain. Initial controls and candidates used the unmodified main source. During the
attention-DP experiment, a zero-token grouped Gemma RMSNorm launch failed.
The fork now includes a host-side empty-launch guard in `fc2ad6c`, plus empty,
nonempty and CUDA-graph regression coverage in `4b1c9d3`. Confirmation and final
results must record this patch explicitly; no upstream PR was opened.

Other completed screens cover 4K/16K prefill chunks, decode scheduling steps,
scheduler receive intervals, lazy Mamba cache checkpointing, prefill scheduling intervals,
maximum running batch 32, and FP32 FlashInfer MTP verification. None is selected
from the initial two-run screen. A 16K chunk and patched attention DP4 remain
confirmation candidates. Attention DP8 at memory fraction 0.82 ran out of memory
while capturing a prefill graph and is being retried with more temporary-buffer
headroom. Mixed prefill/decode passed single-request smoke checks but failed its
first concurrent run with a CUDA device assertion; its partial output is excluded
from the successful-results CSV files. Details are in
[failed-candidates.json](failed-candidates.json).

The existing NUMA binding was verified against the CPU affinity of both TP4
replicas and the GPU topology. No affinity change or NUMA performance gain is
claimed. The MTP confirmation retains BF16 recurrent state, three NEXTN steps,
four verification tokens and Triton verification. Five steps exceed the current
QSA verification-width limit; that startup failure is recorded separately.
