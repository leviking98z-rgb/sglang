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
experiments; it does not by itself establish an end-to-end gain. Engine source
remains unmodified.
