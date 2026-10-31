"""Compare main's existing HC combine kernels on an otherwise idle owned GPU."""

import argparse
import json
import statistics

import torch
from sglang.kernels.ops.elementwise.hc_combine import hc_combine, hc_combine_split


def measure(function):
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            function()
    values = []
    for _ in range(5):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):
            graph.replay()
        stop.record()
        stop.synchronize()
        values.append(start.elapsed_time(stop) * 1000 / 2000)
    return {"median_us": statistics.median(values), "samples_us": values}


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
torch.manual_seed(20261031)
torch.set_float32_matmul_precision("highest")
results = []
for batch in (1, 8, 16, 24, 32, 40, 64, 128, 512):
    residual = torch.randn(batch, 10240, device="cuda", dtype=torch.bfloat16)
    normed = torch.randn_like(residual)
    block = torch.randn(batch, 2560, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16) / 101.192885
    output = torch.empty_like(residual)
    arguments = (block, residual, normed, weight, 4, 2560)
    gate = 2 * torch.sigmoid(normed.float() @ weight.float().T / 4)
    expected = (residual.float().reshape(batch, 4, 2560)
                + gate[:, :, None] * block.float()[:, None, :]).reshape(batch, 10240)
    expected = expected.to(torch.bfloat16)
    record = {"batch": batch, "default": "split" if batch <= 32 else "unsplit"}
    for name, kernel in (("unsplit", hc_combine), ("split", hc_combine_split)):
        actual = kernel(*arguments, out=output)
        torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)
        record[name] = measure(
            lambda kernel=kernel, arguments=arguments, output=output: kernel(*arguments, out=output)
        )
        record[name]["max_abs_error"] = (actual.float() - expected.float()).abs().max().item()
    results.append(record)
    print(json.dumps(record), flush=True)
with open(args.output, "x") as stream:
    json.dump({"device": torch.cuda.get_device_name(), "results": results,
               "scope": "Graph-replay microbenchmark; serving validation required."}, stream, indent=2)
    stream.write("\n")
