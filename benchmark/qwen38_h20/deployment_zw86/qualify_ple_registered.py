"""Exercise contiguous registered PLE storage on the production gather kernel."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from unittest import mock

import torch
import triton
import cuda.bindings.runtime as cuda_rt
import sglang

from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel
from sglang.srt.models.qwen4_exp_ple_table import allocate_ple_host_table

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, required=True)
parser.add_argument("--receipt", required=True)
parser.add_argument("--full", action="store_true")
parser.add_argument("--barrier-dir")
parser.add_argument("--peers", type=int, default=1)
args = parser.parse_args()
torch.cuda.set_device(args.device)
torch.set_num_threads(8)
os.environ["SGLANG_QWEN4_PLE_HOST_REGISTER_CHUNK_MB"] = "256" if args.full else "1"
started = time.time()
source = Path(sglang.__file__).parent
source_hashes = {
    name: hashlib.sha256((source / name).read_bytes()).hexdigest()
    for name in [
        "srt/environ.py",
        "srt/models/qwen4_exp.py",
        "srt/models/qwen4_exp_ple_table.py",
    ]
}
results = []
keep = []
rollback_exact = False
shapes = [(80000384, 160), (80001280, 160)] if args.full else [(8193, 160)]

try:
    if not args.full:
        # Register the first real range, then inject failure in the second.
        # The allocator must unregister that first range before propagating it.
        original_register = cuda_rt.cudaHostRegister
        original_unregister = cuda_rt.cudaHostUnregister
        register_calls = []

        def fail_second(ptr, size, flags):
            register_calls.append(ptr)
            if len(register_calls) == 2:
                return (cuda_rt.cudaError_t.cudaErrorInvalidValue,)
            return original_register(ptr, size, flags)

        with mock.patch.object(
            cuda_rt, "cudaHostRegister", side_effect=fail_second
        ), mock.patch.object(
            cuda_rt, "cudaHostUnregister", wraps=original_unregister
        ) as unregister:
            try:
                allocate_ple_host_table((8193, 160), torch.bfloat16, "pinned")
            except RuntimeError as exc:
                assert "PLE cudaHostRegister failed" in str(exc), repr(exc)
            else:
                raise AssertionError("Injected registration error was swallowed")
            assert len(register_calls) == 2
            unregister.assert_called_once_with(register_calls[0])
        gc.collect()
        rollback_exact = True
    for shape in shapes:
        begin = time.monotonic()
        table = allocate_ple_host_table(shape, torch.bfloat16, "pinned")
        ranges = table._sglang_ple_registered_ranges
        # Include rows straddling registration boundaries, table end, and TP masks.
        row_ids = {0, shape[0] - 1, shape[0] // 2}
        for ptr, _ in ranges[1:]:
            crossing = (ptr - table.data_ptr()) // (shape[1] * 2)
            row_ids.update([crossing - 1, crossing, crossing + 1])
        row_ids = sorted(x for x in row_ids if 0 <= x < shape[0])
        row_tensor = torch.tensor(row_ids, dtype=torch.int64)
        values = ((row_tensor[:, None] % 127) + torch.arange(shape[1])[None, :]).to(
            torch.bfloat16
        )
        table[row_tensor] = values
        storage = table.untyped_storage()
        ptr = table.data_ptr()
        parameter = torch.nn.Parameter(table, requires_grad=False)
        view = parameter.detach().view(-1, shape[1])
        del table, parameter, storage
        gc.collect()
        vocab_start = 1000
        ids = torch.tensor(
            [
                vocab_start - 1,
                *[x + vocab_start for x in row_ids],
                vocab_start + shape[0],
            ],
            device="cuda",
        )
        out = torch.empty(ids.numel(), shape[1], dtype=torch.bfloat16, device="cuda")

        def gather():
            _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                ptr,
                ids,
                out,
                embedding_dim=shape[1],
                tp_vocab_start=vocab_start,
                tp_vocab_end=vocab_start + shape[0],
                is_fp8=False,
                BLOCK_D=triton.next_power_of_2(shape[1]),
            )

        gather()
        torch.cuda.synchronize()
        expected = torch.zeros_like(out)
        expected[1:-1] = values.to("cuda")
        assert torch.equal(out, expected), "eager gather differs"
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gather()
        for _ in range(3):
            out.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, expected), "captured gather differs"
        del graph
        results.append(
            dict(
                shape=shape,
                bytes=shape[0] * shape[1] * 2,
                registrations=len(ranges),
                seconds=time.monotonic() - begin,
                sampled_rows=len(row_ids),
                eager_exact=True,
                graph_exact=True,
                parameter_view_lifetime=True,
            )
        )
        keep.append(view)

    if args.barrier_dir:
        barrier = Path(args.barrier_dir)
        barrier.mkdir(exist_ok=True)
        (barrier / f"rank-{args.device}.ready").write_text(str(time.time()))
        deadline = time.monotonic() + 180
        while len(list(barrier.glob("rank-*.ready"))) < args.peers:
            assert (
                time.monotonic() < deadline
            ), "Concurrent table allocation barrier timed out"
            time.sleep(0.5)
        # Keep every rank's two real tables registered together for this window.
        time.sleep(10)

    # Storage, not the original tensor, owns unregister. Observe exact bases.
    original = cuda_rt.cudaHostUnregister
    with mock.patch.object(cuda_rt, "cudaHostUnregister", wraps=original) as unregister:
        del view
        expected_unregisters = sum(r["registrations"] for r in results)
        keep.clear()
        gc.collect()
        assert unregister.call_count == expected_unregisters, (
            unregister.call_count,
            expected_unregisters,
        )
    from sglang.srt.models.qwen4_exp_ple_table import _PLE_RETAINED_MAPPINGS

    assert not _PLE_RETAINED_MAPPINGS, "registration cleanup retained a failed range"
    receipt = dict(
        passed=True,
        device=args.device,
        full=args.full,
        results=results,
        cleanup_exact=True,
        rollback_exact=rollback_exact,
        concurrent_peers=args.peers,
        source_sha256=source_hashes,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cuda_bindings=importlib.metadata.version("cuda-bindings"),
        started_unix=started,
        completed_unix=time.time(),
    )
except BaseException as exc:
    receipt = dict(
        passed=False,
        device=args.device,
        full=args.full,
        results=results,
        error=repr(exc),
        started_unix=started,
        completed_unix=time.time(),
    )
    raise
finally:
    Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
