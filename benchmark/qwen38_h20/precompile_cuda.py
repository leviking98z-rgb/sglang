import importlib.metadata
import json
import time
import traceback

from runtime import PROJECT

started = time.time()
result = {"started_unix": started}
try:
    from sglang.srt.environ import redirect_third_party_caches

    redirect_third_party_caches()
    import torch

    result["versions"] = {
        name: importlib.metadata.version(name)
        for name in [
            "torch",
            "nvidia-cuda-nvcc",
            "nvidia-cuda-crt",
            "nvidia-nvvm",
            "nvidia-cuda-runtime",
            "nvidia-cuda-nvrtc",
            "flashinfer-python",
        ]
    }
    print(json.dumps(result["versions"]), flush=True)
    assert torch.version.cuda == "13.0"
    for name in ["nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm"]:
        assert result["versions"][name] == "13.0.88"
    x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
    assert (x @ x).float().sum().item() == 2097152
    from flashinfer.jit.comm import gen_trtllm_comm_module

    module = gen_trtllm_comm_module().build_and_load()
    result.update(status="complete", module=str(module))
    print("FlashInfer TRTLLM communication JIT build and load passed", flush=True)
except BaseException as exc:
    result.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
    traceback.print_exc()
    raise
finally:
    result["seconds"] = time.time() - started
    (PROJECT / "logs/precompile-cuda130-01-result.json").write_text(
        json.dumps(result, indent=2)
    )
