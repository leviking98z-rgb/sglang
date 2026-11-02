import os
from pathlib import Path

PROJECT = Path(os.environ["QWEN38_PROJECT"])
RUNTIME = Path("/root/.cache") / ("qwen38-runtime-" + PROJECT.name)
PYTHON = str(RUNTIME / "venv/bin/python")
MODEL = str(RUNTIME / "model/Qwen3.8-Flash-Next")
MODEL_NAME = "Qwen/Qwen3.8-Flash-Next"


def environment():
    prepare_cuda_linker_layout()
    env = os.environ.copy()
    cuda_root = RUNTIME / "venv/lib/python3.12/site-packages/nvidia/cu13"
    env["CUDA_HOME"] = str(cuda_root)
    env["LD_LIBRARY_PATH"] = (
        "/opt/cudacompat13/usr/local/cuda-13.3/compat:/opt/conda/lib:"
        + str(cuda_root / "lib")
        + ":"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PATH"] = (
        str(RUNTIME / "venv/bin")
        + ":"
        + str(cuda_root / "bin")
        + ":"
        + env.get("PATH", "")
    )
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["PYTHONPATH"] = str(RUNTIME / "sglang/python")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "8"
    env["PYTHONUNBUFFERED"] = "1"
    env["SGLANG_QWEN4_PLE_HOST_REGISTER_CHUNK_MB"] = "256"
    cache_root = Path("/root/.cache") / ("sglang-" + PROJECT.name + "-cu130")
    env["SGLANG_CACHE_DIR"] = str(cache_root)
    env["SGLANG_JIT_CACHE_DIR"] = str(cache_root / "jit")
    env["PYTHONPYCACHEPREFIX"] = str(cache_root / "pycache")
    return env


def prepare_cuda_linker_layout():
    """Expose the conventional toolkit layout for pip's CUDA 13 libraries."""
    cuda_root = RUNTIME / "venv/lib/python3.12/site-packages/nvidia/cu13"
    assert (cuda_root / "lib/libcudart.so.13").is_file()
    if not (cuda_root / "lib64").exists():
        (cuda_root / "lib64").symlink_to("lib", target_is_directory=True)
    if not (cuda_root / "lib/libcudart.so").exists():
        (cuda_root / "lib/libcudart.so").symlink_to("libcudart.so.13")
