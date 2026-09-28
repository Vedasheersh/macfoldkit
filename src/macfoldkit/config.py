"""Portable runtime locations. No model framework is imported by the frontend."""
import os
from pathlib import Path
import platform
import sys

PACKAGE = Path(__file__).resolve().parent
BACKENDS = ("boltz", "colabfold", "mosaic")


def cache_home(value=None):
    if value is not None:
        return Path(value).expanduser().resolve()
    configured = os.environ.get("MACFOLDKIT_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path.home() / ("Library/Caches" if sys.platform == "darwin" else ".cache")
    return base / "macfoldkit"


def runtime_python(home, backend):
    return home / "runtimes" / backend / "bin" / "python"


def supported_platform():
    return sys.platform == "darwin" and platform.machine().lower() == "arm64"


def runtime_env(home, backend):
    env = dict(os.environ)
    env.update(MACFOLDKIT_HOME=str(home), PYTHONUNBUFFERED="1",
               HF_HOME=str(home / "cache/huggingface"),
               MPLCONFIGDIR=str(home / "cache/matplotlib"),
               XDG_CACHE_HOME=str(home / "cache"),
               MLX_ENABLE_TF32="0", JAX_MPS_ASYNC_DISPATCH="0")
    # Do not inherit an unrelated experiment's plugin, module path or GPU arch.
    for key in ("JAX_MPS_LIBRARY_PATH", "MLX_METAL_GPU_ARCH", "PYTHONPATH", "PYTHONOPTIMIZE"):
        env.pop(key, None)
    source = {"boltz": "fastplms", "boltz-preprocess": "boltz", "mosaic": "mosaic"}.get(backend)
    if source:
        env["PYTHONPATH"] = str(home / "sources" / source / "src")
    if backend == "boltz":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["ESMCFOLD_CCD_PATH"] = str(home / "cache/boltz/fastplms-ccd.pkl")
    else:
        env["JAX_PLATFORMS"] = "mps,cpu"
    return env
