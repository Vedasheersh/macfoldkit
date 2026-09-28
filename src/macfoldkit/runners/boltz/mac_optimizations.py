"""Opt-in inference optimizations for the pinned FastPLMs Boltz2 runtime."""
from __future__ import annotations

import torch


def optimize_model(model, precision: str = "bfloat16", chunk_size: int = 32) -> dict:
    """Compile pair updates with bounded attention chunks and FP32 residuals.

    This installs the shared pair function used by trunk, MSA and confidence
    blocks. Use one precision policy per process. Parameters and residual storage
    remain FP32. BF16 autocast accelerates eligible pair operations; accumulation
    is chosen by the backend. Diffusion is unchanged.
    """
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("Use float32 or bfloat16; float16 overflowed in validation.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if model.training or next(model.parameters()).device.type != "mps":
        raise ValueError("Expected an eval-mode model on MPS")
    globals_ = model.core.pairformer_module.layers[0].forward.__globals__
    original = globals_.setdefault("_mac_original_pair_update", globals_["_pair_update"])

    def pair_update(modules, pair_states, pair_mask, chunk_size_unused,
                    use_kernels, use_cuequiv_mul, use_cuequiv_attn):
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=precision == "bfloat16"):
            return original(modules, pair_states, pair_mask, chunk_size,
                            use_kernels, use_cuequiv_mul, use_cuequiv_attn)

    globals_["_pair_update"] = torch.compile(pair_update, backend="inductor")
    return {"compiled_pair_update": True, "pair_precision": precision,
            "pair_chunk": chunk_size, "torch_version": torch.__version__}
