"""Bounded Mac batching policy; original model math is unchanged.

Only column attention and transitions are widened. Changing batches may select
different GPU kernels and rounding; bitwise equivalence is not guaranteed.
Budgets refer to a single
score/hidden tensor, not total process memory. Inputs beyond measured lengths
retain the original policy. The actual peak footprint is measured separately.
"""
from contextvars import ContextVar
from functools import wraps

import jax.numpy as jnp
from alphafold.model import fused_ops, mapping, modules


def install():
    if getattr(modules, "_codex_mac_batching_installed", False):
        return
    modules._codex_mac_batching_installed = True
    active = ContextVar("wider_column_attention", default=False)
    original_subbatch = mapping.inference_subbatch
    original_transition = fused_ops.transition_subbatch
    seen = set()

    def report(kind, shape, old, new):
        key = (kind, tuple(shape), old, new)
        if key not in seen:
            seen.add(key)
            print("EVOFORMER_BATCH_TRACE", kind, tuple(shape), old, "->", new, flush=True)

    def within_scope(shape, dtype):
        return (jnp.dtype(dtype) == jnp.dtype(jnp.bfloat16)
                and len(shape) == 3 and min(shape[:2]) >= 128
                and max(shape[:2]) <= 427)

    def shard_for(rows, bytes_per_row, budget):
        limit = max(1, budget // max(1, bytes_per_row))
        return None if rows <= limit else (limit if limit < 4 else (limit // 4) * 4)

    def subbatch(module, subbatch_size, batched_args, nonbatched_args, low_memory=True,
                 input_subbatch_dim=0, output_subbatch_dim=None):
        q = batched_args[0]
        if (active.get() and low_memory and input_subbatch_dim == 0
                and isinstance(module, (modules.Attention, modules.GlobalAttention))
                and within_scope(q.shape, q.dtype)):
            heads = module.config.num_head
            queries = 1 if isinstance(module, modules.GlobalAttention) else q.shape[1]
            keys = batched_args[1].shape[1]
            per_row = heads * queries * keys * jnp.dtype(q.dtype).itemsize
            selected = shard_for(q.shape[0], per_row, 128 * 1024**2)
            report(type(module).__name__, q.shape, subbatch_size, selected)
            subbatch_size = selected
        return original_subbatch(module, subbatch_size, batched_args, nonbatched_args,
            low_memory=low_memory, input_subbatch_dim=input_subbatch_dim,
            output_subbatch_dim=output_subbatch_dim)

    def transition(gc, shape, hidden, dtype):
        original = original_transition(gc, shape, hidden, dtype)
        if within_scope(shape, dtype):
            per_row = int(shape[1]) * int(hidden) * jnp.dtype(dtype).itemsize
            selected = shard_for(int(shape[0]), per_row, 256 * 1024**2)
            report("Transition", shape, original, selected)
            return selected
        return original

    def wrap_column(original):
        @wraps(original)
        def call(self, *args, **kwargs):
            token = active.set(True)
            try:
                return original(self, *args, **kwargs)
            finally:
                active.reset(token)
        return call

    modules.MSAColumnAttention.__call__ = wrap_column(modules.MSAColumnAttention.__call__)
    modules.MSAColumnGlobalAttention.__call__ = wrap_column(modules.MSAColumnGlobalAttention.__call__)
    mapping.inference_subbatch = subbatch
    fused_ops.transition_subbatch = transition
