"""OuterProductMean with a contraction order suited to a single-sequence MSA.

Upstream contracts the MSA-sequence axis first:

    act = einsum('acb,ade->dceb', left_act, right_act)     # [N_res, c, c, N_res]
    act = einsum('dceb,cef->dbf', act, output_w)

The comment in modules.py calls this "faster", and it is when the MSA is deep: contracting
over N_seq early shrinks the work that follows. This experiment runs with
max_msa_clusters = 1, where it is the wrong way round -- contracting a length-1 axis gains
nothing and the intermediate [N_res, c, c, N_res] is materialised anyway. At N_res=124,
c=32 that intermediate is 63 MB, and the second einsum then costs about 4 GFLOP.

Folding output_w into right_act first gives the same result:

    tmp = einsum('ade,cef->adcf', right_act, output_w)      # [N_seq, N_res, c, c_z], 2 MB
    act = einsum('abc,adcf->bdf', left_act, tmp)

Measured in isolation at these shapes: 1.544 ms -> 0.197 ms, a 7.85x reduction, agreeing
with the original to 5.9e-07 relative L2 (float32 reassociation).

This is an experiment-local monkeypatch, not a fix to the vendored AlphaFold source, and it
is gated on the MSA actually being shallow so the upstream order is kept whenever it is the
right one. The CPU/MPS gradient gate must be re-run when it is enabled: the arithmetic
changes, even though the mathematics does not.
"""
import haiku as hk
import jax.numpy as jnp

from mosaic.alphafold.model import modules, common_modules, mapping

MAX_SEQUENCES_FOR_REORDER = 8

_Original = modules.OuterProductMean


class _FastOuterProductMean(modules.OuterProductMean):
    """Subclass rather than a method assignment: haiku wraps methods at class creation to
    establish the module name scope, so replacing __call__ afterwards would place the
    submodule parameters under the wrong path."""

    def __call__(self, act, mask, is_training=True):
        if act.shape[0] > MAX_SEQUENCES_FOR_REORDER:
            return _Original.__call__(self, act, mask, is_training)

        gc = self.global_config
        c = self.config

        mask = mask[..., None]
        act = common_modules.LayerNorm([-1], True, True, name='layer_norm_input')(act)

        left_act = mask * common_modules.Linear(
            c.num_outer_channel, initializer='linear', name='left_projection')(act)
        right_act = mask * common_modules.Linear(
            c.num_outer_channel, initializer='linear', name='right_projection')(act)

        if gc.zero_init:
            init_w = hk.initializers.Constant(0.0)
        else:
            init_w = hk.initializers.VarianceScaling(scale=2., mode='fan_in')

        output_w = hk.get_parameter(
            'output_w',
            shape=(c.num_outer_channel, c.num_outer_channel, self.num_output_channel),
            dtype=act.dtype, init=init_w)
        output_b = hk.get_parameter(
            'output_b', shape=(self.num_output_channel,),
            dtype=act.dtype, init=hk.initializers.Constant(0.0))

        # Independent of the subbatched axis, so computed once rather than per chunk.
        folded = jnp.einsum('ade,cef->adcf', right_act, output_w)

        def compute_chunk(left_act):
            return jnp.einsum('abc,adcf->bdf', left_act, folded) + output_b

        act = mapping.inference_subbatch(
            compute_chunk, c.chunk_size, batched_args=[left_act], nonbatched_args=[],
            low_memory=True, input_subbatch_dim=1, output_subbatch_dim=0)

        epsilon = 1e-3
        norm = jnp.einsum('abc,adc->bdc', mask, mask)
        act /= epsilon + norm
        return act


def enable():
    """Install the reordered contraction. Returns True if it was newly installed."""
    if modules.OuterProductMean is _FastOuterProductMean:
        return False
    modules.OuterProductMean = _FastOuterProductMean
    return True


def disable():
    modules.OuterProductMean = _Original
