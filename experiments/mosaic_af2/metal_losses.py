"""Mosaic contact losses with a scatter-free sorted-mean reverse derivative.

Loss equations adapted from Mosaic (MIT; see MOSAIC_LICENSE). The primal and
stable tie selection match upstream; only the VJP avoids negative-stride scatter
updates, which drop gradients with the tested jax-mps 0.11.0 backend.
"""
from functools import partial

import jax
import jax.numpy as jnp
from mosaic.losses.structure_prediction import (
    WithinBinderContact as UpstreamWithinBinderContact,
    BinderTargetContact as UpstreamBinderTargetContact,
    contact_cross_entropy,
)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def sorted_top_k_mean(values, k):
    """Original descending sort/mean, with the same selected subgradient.

    At a boundary tie, JAX's descending value sort reverses a stable ascending
    sort, selecting the largest original indices first. Preserve that convention.
    Like the original slice, k greater than the row length selects the whole row.
    Inputs must be finite; the caller checks the raw objective and gradient.
    """
    if k < 1 or values.shape[-1] < 1:
        raise ValueError('sorted_top_k_mean requires a nonempty row and k >= 1')
    return jnp.sort(values, axis=-1, descending=True)[..., :k].mean(-1)


def _sorted_mean_forward(values, k):
    count = min(k, values.shape[-1])
    indices = jnp.argsort(values, axis=-1, stable=True)[..., -count:]
    selected = jnp.any(jnp.arange(values.shape[-1]) == indices[..., None], axis=-2)
    return sorted_top_k_mean(values, k), selected


def _sorted_mean_backward(k, selected, cotangent):
    count = min(k, selected.shape[-1])
    return (cotangent[..., None] * selected / count,)


sorted_top_k_mean.defvjp(_sorted_mean_forward, _sorted_mean_backward)


class WithinBinderContact(UpstreamWithinBinderContact):
    def __call__(self, sequence, output, key):
        n = sequence.shape[0]
        scores = contact_cross_entropy(output.distogram_logits[:n, :n],
            self.max_contact_distance, bins=output.distogram_bins)
        allowed = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :]) > self.min_sequence_separation
        average = sorted_top_k_mean(scores + (1 - allowed) * -30,
            self.num_contacts_per_residue).mean()
        return -average, {'intra_contact': average}


class BinderTargetContact(UpstreamBinderTargetContact):
    def __call__(self, sequence, output, key):
        n = sequence.shape[0]
        scores = contact_cross_entropy(output.distogram_logits[:n, n:],
            self.contact_distance, bins=output.distogram_bins)
        if self.epitope_idx is not None:
            scores = scores[:, self.epitope_idx]
        per_position = sorted_top_k_mean(scores, 3)
        if self.paratope_idx is not None:
            per_position = per_position[self.paratope_idx]
        if self.paratope_size is not None:
            # Retain upstream's top_k tie convention for paratope selection.
            per_position = jax.lax.top_k(per_position, self.paratope_size)[0]
        average = per_position.mean()
        return -average, {'target_contact': average}
