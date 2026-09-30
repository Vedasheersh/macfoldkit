"""Save both complete contact-loss gradients for cross-platform comparison.

On CPU also require the adapter's primal and VJP to match upstream Mosaic.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import mosaic.losses.structure_prediction as sp
from metal_losses import WithinBinderContact, BinderTargetContact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path, help='CPU NPZ reference; require corrected VJPs to agree')
    args = parser.parse_args()
    logits = jnp.asarray(np.random.default_rng(9).normal(size=(60, 60, 64)).astype('float32'))
    sequence = jnp.full((40, 20), 0.05)
    bins = jnp.linspace(2, 22, 64)
    rows, arrays = [], {}
    for name, cls, upstream, kwargs in [
        ('intra', WithinBinderContact, sp.WithinBinderContact, dict(num_contacts_per_residue=8)),
        ('inter', BinderTargetContact, sp.BinderTargetContact, dict(contact_distance=8., paratope_size=12)),
        ('selected', BinderTargetContact, sp.BinderTargetContact,
         dict(contact_distance=8., paratope_idx=jnp.array([0, 1, 3, 4]), epitope_idx=jnp.array([0, 3, 5, 6])))]:
        values = {}
        for method, term in [('upstream', upstream(**kwargs)), ('corrected', cls(**kwargs))]:
            fn = lambda z: term(sequence, SimpleNamespace(distogram_logits=z, distogram_bins=bins), jax.random.key(7))[0]
            value, gradient = jax.block_until_ready(jax.jit(jax.value_and_grad(fn))(logits))
            arrays[f'{name}_{method}'] = np.asarray(gradient)
            values[method] = float(value)
        np.testing.assert_allclose(values['corrected'], values['upstream'], rtol=0, atol=0)
        if jax.default_backend() == 'cpu':
            np.testing.assert_allclose(arrays[f'{name}_corrected'], arrays[f'{name}_upstream'], rtol=1e-6, atol=1e-9)
        rows.append(dict(term=name, **values))
    np.savez(args.output, **arrays)
    comparisons = []
    if args.reference:
        with np.load(args.reference) as reference:
            for name, gradient in arrays.items():
                expected = reference[name.replace('_upstream', '_corrected')]
                relative = float(np.linalg.norm(gradient-expected) / np.linalg.norm(expected))
                comparisons.append(dict(loss=name, relative_l2=relative))
                if name.endswith('_corrected') and not relative <= 1e-5:
                    raise AssertionError(f'{name} CPU/MPS mismatch: {relative}')
    print(json.dumps(dict(platform=jax.default_backend(), losses=rows,
                         comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
