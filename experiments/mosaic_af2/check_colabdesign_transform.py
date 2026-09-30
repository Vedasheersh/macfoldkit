"""Pre-run checks for the ColabDesign/BindCraft schedule; run on CPU and MPS separately.

The four-stage schedule changes the model input path, so this exercises the transform and
its reverse derivative alone -- no model weights, no AF2 -- before any long GPU run.

Two things are measured:

1. Parity. The same (soft, temp, hard) settings the schedule uses, at two logit scales.
   Compare the two platforms' JSON: gradient magnitudes and zero-gradient row counts
   should agree, which establishes that the straight-through path (argmax, one_hot,
   stop_gradient, low-temperature softmax and its VJP) adds no backend divergence.

2. Saturation. At temp 0.01 the softmax saturates and the gradient vanishes on a
   logit-scale-dependent fraction of residues. This is a property of the method, not of a
   backend, so it reproduces on CPU. It bounds the usable initialization scale: the
   schedule's final two stages become no-ops if the logits are too large.
"""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.optimizers import (_colabdesign_transform, _colabdesign_pullback,
                               _seq_grad_norm)

# Start and end settings of each stage in mosaic.optimizers.bindcraft_design.
STAGES = [('logits1_start', 0.0, 1.0, 0.0), ('logits1_end', 0.9, 1.0, 0.0),
          ('logits2_end', 1.0, 1.0, 0.0), ('soft_mid', 1.0, 0.505, 0.0),
          ('soft_end', 1.0, 1e-2, 0.0), ('hard', 1.0, 1e-2, 1.0)]
LENGTH, TOKENS = 48, 20


def _pullback(z, cotangent, soft, temp, hard):
    return _colabdesign_pullback(z, cotangent, jnp.asarray(soft, jnp.float32),
                                 jnp.asarray(temp, jnp.float32), jnp.asarray(hard, jnp.float32))


def main():
    rng = np.random.default_rng(7)
    cotangent = rng.normal(size=(LENGTH, TOKENS)).astype('float32')
    # Row-centered, matching what the patched evaluator hands back to the optimizer.
    cotangent = jnp.asarray(cotangent - cotangent.mean(-1, keepdims=True))
    base = rng.normal(size=(LENGTH, TOKENS)).astype('float32')

    parity = []
    for name, array in [('bindcraft_init_0.01', 0.01 * base), ('evolved_logits_x30', 30.0 * base)]:
        z = jnp.asarray(array.astype('float32'))
        for stage, soft, temp, hard in STAGES:
            pseudo = jax.block_until_ready(jax.jit(_colabdesign_transform)(
                z, jnp.asarray(soft, jnp.float32), jnp.asarray(temp, jnp.float32),
                jnp.asarray(hard, jnp.float32)))
            gradient = jax.block_until_ready(_pullback(z, cotangent, soft, temp, hard))
            normalized = jax.block_until_ready(jax.jit(_seq_grad_norm)(gradient))
            values = np.asarray(pseudo, dtype='float64')
            g = np.asarray(gradient, dtype='float64')
            if not np.isfinite(values).all() or not np.isfinite(g).all():
                raise ValueError(f'Nonfinite transform or gradient at {name}/{stage}')
            if not np.isfinite(np.asarray(normalized)).all():
                raise ValueError(f'Nonfinite normalized gradient at {name}/{stage}')
            parity.append(dict(init=name, stage=stage, soft=soft, temp=temp, hard=hard,
                on_simplex=bool(np.allclose(values.sum(-1), 1.0, atol=1e-4) and values.min() >= -1e-6),
                pseudo_row_sum_mean=float(values.sum(-1).mean()),
                gradient_absolute_max=float(np.abs(g).max()),
                zero_gradient_rows=int((np.abs(g).sum(-1) == 0).sum())))

    # The v1 initialization is included by name: it is the scale a v2 would inherit by
    # accident, and the README cites its saturation count.
    scales = [('v1_gumbel_x0.5', (rng.gumbel(size=(LENGTH, TOKENS)) * 0.5).astype('float32'))]
    scales += [(f'normal_x{scale}', base * scale) for scale in
               [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]]
    saturation = []
    for label, array in scales:
        z = jnp.asarray(array.astype('float32'))
        row = dict(init=label, logit_std=float(np.std(np.asarray(z))))
        for temp in [1.0, 0.1, 1e-2]:
            g = np.asarray(jax.block_until_ready(_pullback(z, cotangent, 1.0, temp, 0.0)),
                           dtype='float64')
            row[f'zero_gradient_rows_at_temp_{temp}'] = int((np.abs(g).sum(-1) == 0).sum())
        saturation.append(row)

    report = dict(platform=jax.default_backend(), length=LENGTH,
                  stages_from='mosaic.optimizers.bindcraft_design', parity=parity,
                  saturation=saturation,
                  note=('Saturation is a property of the schedule, not of a backend; it '
                        'reproduces on CPU and bounds the usable initialization scale.'))
    print(json.dumps(report, indent=2))
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'w') as handle:
            json.dump(report, handle, indent=2)


if __name__ == '__main__':
    main()
