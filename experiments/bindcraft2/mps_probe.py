"""First MPS execution of BindCraft2 primitives; the feasibility audit was CPU-only.

Runs BindCraft2's own attention and fused triangle multiplicative update, forward and
backward, and reports which attention backend it selects on this machine. Compare the CPU
and MPS JSON files.

BindCraft2 is source-available under a hosting-restricted, non-OSI licence. It is read
from a separate local checkout and is neither vendored nor redistributed here.
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from bindcraft.af import accel


def grad_check(fn, arrays, label):
    """Forward value plus gradients w.r.t. every input, with timing."""
    scalar = lambda *a: jnp.sum(fn(*a) ** 2)
    value, grads = jax.block_until_ready(
        jax.jit(jax.value_and_grad(scalar, argnums=tuple(range(len(arrays)))))(*arrays))
    start = time.perf_counter()
    value, grads = jax.block_until_ready(
        jax.jit(jax.value_and_grad(scalar, argnums=tuple(range(len(arrays)))))(*arrays))
    seconds = time.perf_counter() - start
    return dict(label=label, value=float(value), seconds=seconds,
                grad_norms=[float(np.linalg.norm(np.asarray(g, dtype='float64'))) for g in grads],
                grad_zero_fraction=[float((np.asarray(g) == 0).mean()) for g in grads],
                finite=bool(all(np.isfinite(np.asarray(g)).all() for g in grads)))


def main():
    out = sys.argv[1]
    backend = jax.default_backend()
    report = {'backend': backend, 'jax': jax.__version__,
              'device': str(jax.devices()[0]),
              'selected_attention_backend': accel.supported_attention_backend('auto'),
              'cuequivariance_available': accel.cuequivariance_available(),
              'checks': []}

    rng = np.random.default_rng(7)
    B, Q, H, C = 1, 124, 4, 32
    f32 = lambda *s: jnp.asarray(rng.standard_normal(s).astype('float32'))
    q, k, v = f32(B, Q, H, C), f32(B, Q, H, C), f32(B, Q, H, C)
    bias = f32(B, 1, 1, Q)
    nonbatched = f32(H, Q, Q)

    for name in ('stock', 'chunked'):
        fn = (lambda a, b, c, d, e: accel.attend(a, b, c, d, e, backend='stock')) if name == 'stock' \
            else (lambda a, b, c, d, e: accel.attend(a, b, c, d, e, backend='chunked', chunk_size=32))
        try:
            report['checks'].append(grad_check(fn, (q, k, v, bias, nonbatched), f'attend_{name}'))
        except Exception as error:
            report['checks'].append({'label': f'attend_{name}', 'error': repr(error)[:300]})

    # Triangle multiplicative update: the other AF2 primitive, and the one whose
    # descending-sort cousins broke gradients on this backend in Mosaic.
    try:
        N, Cz = 124, 64
        activations = f32(N, N, Cz)
        mask = jnp.ones((N, N), dtype=jnp.float32)
        params = {'left_norm_input': {'scale': jnp.ones((Cz,)), 'offset': jnp.zeros((Cz,))},
                  'projection': {'weights': f32(Cz, 2 * Cz), 'bias': jnp.zeros((2 * Cz,))},
                  'gate': {'weights': f32(Cz, 2 * Cz), 'bias': jnp.zeros((2 * Cz,))},
                  'center_norm': {'scale': jnp.ones((Cz,)), 'offset': jnp.zeros((Cz,))},
                  'output_projection': {'weights': f32(Cz, Cz), 'bias': jnp.zeros((Cz,))},
                  'gating_linear': {'weights': f32(Cz, Cz), 'bias': jnp.zeros((Cz,))}}
        fn = lambda a: accel.fused_triangle_multiplicative_update(a, mask, params, True)
        report['checks'].append(grad_check(fn, (activations,), 'triangle_multiplicative_update'))
    except Exception as error:
        report['checks'].append({'label': 'triangle_multiplicative_update',
                                 'error': repr(error)[:300]})

    print(json.dumps(report, indent=2))
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=2)


if __name__ == '__main__':
    main()
