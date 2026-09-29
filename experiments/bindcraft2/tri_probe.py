"""Exercise BindCraft2's non-fused (XLA) triangle multiplication on MPS, forward and backward.

`bindcraft/af/alphafold/model/modules.py` has three triangle-multiplication paths:

  TriangleMultiplication.__call__
    -> _fused_triangle_multiplication   (config.fuse_projection_weights = True, the AF-M 2.3+
       default), which itself dispatches to _cueq_triangle_multiplication only when
       global_config.use_cueq is set AND accel.cuequivariance_available(); otherwise it runs
       ordinary JAX. This is the fallback the CUDA-only fused kernel declines to.
    -> _triangle_multiplication        (config.fuse_projection_weights = False, AF2 / AF-M<2.3)

Both ordinary-JAX paths are run here, outgoing and incoming, forward and backward, against a
full mask and a padded mask. Compare the CPU and MPS JSON outputs.

BindCraft2 is source-available under a hosting-restricted, non-OSI licence. It is read from a
separate local checkout and is neither vendored nor redistributed here.
"""
import json
import sys
import time

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from bindcraft.af.alphafold.model import modules


def make_config(fuse, outgoing, intermediate):
    config = ml_collections.ConfigDict({
        'equation': 'ikc,jkc->ijc' if outgoing else 'kjc,kic->ijc',
        'num_intermediate_channel': intermediate,
        'fuse_projection_weights': fuse,
        'dropout_rate': 0.0,
        'orientation': 'per_row',
        'shared_dropout': True,
    })
    global_config = ml_collections.ConfigDict({
        'zero_init': False,      # zero_init would null the output projection and its gradient
        'bfloat16': False,
        'bfloat16_output': False,
        'multimer_mode': True,
        'subbatch_size': 4,
        'use_dgram': False,
        'use_remat': False,
        'use_cueq': False,       # cuequivariance_available() is false here anyway
    })
    return config, global_config


def randomise(params, rng):
    """Replace Haiku's initialisation with identical deterministic values on every backend."""
    out = {}
    for module, leaves in params.items():
        out[module] = {}
        for name, value in leaves.items():
            shape = tuple(value.shape)
            draw = rng.standard_normal(shape).astype('float32')
            if name == 'scale':
                out[module][name] = jnp.asarray(1.0 + 0.1 * draw)
            elif name == 'offset':
                out[module][name] = jnp.asarray(0.1 * draw)
            elif name == 'bias':
                out[module][name] = jnp.asarray(0.1 * draw)
            else:
                out[module][name] = jnp.asarray(draw / np.sqrt(shape[0]))
    return out


def summarise(tree):
    flat = jax.tree_util.tree_leaves(tree)
    return flat


def grad_check(apply_fn, params, act, mask, label, store):
    def scalar(p, a):
        return jnp.sum(apply_fn(p, None, a, mask) ** 2)

    jitted = jax.jit(jax.value_and_grad(scalar, argnums=(0, 1)))
    value, grads = jax.block_until_ready(jitted(params, act))
    start = time.perf_counter()
    value, grads = jax.block_until_ready(jitted(params, act))
    seconds = time.perf_counter() - start

    param_grads, act_grad = grads
    names, norms, zeros = [], [], []
    for module in sorted(param_grads):
        for leaf in sorted(param_grads[module]):
            g = np.asarray(param_grads[module][leaf])
            names.append(f'{module}/{leaf}')
            norms.append(float(np.linalg.norm(g.astype('float64'))))
            zeros.append(float((g == 0).mean()))
            store[f'{label}::grad::{module}/{leaf}'] = g
    names.append('activations')
    ag = np.asarray(act_grad)
    norms.append(float(np.linalg.norm(ag.astype('float64'))))
    zeros.append(float((ag == 0).mean()))
    store[f'{label}::grad::activations'] = ag

    # Row-level zero detection: the signature of a scatter/sort backward defect is a whole
    # activation row receiving no gradient on one backend but not the other.
    row_norms = np.linalg.norm(ag.astype('float64').reshape(ag.shape[0], -1), axis=1)
    forward = jax.block_until_ready(jax.jit(apply_fn)(params, None, act, mask))
    fwd = np.asarray(forward)
    store[f'{label}::forward'] = fwd

    return dict(
        label=label,
        value=float(value),
        seconds=seconds,
        forward_sum=float(fwd.astype('float64').sum()),
        forward_absmax=float(np.abs(fwd).max()),
        grad_names=names,
        grad_norms=norms,
        grad_zero_fraction=zeros,
        activation_zero_rows=int((row_norms == 0).sum()),
        activation_row_norm_min=float(row_norms.min()),
        finite=bool(np.isfinite(ag).all() and all(
            np.isfinite(np.asarray(param_grads[m][l])).all()
            for m in param_grads for l in param_grads[m])),
    )


def main():
    out = sys.argv[1]
    report = {
        'backend': jax.default_backend(),
        'jax': jax.__version__,
        'haiku': hk.__version__,
        'device': str(jax.devices()[0]),
        'checks': [],
    }
    store = {}

    N, C, INTERMEDIATE = (int(x) for x in (sys.argv[2:5] or ('48', '32', '24')))
    report['shape'] = dict(residues=N, channels=C, intermediate=INTERMEDIATE)
    seed_rng = np.random.default_rng(11)
    act_np = seed_rng.standard_normal((N, N, C)).astype('float32')
    full_mask = np.ones((N, N), dtype='float32')
    seq_mask = np.ones((N,), dtype='float32')
    seq_mask[40:] = 0.0                      # eight padded residues
    padded_mask = (seq_mask[:, None] * seq_mask[None, :]).astype('float32')

    for fuse in (True, False):
        for outgoing in (True, False):
            for mask_name, mask_np in (('full', full_mask), ('padded', padded_mask)):
                path = 'fused_xla' if fuse else 'unfused_af2'
                direction = 'outgoing' if outgoing else 'incoming'
                label = f'triangle_{path}_{direction}_{mask_name}mask'
                try:
                    config, global_config = make_config(fuse, outgoing, INTERMEDIATE)

                    def forward(a, m):
                        return modules.TriangleMultiplication(config, global_config)(a, m)

                    transformed = hk.transform(forward)
                    act = jnp.asarray(act_np)
                    mask = jnp.asarray(mask_np)
                    init_params = transformed.init(jax.random.PRNGKey(0), act, mask)
                    params = randomise(
                        {k: dict(v) for k, v in init_params.items()},
                        np.random.default_rng(23))
                    report['checks'].append(
                        grad_check(transformed.apply, params, act, mask, label, store))
                except Exception as error:  # noqa: BLE001
                    report['checks'].append({'label': label, 'error': repr(error)[:600]})

    print(json.dumps(report, indent=2))
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=2)
    np.savez(out.replace('.json', '') + '-arrays.npz', **store)


if __name__ == '__main__':
    main()
