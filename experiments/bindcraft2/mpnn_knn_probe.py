"""Does a gradient reach BindCraft2's ProteinMPNN k-nearest-neighbour graph, and does that
graph come out the same on MPS as on CPU?

`bindcraft/mpnn/modules.py:200` builds the KNN graph with

    return jax.lax.approx_min_k(D_masked, k, reduction_dimension=-1)[1]

-- `approx_min_k`, not `top_k` or `argsort`, and subscript [1] keeps only the int32 indices.
Two things follow that this script measures rather than asserts:

  1. an integer output carries no cotangent, so no gradient can flow through the *selection*
     (it still flows through the gathered distance values at modules.py:214); and
  2. `approx_min_k` is an approximate primitive, so a backend is free to return a different
     neighbour set. That is a forward-correctness question for MPS independent of gradients,
     so the indices are checked against an exact argsort and against the other backend.

BindCraft2 is source-available under a hosting-restricted, non-OSI licence. It is read from a
separate local checkout and is neither vendored nor redistributed here.
"""
import json
import sys
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from bindcraft.mpnn import modules as mpnn_modules


def main():
    out = sys.argv[1]
    report = {'backend': jax.default_backend(), 'jax': jax.__version__,
              'device': str(jax.devices()[0]), 'checks': []}
    store = {}

    L, K = 96, 48                      # v_48_020 reports 48 neighbours, not the class default 30
    rng = np.random.default_rng(5)
    # A plausible backbone: a noisy helix, so the neighbour ordering is not degenerate.
    t = np.arange(L)
    ca = np.stack([2.3 * np.cos(t * 1.75), 2.3 * np.sin(t * 1.75), 1.5 * t], -1)
    backbone = (ca[:, None, :] + rng.standard_normal((L, 4, 3)) * 0.6).astype('float32')
    mask_np = np.ones((L,), dtype='float32')
    mask_np[88:] = 0.0

    X = jnp.asarray(backbone)
    mask = jnp.asarray(mask_np)
    residue_idx = jnp.arange(L, dtype=jnp.int32)
    chain_idx = jnp.where(jnp.arange(L) < 48, 0, 1).astype(jnp.int32)

    def features(x):
        module = mpnn_modules.ProteinFeatures(
            edge_features=128, node_features=128, top_k=K, augment_eps=0.0)
        return module({'X': x, 'mask': mask,
                       'residue_idx': residue_idx, 'chain_idx': chain_idx})

    transformed = hk.transform(features)
    params = transformed.init(jax.random.PRNGKey(0), X)

    # --- the neighbour indices themselves -------------------------------------------------
    edges = jax.jit(lambda p, x: transformed.apply(p, jax.random.PRNGKey(0), x)[1])
    e_idx = np.asarray(jax.block_until_ready(edges(params, X)))
    store['E_idx'] = e_idx

    # Exact reference, computed the same way modules.py:196-199 does.
    dx = backbone[:, 1, :][None, :, :] - backbone[:, 1, :][:, None, :]
    d = np.sqrt((dx ** 2).sum(-1) + 1e-6)
    mask_2d = mask_np[None, :] * mask_np[:, None]
    d_masked = np.where(mask_2d.astype(bool), d, d.max(-1, keepdims=True))
    exact = np.argsort(d_masked, axis=-1, kind='stable')[:, :K]
    agree_as_set = float(np.mean([
        len(set(e_idx[i]) & set(exact[i])) / K for i in range(L)]))
    report['checks'].append(dict(
        label='approx_min_k_indices',
        dtype=str(e_idx.dtype),
        integer_output=bool(np.issubdtype(e_idx.dtype, np.integer)),
        exact_order_match_fraction=float((e_idx == exact).mean()),
        neighbour_set_overlap_fraction=agree_as_set,
        k=K, residues=L))

    # --- can a gradient reach the selection at all? --------------------------------------
    # Differentiate a function of only the indices. If any cotangent existed it would show up
    # here; JAX gives integers no tangent space, so this must be exactly zero.
    def index_only(x):
        idx = transformed.apply(params, jax.random.PRNGKey(0), x)[1]
        return jnp.sum(idx.astype(jnp.float32) ** 2)
    try:
        g = np.asarray(jax.block_until_ready(jax.jit(jax.grad(index_only))(X)))
        report['checks'].append(dict(
            label='grad_through_indices_only',
            grad_norm=float(np.linalg.norm(g.astype('float64'))),
            all_exactly_zero=bool((g == 0).all()),
            finite=bool(np.isfinite(g).all())))
    except Exception as error:  # noqa: BLE001
        report['checks'].append({'label': 'grad_through_indices_only',
                                 'error': repr(error)[:400]})

    # --- the counterfactual: if anyone DID differentiate the featuriser, does MPS match? ---
    def edge_energy(p, x):
        e, _ = transformed.apply(p, jax.random.PRNGKey(0), x)
        return jnp.sum(e ** 2)
    jitted = jax.jit(jax.value_and_grad(edge_energy, argnums=(0, 1)))
    value, grads = jax.block_until_ready(jitted(params, X))
    start = time.perf_counter()
    value, grads = jax.block_until_ready(jitted(params, X))
    seconds = time.perf_counter() - start
    param_grads, x_grad = grads
    xg = np.asarray(x_grad)
    store['grad_X'] = xg
    store['forward_E'] = np.asarray(jax.jit(
        lambda p, x: transformed.apply(p, jax.random.PRNGKey(0), x)[0])(params, X))
    names, norms, zeros = [], [], []
    for module in sorted(param_grads):
        for leaf in sorted(param_grads[module]):
            g = np.asarray(param_grads[module][leaf])
            names.append(f'{module}/{leaf}')
            norms.append(float(np.linalg.norm(g.astype('float64'))))
            zeros.append(float((g == 0).mean()))
            store[f'grad_param::{module}/{leaf}'] = g
    row_norms = np.linalg.norm(xg.astype('float64').reshape(L, -1), axis=1)
    report['checks'].append(dict(
        label='grad_through_protein_features',
        value=float(value), seconds=seconds,
        grad_names=names + ['X'],
        grad_norms=norms + [float(np.linalg.norm(xg.astype('float64')))],
        grad_zero_fraction=zeros + [float((xg == 0).mean())],
        x_zero_rows=int((row_norms == 0).sum()),
        x_zero_rows_are_padding=bool(
            set(np.flatnonzero(row_norms == 0).tolist()) <= set(np.flatnonzero(mask_np == 0).tolist())),
        finite=bool(np.isfinite(xg).all())))

    print(json.dumps(report, indent=2))
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=2)
    np.savez(out.replace('.json', '') + '-arrays.npz', **store)


if __name__ == '__main__':
    main()
