"""Experimental single-chain fixed-backbone design using Mosaic ProteinMPNN."""
import argparse
import importlib.metadata
import json
import resource
import time
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import TOKENS, LossTerm
from mosaic.losses.protein_mpnn import FixedStructureInverseFoldingLL, load_chain
from mosaic.optimizers import simplex_APGM
from mosaic.proteinmpnn.mpnn import ProteinMPNN


class FixedOrder(LossTerm):
    inner: FixedStructureInverseFoldingLL
    seed: int = 7

    def __call__(self, x, *, key):
        # Deterministic objective for the numerical/optimizer check only.
        return self.inner(x, key=jax.random.key(self.seed))


def ready(x):
    return jax.block_until_ready(x)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--structure', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--platform', choices=['cpu', 'mps'], required=True)
    ap.add_argument('--steps', type=int, default=25)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    if args.steps < 1:
        ap.error("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    devices = jax.devices()
    require(devices[0].platform == args.platform, f'Unexpected devices: {devices}')
    print('DEVICES', devices, flush=True)
    np.random.seed(args.seed)
    st = gemmi.read_structure(str(args.structure))
    st.remove_ligands_and_waters()
    st.remove_alternative_conformations()
    st.remove_empty_chains()
    require(len(st) == 1 and len(st[0]) == 1, 'Design requires exactly one protein chain')
    native, coords = load_chain(st[0][0])
    require(np.isfinite(coords).all(), 'Backbone contains missing atoms')
    require(set(native) <= set(TOKENS), 'Backbone contains noncanonical residues')
    mpnn = ProteinMPNN.from_pretrained(backbone_noise=0.0)
    loss = FixedOrder(FixedStructureInverseFoldingLL.from_structure(st, mpnn), seed=args.seed)
    ready(loss.inner.encoded_state)
    encoded_s = time.perf_counter() - start
    n = len(native)
    x = jnp.full((n, 20), 1 / 20, dtype=jnp.float32)
    key = jax.random.key(args.seed)
    vg = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))
    f = eqx.filter_jit(loss)
    t = time.perf_counter()
    (value, aux), grad = ready(vg(x, key=key))
    compile_first_s = time.perf_counter() - t
    value_np, grad_np = float(value), np.asarray(grad)
    require(np.isfinite(value_np) and np.isfinite(grad_np).all(), 'Nonfinite raw loss or gradient')
    require(np.linalg.norm(grad_np - grad_np.mean(-1, keepdims=True)) > 1e-5, 'Gradient has no useful simplex direction')
    timings = []
    for _ in range(5):
        t = time.perf_counter()
        ready(vg(x, key=key))
        timings.append(time.perf_counter() - t)
    direction = grad_np - grad_np.mean(-1, keepdims=True)
    direction /= np.linalg.norm(direction)
    eps = 0.01
    vp = float(ready(f(x + eps * direction, key=key))[0])
    vm = float(ready(f(x - eps * direction, key=key))[0])
    fd = (vp - vm) / (2 * eps)
    ad = float(np.sum(grad_np * direction))
    fd_rel_error = abs(fd - ad) / max(abs(ad), 1e-8)
    require(fd_rel_error < 0.02, f'Finite difference mismatch: {(fd, ad, fd_rel_error)}')
    print('RAW_GRADIENT_OK', value_np, float(np.linalg.norm(grad_np)), 'FD', fd_rel_error, flush=True)
    history = []

    def trajectory(aux, p):
        # Check raw values each iteration before any nan_to_num can hide failure.
        (v, _), g = ready(vg(jnp.asarray(p, dtype=jnp.float32), key=key))
        require(np.isfinite(float(v)) and np.isfinite(np.asarray(g)).all(), 'Nonfinite optimization loss or gradient')
        p = np.asarray(p)
        require(p.min() >= -1e-7 and np.allclose(p.sum(-1), 1, atol=1e-6), 'Invalid sequence probabilities')
        history.append(float(v))
        return float(v)

    t = time.perf_counter()
    final, _, _ = simplex_APGM(
        loss_function=loss, x=x, n_steps=args.steps, stepsize=0.5,
        momentum=0.0, key=key, trajectory_fn=trajectory,
    )
    ready(final)
    optimizer_s = time.perf_counter() - t
    final = np.asarray(final, dtype=np.float32)
    final_loss = float(ready(f(jnp.asarray(final), key=key))[0])
    require(final_loss < value_np - 0.001, f'No meaningful objective improvement: {value_np} -> {final_loss}')
    candidate = ''.join(TOKENS[i] for i in final.argmax(-1))
    hard = jax.nn.one_hot(final.argmax(-1), 20)
    hard_loss = float(ready(f(hard, key=key))[0])
    require(np.isfinite(hard_loss), 'Nonfinite discrete sequence loss')
    summary = {
        'platform': args.platform, 'devices': [str(d) for d in devices],
        'residues': n, 'dtype': str(x.dtype), 'seed': args.seed,
        'versions': {p: importlib.metadata.version(p) for p in
                     ['jax', 'jaxlib', 'jax-mps', 'equinox', 'torch', 'numpy', 'gemmi']},
        'load_encode_seconds': encoded_s,
        'compile_first_gradient_seconds': compile_first_s,
        'warm_gradient_seconds': timings,
        'warm_gradient_median_seconds': float(np.median(timings)),
        'initial_loss': value_np, 'gradient_norm': float(np.linalg.norm(grad_np)),
        'directional_autodiff': ad, 'directional_finite_difference': fd,
        'finite_difference_relative_error': fd_rel_error,
        'steps': args.steps, 'optimizer_with_raw_checks_seconds': optimizer_s,
        'final_soft_loss': final_loss, 'final_discrete_loss': hard_loss,
        'loss_history': history,
        'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'total_seconds': time.perf_counter() - start,
        'scope': 'Experimental fixed-backbone design; candidate has not been refolded or experimentally validated.',
    }
    np.savez(args.output / 'arrays.npz', initial=np.asarray(x), gradient=grad_np,
             final=final, encoded_nodes=np.asarray(loss.inner.encoded_state[0]),
             encoded_edges=np.asarray(loss.inner.encoded_state[1]),
             neighbor_indices=np.asarray(loss.inner.encoded_state[2]))
    (args.output / 'candidate.fasta').write_text('>macfoldkit_experimental_design\n' + candidate + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
