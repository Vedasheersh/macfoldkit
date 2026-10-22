"""One full BindCraft2 AF2 forward pass, on CPU or MPS, with real AlphaFold parameters.

Builds `bindcraft.af2.AlphaFoldDesignModel` directly and calls `.predict()` on a single
short binder chain. This is the ordinary prediction path -- `RunModel.apply` under
`jax.jit`, Haiku `layer_stack`/remat, bfloat16 evoformer -- and it deliberately avoids
`bindcraft.cli`, `design_workers` and `selfcheck`, the three modules that do NVIDIA device
discovery and multiprocessing. Nothing in the BindCraft2 checkout is modified.

BindCraft2 is source-available under a hosting-restricted, non-OSI licence. It is read from a
separate local checkout and is neither vendored nor redistributed here.
"""
import argparse
import json
import os
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model', default='model_1_ptm')
    parser.add_argument('--length', type=int, default=40)
    parser.add_argument('--target-length', type=int, default=0,
                        help='if set, add a second chain of this length (complex prediction)')
    parser.add_argument('--num-recycle', type=int, default=0)
    parser.add_argument('--arrays', default=None)
    parser.add_argument('--float32', action='store_true',
                        help='turn the evoformer bfloat16 off, to separate bf16 rounding '
                             'from a backend discrepancy')
    args = parser.parse_args()

    from bindcraft.af2 import AlphaFoldDesignModel
    from bindcraft.af import accel
    from bindcraft.protein import Protein

    report = {
        'backend': jax.default_backend(),
        'jax': jax.__version__,
        'device': str(jax.devices()[0]),
        'attention_backend': accel.supported_attention_backend('auto'),
        'cuequivariance_available': accel.cuequivariance_available(),
        'model': args.model,
        'binder_length': args.length,
        'target_length': args.target_length,
        'num_recycle': args.num_recycle,
    }

    key = jax.random.PRNGKey(0)
    binder_key, target_key = jax.random.split(key)
    complex_chains = {'binder': Protein.empty(args.length, binder_key)}
    if args.target_length:
        complex_chains['target_probe'] = Protein.empty(args.target_length, target_key)
    states = {'binder_alone': complex_chains}

    build_start = time.perf_counter()
    model = AlphaFoldDesignModel(
        presets=args.model,
        data_dir=args.data_dir,
        key=jax.random.PRNGKey(3),
        num_recycle=args.num_recycle,
        dropout=False,
        subbatch_size=None,
        attention_backend='auto',
        use_cueq=False,
    )
    report['seconds_build_and_load_params'] = time.perf_counter() - build_start

    # AlphaFoldDesignModel._alphafold_runner hardcodes global_config.bfloat16 = True. The
    # runner's forward function closes over self.config and reads it at trace time, and no
    # trace has happened yet, so mutating it here is enough -- and it leaves the BindCraft2
    # checkout untouched.
    report['bfloat16'] = not args.float32
    if args.float32:
        for runner in model.alphafold_runners.values():
            with runner.config.unlocked():
                runner.config.model.global_config.bfloat16 = False
                runner.config.model.global_config.bfloat16_output = False
    report['parameter_tensors'] = len(jax.tree_util.tree_leaves(
        model.model_parameters[args.model]))
    report['parameter_count'] = int(sum(
        np.prod(x.shape) for x in jax.tree_util.tree_leaves(model.model_parameters[args.model])))

    compile_start = time.perf_counter()
    predictions = model.predict(states, model=args.model)
    jax.block_until_ready(jax.tree_util.tree_leaves(predictions['binder_alone'].metrics))
    report['seconds_first_call_including_compile'] = time.perf_counter() - compile_start

    first_plddt = np.asarray(predictions['binder_alone'].metrics['plddt'])

    run_start = time.perf_counter()
    predictions = model.predict(states, model=args.model)
    jax.block_until_ready(jax.tree_util.tree_leaves(predictions['binder_alone'].metrics))
    report['seconds_second_call'] = time.perf_counter() - run_start
    # Within-backend determinism: the same key and the same inputs, run twice.
    report['repeat_bit_identical'] = bool(np.array_equal(
        first_plddt, np.asarray(predictions['binder_alone'].metrics['plddt'])))

    prediction = predictions['binder_alone']
    metrics = {}
    store = {}
    for name, value in sorted(prediction.metrics.items()):
        array = np.asarray(value, dtype='float64')
        store[f'metric::{name}'] = array
        metrics[name] = dict(shape=list(array.shape), mean=float(array.mean()),
                             min=float(array.min()), max=float(array.max()),
                             finite=bool(np.isfinite(array).all()))
    report['metrics'] = metrics

    chains = {}
    for chain_name, chain in sorted(prediction.protein_complex.items()):
        atoms = np.asarray(chain.atoms, dtype='float64')
        mask = np.asarray(chain.atom_mask)
        store[f'atoms::{chain_name}'] = atoms
        store[f'atom_mask::{chain_name}'] = mask
        present = atoms[mask.astype(bool)]
        ca = atoms[:, 1, :]
        span = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
        chains[chain_name] = dict(
            residues=int(atoms.shape[0]),
            resolved_atoms=int(mask.sum()),
            coordinate_absmax=float(np.abs(present).max()) if present.size else 0.0,
            radius_of_gyration=float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean())),
            ca_ca_span_mean=float(span.mean()),
            ca_ca_span_max=float(span.max()),
            finite=bool(np.isfinite(present).all()),
        )
    report['chains'] = chains
    report['peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)

    print(json.dumps(report, indent=2))
    with open(args.out, 'w') as handle:
        json.dump(report, handle, indent=2)
    if args.arrays:
        np.savez(args.arrays, **store)


if __name__ == '__main__':
    main()
