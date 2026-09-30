"""AF2 backward pass on MPS, through BindCraft2's own sequence-gradient pieces.

`bindcraft/af2.py:378` compiles `jax.value_and_grad(sequence_design_loss, argnums=2)`, where
`sequence_design_loss` calls `prepare_design_sequence_features` -> `alphafold_input_features`
-> `recycled_alphafold_outputs` -> `alphafold_prediction_metrics` and then the loss registry
from `bindcraft/loss.py`. This script reuses the first four verbatim and substitutes a plain
scalar for the loss registry, so it exercises the same AF2 gradient path without needing the
campaign's ProteinStates/DesignLoss plumbing.

This measures whether a gradient w.r.t. the design sequence -- the thing sequence
optimization iterates on -- runs on MPS and agrees with CPU.

BindCraft2 is source-available under a hosting-restricted, non-OSI licence. It is read from a
separate local checkout and is neither vendored nor redistributed here.
"""
import argparse
import json
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
    parser.add_argument('--float32', action='store_true')
    parser.add_argument('--arrays', default=None)
    args = parser.parse_args()

    from bindcraft import af2 as bc_af2
    from bindcraft.protein import Protein, ResidueFlags, real_residue_weights
    from bindcraft.prediction import residue_chain_ids

    report = {'backend': jax.default_backend(), 'jax': jax.__version__,
              'device': str(jax.devices()[0]), 'model': args.model,
              'length': args.length, 'bfloat16': not args.float32}

    model = bc_af2.AlphaFoldDesignModel(
        presets=args.model, data_dir=args.data_dir, key=jax.random.PRNGKey(3),
        num_recycle=0, dropout=False, subbatch_size=None,
        attention_backend='auto', use_cueq=False)
    if args.float32:
        for runner in model.alphafold_runners.values():
            with runner.config.unlocked():
                runner.config.model.global_config.bfloat16 = False
                runner.config.model.global_config.bfloat16_output = False

    family = model.model_families[args.model]
    runner = model._alphafold_runner(family, None)
    parameters = model.model_parameters[args.model]

    L = args.length
    chain = Protein.empty(L, jax.random.PRNGKey(1))
    padded = bc_af2.padded_prediction_length(L, model.length_bucket_size)
    pad = padded - L
    sequence = jnp.pad(jnp.asarray(chain.sequence, jnp.float32), [[0, pad], [0, 0]])
    atoms = jnp.pad(jnp.asarray(chain.atoms, jnp.float32), [[0, pad], [0, 0], [0, 0]])
    atom_mask = jnp.pad(jnp.asarray(chain.atom_mask), [[0, pad], [0, 0]])  # stays bool
    flags = jnp.pad(jnp.asarray(chain.flags), [0, pad],
                    constant_values=int(ResidueFlags.PADDING))
    residue_index = jnp.pad(jnp.asarray(chain.residue_index), [0, pad])
    asym_id = jnp.pad(residue_chain_ids((L,)), [0, pad], constant_values=1)
    seq_mask = real_residue_weights(flags)
    interface_asym_id = bc_af2.interface_asym_ids(('binder',), (L,))
    interface_asym_id = jnp.pad(interface_asym_id, [0, pad], constant_values=1)

    # The same four calls bindcraft/af2.py:338-377 makes, with a plain scalar in place of the
    # loss registry so no DesignLoss plumbing is needed.
    def design_scalar(sequences):
        features, profile = bc_af2.prepare_design_sequence_features(
            sequences, flags, jnp.asarray(1.0), jnp.asarray(0.0),
            jnp.asarray(1.0), jnp.asarray(2.0), None)
        model_inputs = bc_af2.alphafold_input_features(
            features, profile, atoms, atom_mask, residue_index, asym_id, seq_mask, flags,
            jnp.asarray(False), model.cyclic_offset_mode, asym_id, 0.0, False)
        outputs = bc_af2.recycled_alphafold_outputs(
            runner, parameters, jax.random.PRNGKey(7), model_inputs, 0)
        metrics = bc_af2.alphafold_prediction_metrics(outputs, seq_mask, interface_asym_id)
        # plDDT is the confidence term every BindCraft2 trajectory loss leans on.
        loss = -(metrics['plddt'] * seq_mask).sum() / (seq_mask.sum() + 1e-8)
        return loss, (metrics['plddt'], metrics['ptm'])

    jitted = jax.jit(jax.value_and_grad(design_scalar, has_aux=True))
    compile_start = time.perf_counter()
    (loss, aux), grad = jax.block_until_ready(jitted(sequence))
    report['seconds_first_call_including_compile'] = time.perf_counter() - compile_start
    start = time.perf_counter()
    (loss, aux), grad = jax.block_until_ready(jitted(sequence))
    report['seconds_second_call'] = time.perf_counter() - start

    g = np.asarray(grad, dtype='float64')
    plddt, ptm = (np.asarray(x, dtype='float64') for x in aux)
    row_norms = np.linalg.norm(g, axis=-1)
    real = np.asarray(seq_mask) > 0
    report.update(
        loss=float(loss),
        plddt_mean=float(plddt[real].mean()),
        ptm=float(ptm),
        grad_norm=float(np.linalg.norm(g)),
        grad_absmax=float(np.abs(g).max()),
        grad_zero_fraction=float((g == 0).mean()),
        grad_zero_fraction_real_residues=float((g[real] == 0).mean()),
        zero_rows=int((row_norms == 0).sum()),
        zero_rows_among_real_residues=int((row_norms[real] == 0).sum()),
        finite=bool(np.isfinite(g).all()),
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30))
    print(json.dumps(report, indent=2))
    with open(args.out, 'w') as handle:
        json.dump(report, handle, indent=2)
    if args.arrays:
        np.savez(args.arrays, grad=g, plddt=plddt, ptm=ptm)


if __name__ == '__main__':
    main()
