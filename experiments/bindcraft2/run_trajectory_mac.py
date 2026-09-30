"""Run one BindCraft2 binder-design trajectory on Apple Silicon.

This is our own driver. BindCraft2 is source-available under a hosting-restricted, non-OSI
licence; it is imported from a separate checkout and neither copied nor redistributed here.

It bypasses BindCraft2's launcher rather than patching it. `bindcraft/cli.py` and
`bindcraft/design_workers.py` plan GPU workers by shelling out to `nvidia-smi` and setting
CUDA_VISIBLE_DEVICES, and `bindcraft/selfcheck.py` probes for CUDA plugin modules. None of
that is needed: importing `bindcraft.trajectory`, `bindcraft.af2`, `bindcraft.protein` and
`bindcraft.settings` pulls in none of those three modules and issues no subprocess call at
all -- verified, not assumed. `run_trajectory` (trajectory.py:278) is the real four-stage
machine, so we call it directly.

One patch is required. `_alphafold_runner` (af2.py:243-259) sets
`model_config.model.global_config.bfloat16 = True` on a fresh deepcopy of the config, so
there is nothing upstream to override. On this backend bfloat16 is both SLOWER than float32
(0.633 s vs 0.573 s for a 100-residue multimer) and responsible for essentially all the
CPU/MPS numerical disagreement -- a full AF2 forward moves from 5.2e-04 relative in float32
to 3.4e-01 in bfloat16. So we wrap the method.

Wrapping the method matters rather than mutating runners after construction: runners are
cached on `(model_family, subbatch_size, attention_backend, use_cueq)` (af2.py:62) and
created lazily, so a post-hoc walk over `model.alphafold_runners` catches only the small
probe runner built in `__init__` and leaves any later one in bfloat16.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import jax


def force_float32_runners():
    """Make every AF2 runner BindCraft2 builds use float32 instead of bfloat16.

    Returns a description of what was patched, for the run record.
    """
    from bindcraft import af2 as bc_af2

    original = bc_af2.AlphaFoldDesignModel._alphafold_runner

    def float32_runner(self, model_family, subbatch_size):
        runner = original(self, model_family, subbatch_size)
        with runner.config.unlocked():
            runner.config.model.global_config.bfloat16 = False
            runner.config.model.global_config.bfloat16_output = False
        return runner

    bc_af2.AlphaFoldDesignModel._alphafold_runner = float32_runner
    return 'AlphaFoldDesignModel._alphafold_runner wrapped: bfloat16 and bfloat16_output False'


def designed_sequences(protein_states, target_chain_prefix):
    """Pull the designed binder sequence out of run_trajectory's protein_states.

    is_target_chain needs the prefix the run used; target chains are named
    <prefix>_<state> by target_chain_name, so anything not matching is a binder.
    """
    import numpy as np
    from bindcraft.protein import AMINO_ACIDS, is_target_chain

    found = {}
    for state_name, chains in protein_states.items():
        for chain_name, protein in chains.items():
            if is_target_chain(chain_name, target_chain_prefix):
                continue
            tokens = np.asarray(protein.sequence).argmax(-1)
            found[f'{state_name}/{chain_name}'] = ''.join(AMINO_ACIDS[t] for t in tokens)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('settings', help='BindCraft2 settings JSON')
    parser.add_argument('--data-dir', required=True, help='AF2 params directory')
    parser.add_argument('--out', required=True, help='trajectory output directory')
    parser.add_argument('--preset', default='model_1_multimer_v3',
                        help='single AF2 preset; the full pool costs ~1.9 GB of parameters')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--subbatch', default=None,
                        help="AF2 attention subbatch size. BindCraft2's resolve_subbatch_size "
                             'returns None below 384 residues, so attention is never chunked '
                             'and peak memory exceeds 24 GiB at ~140 residues. An explicit '
                             'integer chunks it.')
    parser.add_argument('--report', default=None, help='where to write the JSON run record')
    args = parser.parse_args()

    patched = force_float32_runners()

    from bindcraft.settings import read_settings, build_design_settings, design_stage_rounds
    from bindcraft.trajectory import run_trajectory

    settings = read_settings(args.settings)
    design_settings = build_design_settings(settings)
    rounds = design_stage_rounds(settings)

    record = {
        'device': str(jax.devices()[0]),
        'jax': jax.__version__,
        'patch': patched,
        'preset': args.preset,
        'seed': args.seed, 'subbatch_size': args.subbatch or 'auto',
        'stage_rounds': rounds,
        'gradient_rounds': sum(v for k, v in rounds.items() if k != 'mutate'),
        'binder_lengths': list(design_settings.binder.lengths),
        'targets': [{'name': t.name, 'path': t.path, 'chains': t.chains}
                    for t in design_settings.targets],
    }
    print(json.dumps(record, indent=1), flush=True)

    start = time.perf_counter()
    from bindcraft.af2 import AlphaFoldDesignModel
    model = AlphaFoldDesignModel(
        presets=(args.preset,), models=(args.preset,), data_dir=args.data_dir,
        key=jax.random.PRNGKey(args.seed), num_recycle=int(settings.get('design_recycles', 1)),
        dropout=bool(settings.get('design_dropout', True)),
        attention_backend='auto', use_cueq=False,
        subbatch_size=(int(args.subbatch) if args.subbatch not in (None, 'auto')
                       else 'auto'))
    record['model_load_seconds'] = time.perf_counter() - start
    print('model loaded in %.1f s' % record['model_load_seconds'], flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    protein_states, predictions, failure = run_trajectory(
        design_settings, model, jax.random.PRNGKey(args.seed),
        trajectory_directory=args.out)
    record['trajectory_seconds'] = time.perf_counter() - start
    record['process_peak_rss_gb'] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
    stats = jax.devices()[0].memory_stats()
    if stats:
        record['device_peak_gb'] = stats['peak_bytes_in_use'] / 1e9
        record['device_limit_gb'] = stats.get('bytes_limit', 0) / 1e9
        record['device_pool_gb'] = stats.get('pool_bytes', 0) / 1e9
    record['failure'] = failure
    record['sequences'] = designed_sequences(
        protein_states, settings.get('target_chain', 'target'))
    record['prediction_states'] = sorted(predictions) if predictions else []

    print(json.dumps({k: v for k, v in record.items() if k != 'targets'}, indent=1), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
