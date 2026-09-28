"""Small de novo Mosaic AF2 design experiment; uses ordinary differentiable JAX.

Single-checkpoint loader adapted from Mosaic models/af2.py (MIT, see the
package's MOSAIC_LICENSE). Model equations, confidence losses and optimizer
are upstream Mosaic. No MSA search, templates or native-sequence initialization.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import time
from unittest.mock import patch

import equinox as eqx
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import TOKENS, LossTerm
from mosaic.alphafold.model import config, data, modules_multimer, state
from mosaic.models.af2 import (AlphaFold2, AFOutput, Distogram, StructureModuleOutputs,
    set_binder_sequence, af2_output_to_structure_model_output)
from mosaic.losses.confidence_metrics import confidence_metrics, _calculate_bin_centers
import mosaic.losses.structure_prediction as sp
import mosaic.optimizers as optimizers
from mosaic.losses.protein_mpnn import inverse_fold, InverseFoldingSequenceRecovery
from mosaic.proteinmpnn.mpnn import ProteinMPNN


class SingleAF2(AlphaFold2):
    def __init__(self, weights):
        name = 'model_1_multimer_v3'
        params = data.get_model_haiku_params(name, str(weights))
        self.stacked_parameters = jax.tree.map(lambda p: jnp.asarray(p)[None], params)
        self.multimer = True
        cfg = config.model_config(name)
        cfg.max_msa_clusters = cfg.max_extra_msa = 1
        cfg.masked_msa_replace_fraction = 0
        cfg.subbatch_size = None
        cfg.model.num_ensemble_eval = 1
        cfg.model.global_config.subbatch_size = None
        cfg.model.global_config.eval_dropout = False
        cfg.model.global_config.deterministic = True
        cfg.model.global_config.use_remat = True
        cfg.model.global_config.bfloat16 = False
        cfg.model.num_extra_msa = 1
        cfg.model.resample_msa_in_recycling = False

        def forward(features, previous_rep, use_dropout=False):
            prediction, state = modules_multimer.AlphaFold(cfg.model)(
                batch=features, prev_rep=previous_rep, use_dropout=use_dropout)
            confidence = confidence_metrics(prediction)
            return AFOutput(
                distogram=Distogram(**prediction['distogram']),
                iptm=confidence['iptm'],
                predicted_aligned_error=confidence['predicted_aligned_error'],
                pae_logits=prediction['predicted_aligned_error']['logits'],
                pae_bin_centers=_calculate_bin_centers(prediction['predicted_aligned_error']['breaks']),
                predicted_lddt_logits=prediction['predicted_lddt']['logits'],
                plddt=confidence['plddt'],
                structure_module=StructureModuleOutputs(
                    final_atom_mask=prediction['structure_module']['final_atom_mask'],
                    final_atom_positions=prediction['structure_module']['final_atom_positions']),
                recycling_state=state)
        self.af2_forward = hk.transform(forward).apply

    @eqx.filter_jit
    def _forward_with_raw(self, *, PSSM, features, recycling_steps=1, model_idx=0,
                          use_dropout=False, recycling_state=None, key):
        # Equivalent to upstream's scan of length one, with zero initial state.
        # Keep this experiment explicitly bounded to one checkpoint/pass.
        if recycling_steps != 1 or model_idx != 0 or recycling_state is not None:
            raise ValueError('This experimental adapter supports one pass/model only')
        features = set_binder_sequence(PSSM, features, multimer=True)
        n = features['aatype'].shape[0]
        previous = state.AlphaFoldState(prev_pos=jnp.zeros((n, 37, 3)),
            prev_msa_first_row=jnp.zeros((n, 256)), prev_pair=jnp.zeros((n, n, 128)))
        params = jax.tree.map(lambda p: p[0], self.stacked_parameters)
        raw = self.af2_forward(params, jax.random.fold_in(key, 1), features=features,
            previous_rep=previous, use_dropout=use_dropout)
        return af2_output_to_structure_model_output(features, raw), raw


class DesignLoss(LossTerm):
    model: SingleAF2
    features: dict
    terms: eqx.Module

    def __call__(self, x, *, key):
        output = self.model.model_output(PSSM=x, features=self.features,
            recycling_steps=1, model_idx=0, key=jax.random.key(7))
        return self.terms(x, output=output, key=jax.random.key(7))


def plain(tree):
    return jax.tree.map(lambda v: np.asarray(v).tolist(), tree)


def check(value, gradient):
    if not np.isfinite(float(value)) or not np.isfinite(np.asarray(gradient)).all():
        raise ValueError('Nonfinite RAW AF2 loss/gradient; refusing nan_to_num masking')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--length', type=int, default=64)
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--platform', choices=['mps', 'cpu'], required=True)
    ap.add_argument('--probe-only', action='store_true')
    ap.add_argument('--mpnn-weight', type=float, default=0.0)
    ap.add_argument('--mpnn-samples', type=int, default=4)
    args = ap.parse_args()
    if args.length < 32 or args.steps < 1:
        ap.error('length >=32 and steps >=1 required')
    if not np.isfinite(args.mpnn_weight) or args.mpnn_weight < 0 or args.mpnn_samples < 1:
        ap.error('mpnn-weight must be finite/nonnegative and mpnn-samples positive')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != args.platform:
        raise RuntimeError(f'Unexpected devices: {devices}')
    print('DEVICES', devices, flush=True)
    np.random.seed(args.seed)
    model = SingleAF2(args.weights)
    features, _ = model.binder_features(args.length, chains=[])
    features = jax.tree.map(jnp.asarray, features)
    terms = (sp.PLDDTLoss() + 0.05 * sp.WithinBinderPAE()
        + sp.WithinBinderContact(num_contacts_per_residue=8)
        + 0.1 * sp.DistogramRadiusOfGyration() + 0.3 * sp.HelixLoss())
    mpnn = None
    if args.mpnn_weight:
        mpnn = ProteinMPNN.from_pretrained(backbone_noise=0.0)
        terms = terms + args.mpnn_weight * InverseFoldingSequenceRecovery(
            mpnn, temp=jnp.asarray(0.1), num_samples=args.mpnn_samples)
    loss = DesignLoss(model, features, terms)
    # NumPy initialization ensures CPU and GPU start from exactly the same values.
    logits = np.random.default_rng(args.seed).gumbel(size=(args.length, 20)).astype(np.float32) * 0.5
    x = jnp.asarray(np.exp(logits) / np.exp(logits).sum(-1, keepdims=True))
    key = jax.random.key(args.seed)
    raw_vg = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))
    print('COMPILING_FULL_STRUCTURE_CONFIDENCE_GRADIENT', flush=True)
    t = time.perf_counter()
    (initial, initial_aux), gradient = jax.block_until_ready(raw_vg(x, key=key))
    compile_first_s = time.perf_counter() - t
    check(initial, gradient)
    if {d.platform for d in gradient.devices()} != {args.platform}:
        raise RuntimeError(f'Gradient is on unexpected devices: {gradient.devices()}')
    if np.linalg.norm(np.asarray(gradient) - np.asarray(gradient).mean(-1, keepdims=True)) < 1e-8:
        raise ValueError('No useful simplex gradient')
    np.savez(args.output / 'initial-gradient.npz', sequence=np.asarray(x), gradient=np.asarray(gradient), loss=np.asarray(initial))
    t = time.perf_counter()
    jax.block_until_ready(raw_vg(x, key=key))
    warm_s = time.perf_counter() - t
    report = dict(platform=args.platform, length=args.length, seed=args.seed,
        objective_seed=7,
        model='model_1_multimer_v3', precision='float32', forward_passes=1,
        initial_loss=float(initial), initial_aux=plain(initial_aux),
        gradient_devices=[str(d) for d in gradient.devices()],
        mpnn_weight=args.mpnn_weight, mpnn_samples=args.mpnn_samples,
        compile_and_first_gradient_seconds=compile_first_s, warm_gradient_seconds=warm_s,
        versions={n: importlib.metadata.version(n) for n in ['jax','jaxlib','jax-mps','equinox']},
        scope='Small de novo monomer, full AF2 confidence/structure gradients; not binder design or experimental validation.')
    (args.output / 'probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print('FULL_GRADIENT_OK', json.dumps(report), flush=True)
    if args.probe_only:
        return
    history = []
    best = {'value': float(initial), 'x': np.asarray(x).copy()}
    # Replace only the process-local optimizer evaluation helper. Preserve raw
    # errors rather than upstream's nan_to_num; optimizer math is unchanged.
    def checked_eval(loss_function, x, key):
        (value, aux), g = jax.block_until_ready(raw_vg(jnp.asarray(x, dtype=jnp.float32), key=key))
        check(value, g)
        if float(value) < best['value']:
            best.update(value=float(value), x=np.asarray(x).copy())
        history.append({'loss':float(value), 'aux':plain(aux)})
        (args.output/'trajectory.json').write_text(json.dumps(history, indent=2)+'\n')
        return (value, aux), g - g.mean(-1, keepdims=True)
    t = time.perf_counter()
    with patch.object(optimizers, '_eval_loss_and_grad', checked_eval):
        final_x, _ = optimizers.simplex_APGM(loss_function=loss, x=x, n_steps=args.steps,
            stepsize=0.2, momentum=0.0, key=key)
        checked_eval(loss, final_x, key)
    report['optimization_seconds'] = time.perf_counter()-t
    report['steps'] = args.steps
    report['best_soft_loss'] = best['value']
    x_best = jnp.asarray(best['x'])
    if not np.allclose(np.asarray(x_best).sum(-1), 1, atol=1e-5) or np.asarray(x_best).min() < -1e-6:
        raise ValueError('Invalid simplex')
    np.savez(args.output/'optimized.npz', probabilities=best['x'])
    tokens = np.asarray(x_best).argmax(-1)
    sequence = ''.join(TOKENS[i] for i in tokens)
    hard = jax.nn.one_hot(tokens, 20)
    print('PREDICTING_HARD_CANDIDATE', sequence, flush=True)
    output = jax.block_until_ready(model.model_output(PSSM=hard, features=features,
        recycling_steps=1, model_idx=0, key=key))
    if not np.isfinite(np.asarray(output.backbone_coordinates)).all():
        raise ValueError('Nonfinite exported backbone')
    output.to_structure().write_pdb(str(args.output/'hallucinated.pdb'))
    candidates = [{'name':'hallucinated', 'sequence':sequence, 'design_plddt':float(output.plddt.mean())}]
    # Standard MPNN sequence redesign on the newly hallucinated backbone.
    if mpnn is None:
        mpnn = ProteinMPNN.from_pretrained(backbone_noise=0.0)
    redesign = eqx.filter_jit(lambda k: inverse_fold(mpnn, args.length, output,
        temp=0.1, key=k, jacobi_iterations=10))
    for i in range(2):
        redesigned = np.asarray(jax.block_until_ready(redesign(jax.random.key(args.seed+i+100))))
        candidates.append({'name':f'mpnn_{i+1}', 'sequence':''.join(TOKENS[v] for v in redesigned)})
    for item in candidates:
        (args.output/f"{item['name']}.fasta").write_text(f">{item['name']} computational_candidate\n{item['sequence']}\n")
        (args.output/f"{item['name']}.yaml").write_text('version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: '+item['sequence']+'\n      msa: empty\n')
    report.update(candidates=candidates, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        total_seconds=time.perf_counter()-start, device_memory_stats=devices[0].memory_stats())
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print('DESIGN_COMPLETE', json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
