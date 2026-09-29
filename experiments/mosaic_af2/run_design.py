"""Small de novo Mosaic AF2 design experiment; uses ordinary differentiable JAX.

Single-checkpoint loader adapted from Mosaic models/af2.py (MIT, see the
package's MOSAIC_LICENSE). Model equations, confidence losses and optimizer
are upstream Mosaic, with an equivalent contact-loss reduction for Metal.
Optional binder mode conditions on one complete target-chain template. No MSA
search or external sequence upload is performed.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time
from unittest.mock import patch

import equinox as eqx
import gemmi
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
from mosaic.structure_prediction import TargetChain

from metal_losses import WithinBinderContact, BinderTargetContact


class SingleAF2(AlphaFold2):
    def __init__(self, weights, use_remat=True):
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
        # Gradient checkpointing trades recomputation for activation memory. On a
        # 24 GiB machine a 124-231 residue complex peaks around 2.7-5.4 GB, so the
        # trade can be worth reversing; --remat off measures it.
        cfg.model.global_config.use_remat = use_remat
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


def load_target(path):
    """Read one complete standard-protein target, retaining exact residue order."""
    structure = gemmi.read_structure(str(path))
    structure.remove_ligands_and_waters()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()
    if len(structure) != 1 or len(structure[0]) != 1:
        raise ValueError('Target must contain exactly one protein chain and model')
    chain = structure[0][0]
    sequence = gemmi.one_letter_code([r.name for r in chain])
    if not sequence or any(aa not in TOKENS for aa in sequence):
        raise ValueError('Target requires standard amino acids with a complete sequence')
    if len({str(r.seqid) for r in chain}) != len(chain):
        raise ValueError('Target residue identifiers must be unique')
    for residue in chain:
        for name in ('N', 'CA', 'C', 'O'):
            atom = residue.sole_atom(name)
            if not np.isfinite([atom.pos.x, atom.pos.y, atom.pos.z]).all():
                raise ValueError(f'Nonfinite target backbone at {residue.seqid}/{name}')
        for atom in residue:
            if not np.isfinite([atom.pos.x, atom.pos.y, atom.pos.z]).all():
                raise ValueError(f'Nonfinite target atom at {residue.seqid}/{atom.name}')
    provenance = dict(input_name=path.name,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_chain=chain.name, source_residue_ids=[str(r.seqid) for r in chain],
        sequence=sequence, length=len(sequence), template=True, msa=False,
        cleanup='Removed ligands/waters and alternative conformations; preserved protein coordinates')
    return structure, provenance


def validate_features(features, binder_length, target_sequence):
    expected_target = np.asarray([TOKENS.index(aa) for aa in target_sequence])
    total = binder_length + len(target_sequence)
    expected_asym = np.asarray([1] * binder_length + [2] * len(target_sequence))
    if np.asarray(features['aatype']).shape != (total,):
        raise ValueError('Feature length differs from binder plus target')
    if not np.array_equal(np.asarray(features['asym_id']), expected_asym):
        raise ValueError('Feature chain order must be binder then target')
    if not np.array_equal(np.asarray(features['aatype'])[binder_length:], expected_target):
        raise ValueError('Target sequence changed during feature preparation')
    if np.asarray(features['template_all_atom_mask'])[:, :binder_length].any():
        raise ValueError('The designed binder must not have a template')
    if target_sequence:
        backbone_mask = np.asarray(features['template_all_atom_mask'])[:, binder_length:, [0, 1, 2, 4]]
        if not np.all(backbone_mask == 1):
            raise ValueError('Target template has missing backbone atoms')
    test_pssm = jnp.full((binder_length, 20), 0.05)
    changed = set_binder_sequence(test_pssm, features, multimer=True)
    if not np.array_equal(np.asarray(changed['aatype'])[binder_length:], expected_target):
        raise ValueError('Sequence replacement changed the target')


def feature_digest(features):
    digest = hashlib.sha256()
    for name, value in sorted(features.items()):
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def loss_specification(has_target, mpnn_weight, mpnn_samples):
    """Serializable objective definition; monomer order matches the first demo."""
    spec = [dict(type='PLDDTLoss', weight=1.0),
        dict(type='WithinBinderPAE', weight=0.05),
        dict(type='WithinBinderContact', weight=1.0, num_contacts_per_residue=8,
             max_contact_distance=14.0, min_sequence_separation=8),
        dict(type='DistogramRadiusOfGyration', weight=0.1),
        dict(type='HelixLoss', weight=0.3)]
    if has_target:
        spec += [dict(type='BinderTargetContact', weight=1.0, contact_distance=8.0,
                      paratope_size=12, epitope_idx=None),
            dict(type='BinderTargetPAE', weight=0.05),
            dict(type='TargetBinderPAE', weight=0.05),
            dict(type='IPTMLoss', weight=0.25)]
    if mpnn_weight:
        spec.append(dict(type='InverseFoldingSequenceRecovery', weight=mpnn_weight,
            temp=0.1, num_samples=mpnn_samples, jacobi_iterations=10))
    return spec


def build_terms(spec, mpnn):
    classes = {'WithinBinderContact': WithinBinderContact,
               'BinderTargetContact': BinderTargetContact,
               'InverseFoldingSequenceRecovery': InverseFoldingSequenceRecovery}
    combined = None
    for item in spec:
        kwargs = {k: v for k, v in item.items() if k not in ('type', 'weight')}
        cls = classes[item['type']] if item['type'] in classes else getattr(sp, item['type'])
        if item['type'] == 'InverseFoldingSequenceRecovery':
            term = cls(mpnn, **(kwargs | {'temp': jnp.asarray(kwargs['temp'])}))
        else:
            term = cls(**kwargs)
        # Avoid multiplying unit-weight terms: this retains monomer arithmetic.
        weighted = term if item['weight'] == 1 else item['weight'] * term
        combined = weighted if combined is None else combined + weighted
    return combined


# Gradient-checkpointing policies. The AF2 Evoformer is wrapped in a bare
# hk.remat, which recomputes the whole forward during the backward pass and costs
# roughly one extra forward (measured backward multiplier 4.13x). hk.remat is an
# alias of jax.checkpoint and forwards `policy`, so expensive matmul outputs can be
# kept instead of recomputed -- a good trade on a machine with spare memory.
REMAT_POLICIES = {
    'default': None,
    'dots': 'dots_saveable',
    'dots_no_batch': 'dots_with_no_batch_dims_saveable',
    'everything': 'everything_saveable',
}


def apply_remat_policy(name):
    """Patch hk.remat to inject a checkpoint policy. Mathematically a no-op: it
    changes only which intermediates are stored versus recomputed."""
    if name == 'default':
        return
    policy = getattr(jax.checkpoint_policies, REMAT_POLICIES[name])
    original = hk.remat

    def remat_with_policy(fun, **kwargs):
        return original(fun, **{**kwargs, 'policy': policy})

    hk.remat = remat_with_policy


BINDCRAFT_STAGES = [
    dict(name='logits_1', n_steps=50, soft_start=0.0, soft_end=0.9,
         temp_start=1.0, temp_end=1.0, hard=False),
    dict(name='logits_2', n_steps=25, soft_start=0.9, soft_end=1.0,
         temp_start=1.0, temp_end=1.0, hard=False),
    dict(name='soft_anneal', n_steps=45, soft_start=1.0, soft_end=1.0,
         temp_start=1.0, temp_end=0.01, hard=False),
    dict(name='hard', n_steps=5, soft_start=1.0, soft_end=1.0,
         temp_start=0.01, temp_end=0.01, hard=True),
]


def optimizer_settings(args):
    """Single source of truth for what is actually handed to the optimizer.

    validate_protocol compares these values against the frozen protocol, so a
    code change that is not mirrored in the protocol file cannot pass silently.
    """
    if args.optimizer == 'simplex_apgm':
        return dict(algorithm='simplex_APGM', steps=args.steps,
                    stepsize=0.2, momentum=0.0)
    stages = BINDCRAFT_STAGES
    if getattr(args, 'smoke', 0):
        # Path-only smoke test: same stage structure, few steps. Never usable with
        # --protocol, so the frozen schedule cannot be shortened by accident.
        stages = [s | {'n_steps': args.smoke} for s in BINDCRAFT_STAGES]
    return dict(algorithm='mosaic.optimizers.bindcraft_design',
                implementation=('upstream 4 x mosaic.optimizers.colabdesign_stage '
                                '(ColabDesign/BindCraft schedule)'),
                steps=sum(s['n_steps'] for s in stages),
                lr=0.1, norm_seq_grad=True, stages=stages)


def validate_protocol(path, args, target, spec):
    """Check the frozen design settings without modifying screening thresholds."""
    protocol = json.loads(path.read_text())
    executed = optimizer_settings(args)
    if executed['steps'] != args.steps:
        raise ValueError(f"--steps {args.steps} differs from the {args.optimizer} "
                         f"schedule total {executed['steps']}")
    for key, value in executed.items():
        if key not in protocol['optimization']:
            raise ValueError(f'Frozen protocol is missing optimization.{key}')
        if protocol['optimization'][key] != value:
            raise ValueError(f'Frozen protocol optimization.{key}: expected '
                             f"{protocol['optimization'][key]}, executing {value}")
    # A protocol fixes one seed ('seed') or pre-registers a set ('seeds'). The set form
    # exists so a campaign declares every seed it will run BEFORE running any of them;
    # it is not a licence to keep drawing seeds until one passes.
    binder, redesign = protocol['binder'], protocol['redesign']
    if ('seed' in binder) == ('seeds' in binder):
        raise ValueError('Frozen protocol binder needs exactly one of seed or seeds')
    if 'seeds' in binder:
        if sorted(set(binder['seeds'])) != sorted(binder['seeds']):
            raise ValueError('Frozen protocol binder.seeds must be distinct')
        if args.seed not in binder['seeds']:
            raise ValueError(f"Frozen protocol pre-registered seeds {binder['seeds']}; "
                             f'got {args.seed}')
    # MPNN seeds are derived from the binder seed, so a seed set declares the rule.
    if ('seeds' in redesign) == ('seed_offsets' in redesign):
        raise ValueError('Frozen protocol redesign needs exactly one of seeds or seed_offsets')
    mpnn_seeds = ([args.seed + offset for offset in redesign['seed_offsets']]
                  if 'seed_offsets' in redesign else redesign['seeds'])
    required = [
        (args.length, binder['length'], 'binder length'),
        (target['length'], protocol['target']['length'], 'target length'),
        ('model_1_multimer_v3', protocol['model']['name'], 'model'),
        (1, protocol['model']['forward_passes'], 'forward passes'),
        ('float32', protocol['model']['precision'], 'precision'),
        (7, protocol['model']['objective_key'], 'objective seed'),
        ([args.seed + 100, args.seed + 101], mpnn_seeds, 'MPNN seeds'),
    ]
    if 'seed' in binder:
        required.append((args.seed, binder['seed'], 'seed'))
    for actual, expected, label in required:
        if actual != expected:
            raise ValueError(f'Frozen protocol {label}: expected {expected}, got {actual}')
    by_type = lambda values: {v['type']: v for v in values}
    if by_type(spec) != by_type(protocol['loss']):
        raise ValueError('Objective differs from the frozen protocol')
    return protocol


def normalize_structure(output, binder_sequence, target_sequence):
    expected = [binder_sequence] + ([target_sequence] if target_sequence else [])
    full_sequence = ''.join(TOKENS[t] for t in np.asarray(output.full_sequence).argmax(-1))
    if full_sequence != ''.join(expected):
        raise ValueError('Hard model output changed a candidate or target sequence')
    structure = output.to_structure()
    if len(structure) != 1 or len(structure[0]) != len(expected):
        raise ValueError('Hard model output has the wrong chain count')
    for idx, (chain, sequence) in enumerate(zip(structure[0], expected, strict=True)):
        if gemmi.one_letter_code([r.name for r in chain]) != sequence:
            raise ValueError('Exported chain order or sequence mismatch')
        chain.name = chr(ord('A') + idx)
        for number, residue in enumerate(chain, 1):
            residue.seqid = gemmi.SeqId(number, ' ')
    return structure


def save_hard_output(path, output, sequence):
    names = ('plddt', 'pae', 'pae_logits', 'pae_bins', 'asym_id', 'residue_idx',
             'full_sequence', 'backbone_coordinates', 'structure_coordinates',
             'atom37_coords', 'atom37_mask', 'distogram_logits', 'distogram_bins')
    arrays = {name: np.asarray(getattr(output, name)) for name in names}
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite hard model output: {name}')
    np.savez_compressed(path, sequence=np.asarray(sequence), **arrays)


def named_aux(aux):
    result = {}
    for item in aux if isinstance(aux, (list, tuple)) else [aux]:
        if isinstance(item, dict):
            result.update(plain(item))
        elif isinstance(item, (list, tuple)):
            result.update(named_aux(item))
    return result


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
    ap.add_argument('--target', type=Path, help='One complete protein target chain, used as a template')
    ap.add_argument('--protocol', type=Path, help='Optional frozen binder protocol JSON to verify and copy')
    ap.add_argument('--optimizer', choices=['simplex_apgm', 'bindcraft'], default='simplex_apgm',
                    help='simplex_apgm reproduces the v1 single soft stage; bindcraft runs the '
                         'upstream four-stage schedule whose final stage evaluates a one-hot')
    ap.add_argument('--remat-policy', choices=sorted(REMAT_POLICIES), default='default',
                    help='Which intermediates the Evoformer checkpoint keeps. default\n'
                         'recomputes everything (upstream behaviour); dots keeps matmul\n'
                         'outputs, trading memory for less recomputation.')
    ap.add_argument('--remat', choices=['on', 'off'], default='on',
                    help='Gradient checkpointing in the Evoformer. on (default) matches every\n'
                         'published run; off uses more activation memory for less recomputation.')
    ap.add_argument('--smoke', type=int, default=0, metavar='N',
                    help='Path-only smoke test: run N steps per bindcraft stage. Refused with '
                         '--protocol so the frozen schedule cannot be shortened by accident.')
    args = ap.parse_args()
    if args.smoke:
        if args.optimizer != 'bindcraft' or args.protocol:
            ap.error('--smoke requires --optimizer bindcraft and forbids --protocol')
        if args.smoke < 1:
            ap.error('--smoke must be positive')
        args.steps = 4 * args.smoke
    if args.length < 32 or args.steps < 1:
        ap.error('length >=32 and steps >=1 required')
    if not np.isfinite(args.mpnn_weight) or args.mpnn_weight < 0 or args.mpnn_samples < 1:
        ap.error('mpnn-weight must be finite/nonnegative and mpnn-samples positive')
    if args.protocol and not args.target:
        ap.error('--protocol requires --target')
    target_structure, target = load_target(args.target) if args.target else (None, None)
    target_sequence = target['sequence'] if target else ''
    spec = loss_specification(bool(target), args.mpnn_weight, args.mpnn_samples)
    protocol = validate_protocol(args.protocol, args, target, spec) if args.protocol else None
    args.output.mkdir(parents=True, exist_ok=False)
    if protocol is not None:
        (args.output / 'protocol.json').write_bytes(args.protocol.read_bytes())
    if target:
        target_structure.make_mmcif_document().write_file(str(args.output / 'target-input.cif'))
    start = time.perf_counter()
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != args.platform:
        raise RuntimeError(f'Unexpected devices: {devices}')
    print('DEVICES', devices, flush=True)
    np.random.seed(args.seed)
    apply_remat_policy(args.remat_policy)
    model = SingleAF2(args.weights, use_remat=args.remat == 'on')
    chains = [TargetChain(target_sequence, use_msa=False,
                         template_chain=target_structure[0][0])] if target else []
    features, _ = model.binder_features(args.length, chains=chains)
    validate_features(features, args.length, target_sequence)
    features = jax.tree.map(jnp.asarray, features)
    mpnn = None
    if args.mpnn_weight:
        mpnn = ProteinMPNN.from_pretrained(backbone_noise=0.0)
    terms = build_terms(spec, mpnn)
    loss = DesignLoss(model, features, terms)
    settings = optimizer_settings(args)
    stages = settings.get('stages', [])
    # NumPy initialization ensures CPU and GPU start from exactly the same values.
    if args.optimizer == 'simplex_apgm':
        logits = np.random.default_rng(args.seed).gumbel(size=(args.length, 20)).astype(np.float32) * 0.5
        x0 = jnp.asarray(np.exp(logits) / np.exp(logits).sum(-1, keepdims=True))
        x = x0
    else:
        # BindCraft's documented initialization. The scale matters: at the schedule's
        # temp 0.01 the softmax saturates, and logit std above ~0.5 silently zeroes the
        # gradient on a growing fraction of residues (measured: 8/48 rows at std 0.63,
        # 44/48 at std 16), which would make the last two stages no-ops.
        z0 = 0.01 * np.random.default_rng(args.seed).standard_normal((args.length, 20))
        z0 = (z0 - z0.mean(-1, keepdims=True)).astype(np.float32)
        x0 = jnp.asarray(z0)
        # Probe at the first pseudo-sequence the schedule actually evaluates, so the
        # CPU/MPS parity gate covers the stage-1 regime rather than a simplex point.
        # Computed in host NumPy, not on device: the gate compares two gradients at
        # one input, so the input itself must be bit-identical on both platforms.
        first = stages[0]
        soft0 = np.float32(first['soft_start']
                           + (first['soft_end'] - first['soft_start']) / first['n_steps'])
        scaled = z0 / np.float32(first['temp_start'])
        exponential = np.exp(scaled - scaled.max(-1, keepdims=True))
        soft_sequence = exponential / exponential.sum(-1, keepdims=True)
        probe = (soft0 * soft_sequence + (np.float32(1.0) - soft0) * z0).astype(np.float32)
        x = jnp.asarray(probe)
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
        target=target, total_length=args.length + len(target_sequence),
        chain_order=['binder_A', 'target_B'] if target else ['binder_A'],
        feature_sha256=feature_digest(features), loss_terms=spec,
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest() if args.protocol else None,
        contact_loss_implementation='metal_losses.py: equivalent contact reduction',
        model='model_1_multimer_v3', precision='float32', forward_passes=1,
        initial_loss=float(initial), initial_aux=plain(initial_aux),
        gradient_devices=[str(d) for d in gradient.devices()],
        mpnn_weight=args.mpnn_weight, mpnn_samples=args.mpnn_samples,
        compile_and_first_gradient_seconds=compile_first_s, warm_gradient_seconds=warm_s,
        # Reported at probe time so a target's feasibility on this machine is known
        # before committing to a full optimization.
        remat=args.remat, remat_policy=args.remat_policy, probe_device_memory_stats=devices[0].memory_stats(),
        probe_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        versions={n: importlib.metadata.version(n) for n in ['jax','jaxlib','jax-mps','equinox']},
        scope=('Small templated-target binder design; computational candidates only, no binding or experimental validation.'
               if target else 'Small de novo monomer, full AF2 confidence/structure gradients; not binder design or experimental validation.'))
    (args.output / 'probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print('FULL_GRADIENT_OK', json.dumps(report), flush=True)
    if args.probe_only:
        return
    history = []
    apgm = args.optimizer == 'simplex_apgm'
    best = {'value': float(initial), 'x': np.asarray(x).copy()} if apgm else {'value': np.inf, 'x': None}
    pending = {}
    # Replace only the process-local optimizer evaluation helper. Preserve raw
    # errors rather than upstream's nan_to_num; optimizer math is unchanged.
    def checked_eval(loss_function, x, key):
        (value, aux), g = jax.block_until_ready(raw_vg(jnp.asarray(x, dtype=jnp.float32), key=key))
        check(value, g)
        if apgm:
            if float(value) < best['value']:
                best.update(value=float(value), x=np.asarray(x).copy())
            history.append({'loss':float(value), 'aux':plain(aux)})
            (args.output/'trajectory.json').write_text(json.dumps(history, indent=2)+'\n')
        else:
            # Stash for the trajectory_fn, which alone knows the stage settings.
            # check() already brought the gradient to the host, so counting saturated
            # rows here adds no extra synchronization.
            pending.update(value=float(value), aux=plain(aux), x=np.asarray(x).copy(),
                           zero_grad_rows=int((np.abs(np.asarray(g)).sum(-1) == 0).sum()))
        return (value, aux), g - g.mean(-1, keepdims=True)

    # bindcraft_design reports (soft, temp, hard) per evaluation but not the stage,
    # so synthesize stage indices from a counter and the known schedule lengths.
    bounds, total = [], 0
    for index, stage in enumerate(stages):
        total += stage['n_steps']
        bounds.append((total, index, stage['name']))

    def record(aux, z):
        step = len(history)
        stage_index, stage_name, stage_start = 0, (stages[0]['name'] if stages else ''), 0
        for end, index, name in bounds:
            if step < end:
                stage_index, stage_name = index, name
                stage_start = end - stages[index]['n_steps']
                break
        if not pending or pending['value'] != aux['loss']:
            raise ValueError('Trajectory entry does not match the evaluated iterate')
        entry = {'loss': pending['value'], 'aux': pending['aux'], 'step': step,
                 'stage': stage_index, 'stage_name': stage_name, 'step_in_stage': step - stage_start,
                 'soft': float(aux['soft']), 'temp': float(aux['temp']), 'hard': float(aux['hard']),
                 'nnz': float(aux['nnz']), 'zero_grad_rows': pending['zero_grad_rows']}
        # Select only among hard-stage evaluations: those alone are true one-hot
        # sequences, so their objectives are comparable and the exported sequence
        # is exactly the sequence that was evaluated.
        if entry['hard'] == 1.0 and entry['loss'] < best['value']:
            best.update(value=entry['loss'], x=pending['x'].copy())
        history.append(entry)
        (args.output/'trajectory.json').write_text(json.dumps(history, indent=2)+'\n')
        pending.clear()
        return entry['loss']

    t = time.perf_counter()
    with patch.object(optimizers, '_eval_loss_and_grad', checked_eval):
        if apgm:
            final_x, _ = optimizers.simplex_APGM(loss_function=loss, x=x0, n_steps=args.steps,
                stepsize=0.2, momentum=0.0, key=key)
            checked_eval(loss, final_x, key)
            final_pssm = np.asarray(final_x)
        else:
            pssm, _ = optimizers.bindcraft_design(loss_function=loss, x=x0, lr=settings['lr'],
                logits_iters=(stages[0]['n_steps'], stages[1]['n_steps']),
                soft_iters=stages[2]['n_steps'], hard_iters=stages[3]['n_steps'],
                key=key, trajectory_fn=record)
            final_pssm = np.asarray(pssm)
    report['optimization_seconds'] = time.perf_counter()-t
    report['steps'] = args.steps
    # Name the selected objective for what it is. Under simplex_apgm it is a SOFT
    # objective and the argmax export changes it; under bindcraft the selected
    # iterate is a one-hot, so this is already the hard-sequence objective.
    if apgm:
        report['best_soft_loss'] = best['value']
    else:
        report['best_hard_stage_loss'] = best['value']
    if best['x'] is None:
        raise ValueError('No hard-stage evaluation was recorded; nothing to export')
    x_best = jnp.asarray(best['x'])
    if apgm and (not np.allclose(np.asarray(x_best).sum(-1), 1, atol=1e-5) or np.asarray(x_best).min() < -1e-6):
        raise ValueError('Invalid simplex')
    if not apgm:
        # The selected hard-stage iterate must be an exact one-hot.
        selected = np.asarray(x_best)
        if not np.array_equal(selected, np.eye(20, dtype=selected.dtype)[selected.argmax(-1)]):
            raise ValueError('Selected hard-stage iterate is not one-hot')
        if not np.allclose(final_pssm.sum(-1), 1, atol=1e-5) or final_pssm.min() < -1e-6:
            raise ValueError('Invalid simplex')
        report['bindcraft'] = dict(stages=stages, lr=settings['lr'],
            hard_stage_losses=[e['loss'] for e in history if e['hard'] == 1.0],
            zero_grad_rows_by_stage=[max((e['zero_grad_rows'] for e in history
                if e['stage'] == i), default=0) for i in range(len(stages))],
            final_pssm_temp1_mean_max_probability=float(final_pssm.max(-1).mean()))
        np.savez(args.output/'final-pssm.npz', probabilities=final_pssm)
    np.savez(args.output/'optimized.npz', probabilities=best['x'])
    tokens = np.asarray(x_best).argmax(-1)
    sequence = ''.join(TOKENS[i] for i in tokens)
    hard = jax.nn.one_hot(tokens, 20)
    print('PREDICTING_HARD_CANDIDATE', sequence, flush=True)
    output = jax.block_until_ready(model.model_output(PSSM=hard, features=features,
        recycling_steps=1, model_idx=0, key=key))
    if not np.isfinite(np.asarray(output.backbone_coordinates)).all():
        raise ValueError('Nonfinite exported backbone')
    structure = normalize_structure(output, sequence, target_sequence)
    structure.write_pdb(str(args.output/'hallucinated.pdb'))
    save_hard_output(args.output/'hard-output.npz', output, sequence + target_sequence)
    hard_value, hard_aux = jax.block_until_ready(eqx.filter_jit(
        lambda x, out: terms(x, output=out, key=jax.random.key(7)))(hard, output))
    diagnostics = dict(loss=float(hard_value), terms=named_aux(hard_aux),
        binder_plddt_0_to_100=float(output.plddt[:args.length].mean()) * 100)
    if target:
        diagnostics.update(target_plddt_0_to_100=float(output.plddt[args.length:].mean()) * 100,
            binder_to_target_pae_mean=float(output.pae[:args.length, args.length:].mean()),
            target_to_binder_pae_mean=float(output.pae[args.length:, :args.length].mean()),
            iptm=float(diagnostics['terms']['iptm']))
    if not np.isfinite(float(hard_value)):
        raise ValueError('Nonfinite hard-sequence objective')
    report['hard_diagnostics'] = diagnostics
    candidates = [{'name':'hallucinated', 'sequence':sequence,
                   'design_plddt':float(output.plddt[:args.length].mean())}]
    # Standard MPNN sequence redesign on the newly hallucinated backbone.
    if mpnn is None:
        mpnn = ProteinMPNN.from_pretrained(backbone_noise=0.0)
    redesign = eqx.filter_jit(lambda k: inverse_fold(mpnn, args.length, output,
        temp=0.1, key=k, jacobi_iterations=10))
    for i in range(2):
        redesigned = np.asarray(jax.block_until_ready(redesign(jax.random.key(args.seed+i+100))))
        if redesigned.shape != (args.length,) or np.any((redesigned < 0) | (redesigned >= 20)):
            raise ValueError('MPNN returned an invalid binder sequence')
        candidates.append({'name':f'mpnn_{i+1}', 'sequence':''.join(TOKENS[v] for v in redesigned)})
    # MPNN receives the full complex but updates/returns binder residues only.
    # Recheck the source output and write the unchanged manifest target explicitly.
    normalize_structure(output, sequence, target_sequence)
    for item in candidates:
        (args.output/f"{item['name']}.fasta").write_text(f">{item['name']} computational_candidate\n{item['sequence']}\n")
        yaml = 'version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: '+item['sequence']+'\n      msa: empty\n'
        if target:
            item['target_sequence'] = target_sequence
            yaml += '  - protein:\n      id: B\n      sequence: '+target_sequence+'\n      msa: empty\n'
            (args.output/f"{item['name']}-complex.fasta").write_text(
                f">{item['name']}_binder_A\n{item['sequence']}\n>target_B\n{target_sequence}\n")
        (args.output/f"{item['name']}.yaml").write_text(yaml)
    report.update(candidates=candidates, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        total_seconds=time.perf_counter()-start, device_memory_stats=devices[0].memory_stats())
    for name, value in report.items():
        try:
            json.dumps({name: value})
        except TypeError as error:
            raise TypeError(f'summary.json key {name!r} is not serializable: {error}') from error
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print('DESIGN_COMPLETE', json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
