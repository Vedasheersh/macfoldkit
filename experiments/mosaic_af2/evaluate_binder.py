"""Evaluate the frozen ubiquitin binder screen; no inference or GPU required.

Usage: python evaluate_binder.py DESIGN REFOLDS OUTPUT --protocol protocol.json
       --target ubiquitin-target.cif
REFOLDS has one candidate-named directory per design; prediction.json files may
be nested arbitrarily beneath it. Every expected candidate and seed is reported,
including malformed or missing predictions. Requires NumPy and Gemmi.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path

import gemmi
import numpy as np

# Canonical heavy atoms; terminal OXT is optional. Missing side-chain atoms must
# not silently inflate the apparent quality of the clash/contact screen.
SIDECHAINS = {
    'A': 'CB', 'R': 'CB CG CD NE CZ NH1 NH2', 'N': 'CB CG OD1 ND2',
    'D': 'CB CG OD1 OD2', 'C': 'CB SG', 'Q': 'CB CG CD OE1 NE2',
    'E': 'CB CG CD OE1 OE2', 'G': '', 'H': 'CB CG ND1 CD2 CE1 NE2',
    'I': 'CB CG1 CG2 CD1', 'L': 'CB CG CD1 CD2', 'K': 'CB CG CD CE NZ',
    'M': 'CB CG SD CE', 'F': 'CB CG CD1 CD2 CE1 CE2 CZ', 'P': 'CB CG CD',
    'S': 'CB OG', 'T': 'CB OG1 CG2', 'W': 'CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2',
    'Y': 'CB CG CD1 CD2 CE1 CE2 CZ OH', 'V': 'CB CG1 CG2',
}


@dataclass
class Chain:
    sequence: str
    ca: np.ndarray
    heavy: np.ndarray
    atom_residue: np.ndarray
    ca_plddt: np.ndarray


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_chains(path, expected=None):
    """Strict protein reader: exact chains, sequence, numbering and heavy atoms."""
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError('Expected exactly one structural model')
    names = [c.name for c in structure[0]]
    if len(names) != len(set(names)) or (expected is not None and set(names) != set(expected)):
        raise ValueError(f'Unexpected or repeated chains: {names}')
    result = {}
    for chain in structure[0]:
        residues = list(chain)
        sequence = gemmi.one_letter_code([r.name for r in residues])
        if not sequence or set(sequence) - SIDECHAINS.keys():
            raise ValueError(f'Chain {chain.name} has noncanonical or missing sequence')
        if expected is not None and sequence != expected[chain.name]:
            raise ValueError(f'Chain {chain.name} sequence mismatch')
        ca, heavy, indices, confidence = [], [], [], []
        for i, (residue, aa) in enumerate(zip(residues, sequence)):
            if residue.seqid.num != i + 1 or residue.seqid.icode.strip():
                raise ValueError(f'Chain {chain.name} has missing/reordered/inserted residue numbers')
            atoms = {}
            for atom in residue:
                if atom.element.is_hydrogen:
                    continue
                if atom.name in atoms or atom.altloc not in ('\x00', ' ', '.'):
                    raise ValueError(f'Duplicate or alternative atom {chain.name}:{i+1}:{atom.name}')
                coords = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
                if (not np.isfinite(coords).all() or not np.isfinite(atom.b_iso)
                        or not np.isfinite(atom.occ) or atom.occ <= 0):
                    raise ValueError(f'Missing/nonfinite atom {chain.name}:{i+1}:{atom.name}')
                atoms[atom.name] = atom
                heavy.append(coords)
                indices.append(i)
            required = set(('N CA C O ' + SIDECHAINS[aa]).split())
            if not required.issubset(atoms) or set(atoms) - required - {'OXT'}:
                raise ValueError(f'Missing/unexpected heavy atoms {chain.name}:{i+1}: '
                                 f'missing={sorted(required-set(atoms))}, extra={sorted(set(atoms)-required-{ "OXT" })}')
            atom = atoms['CA']
            ca.append([atom.pos.x, atom.pos.y, atom.pos.z])
            confidence.append(atom.b_iso)
        result[chain.name] = Chain(sequence, np.asarray(ca), np.asarray(heavy),
                                   np.asarray(indices), np.asarray(confidence))
    return result


def proper_transform(reference, moving):
    """Return row-vector Kabsch R,t mapping moving to reference (no reflection)."""
    reference, moving = np.asarray(reference), np.asarray(moving)
    if reference.shape != moving.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError('Coordinate shapes differ or are not N×3')
    if len(reference) < 3 or not np.isfinite(reference).all() or not np.isfinite(moving).all():
        raise ValueError('Insufficient or nonfinite alignment coordinates')
    a, b = reference - reference.mean(0), moving - moving.mean(0)
    if np.linalg.matrix_rank(a, tol=1e-8) < 2 or np.linalg.matrix_rank(b, tol=1e-8) < 2:
        raise ValueError('Degenerate alignment coordinates')
    u, _, vt = np.linalg.svd(b.T @ a)
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    return rotation, reference.mean(0) - moving.mean(0) @ rotation


def coordinate_rmsd(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid RMSD coordinates')
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def aligned_rmsd(reference, moving):
    rotation, translation = proper_transform(reference, moving)
    return coordinate_rmsd(reference, moving @ rotation + translation)


def pose_metrics(reference, moving, original_target):
    rotation, translation = proper_transform(reference['B'].ca, moving['B'].ca)
    return {
        'binder_self_aligned_ca_rmsd_angstrom': aligned_rmsd(reference['A'].ca, moving['A'].ca),
        'target_aligned_binder_ca_rmsd_angstrom': coordinate_rmsd(
            reference['A'].ca, moving['A'].ca @ rotation + translation),
        'refold_target_ca_rmsd_to_input_angstrom': aligned_rmsd(original_target.ca, moving['B'].ca),
    }


def interface_metrics(binder, target, contact_cutoff=5., clash_cutoff=1.5):
    delta = binder.heavy[:, None, :] - target.heavy[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    if not distances.size or not np.isfinite(distances).all():
        raise ValueError('Invalid interface atom coordinates')
    bi, ti = np.where(distances < contact_cutoff)
    pairs = set(zip(binder.atom_residue[bi].tolist(), target.atom_residue[ti].tolist()))
    return {
        'distinct_contacting_binder_residues': len({b for b, _ in pairs}),
        'distinct_contacting_target_residues': len({t for _, t in pairs}),
        'distinct_contacting_residue_pairs': len(pairs),
        'interchain_heavy_atom_contacts': int(len(bi)),
        'interchain_heavy_atom_clashes': int(np.sum(distances < clash_cutoff)),
        'interchain_minimum_heavy_atom_distance_angstrom': float(distances.min()),
        'contacting_residue_pairs_1_based': [[b+1, t+1] for b, t in sorted(pairs)],
    }


def sequence_metrics(sequence):
    if not sequence or set(sequence) - SIDECHAINS.keys():
        raise ValueError('Missing/noncanonical candidate sequence')
    counts = Counter(sequence)
    fractions = np.array(list(counts.values())) / len(sequence)
    return {
        'sequence_entropy_bits': float(-np.sum(fractions * np.log2(fractions))),
        'max_amino_acid_fraction': float(fractions.max()),
        'maximum_homopolymer_run': max(len(list(g)) for _, g in itertools.groupby(sequence)),
        'composition_counts': dict(sorted(counts.items())),
    }


SCREEN_FIELDS = {
    'binder_plddt_0_to_100': ('binder_plddt_min_0_to_100', 'min'),
    'complex_iptm': ('complex_iptm_min', 'min'),
    'binder_self_aligned_ca_rmsd_angstrom': ('binder_self_aligned_ca_rmsd_max_angstrom', 'max'),
    'target_aligned_binder_ca_rmsd_angstrom': ('target_aligned_binder_ca_rmsd_max_angstrom', 'max'),
    'refold_target_ca_rmsd_to_input_angstrom': ('refold_target_ca_rmsd_to_input_max_angstrom', 'max'),
    'distinct_contacting_binder_residues': ('distinct_contacting_binder_residues_min', 'min'),
    'distinct_contacting_target_residues': ('distinct_contacting_target_residues_min', 'min'),
    'distinct_contacting_residue_pairs': ('distinct_contacting_residue_pairs_min', 'min'),
    'interchain_minimum_heavy_atom_distance_angstrom': ('interchain_minimum_heavy_atom_distance_min_angstrom', 'min'),
    'sequence_entropy_bits': ('sequence_shannon_entropy_min_bits', 'min'),
    'max_amino_acid_fraction': ('maximum_single_amino_acid_fraction', 'max'),
    'maximum_homopolymer_run': ('maximum_homopolymer_run', 'max'),
}


def screen_metrics(metrics, screen):
    checks = {}
    for metric, (key, direction) in SCREEN_FIELDS.items():
        value = metrics.get(metric)
        finite = isinstance(value, (int, float)) and np.isfinite(value)
        passed = bool(finite and (value >= screen[key] if direction == 'min' else value <= screen[key]))
        checks[metric] = {'value': value if finite else None, 'threshold': screen[key],
                          'comparison': '>=' if direction == 'min' else '<=', 'passed': passed}
    return checks


def safe_number(metadata, key, low=None, high=None):
    value = metadata.get(key)
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'Missing/nonfinite {key}')
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f'Out-of-range {key}: {value}')
    return float(value)


def evaluate(design_dir, refolds_dir, protocol_path, target_path):
    design = json.loads((design_dir / 'summary.json').read_text())
    protocol = json.loads(protocol_path.read_text())
    protocol_digest, target_digest = sha256(protocol_path), sha256(target_path)
    screen, folding = protocol['screen'], protocol['independent_refold']
    seeds, names = folding['seeds'], protocol['redesign']['candidates']
    if len(seeds) != 2 or len(set(seeds)) != 2 or not screen['both_boltz_seeds_must_pass']:
        raise ValueError('Expected frozen two-distinct-seed rule')
    # Validate every threshold before reading outcomes.
    for key, _ in SCREEN_FIELDS.values():
        safe_number(screen, key)
    original = read_chains(target_path)
    target_chain = protocol['target']['chain']
    if set(original) != {target_chain}:
        raise ValueError('Original target must contain exactly the protocol target chain')
    original = original[target_chain]
    if len(original.sequence) != protocol['target']['length']:
        raise ValueError('Original target length differs from protocol')
    candidates = design.get('candidates', [])
    reference, global_errors = None, []
    if design.get('protocol_sha256') != protocol_digest:
        global_errors.append('Design protocol_sha256 is missing or differs from evaluated protocol')
    recorded_target = design.get('target')
    if not isinstance(recorded_target, dict) or recorded_target.get('input_sha256') != target_digest:
        global_errors.append('Design target input_sha256 is missing or differs from evaluated target')
    if sorted(c.get('name', '') for c in candidates) != sorted(names):
        global_errors.append('Candidate names/count differ from frozen protocol')
    by_name = {c.get('name'): c for c in candidates}
    try:
        hall_sequence = by_name['hallucinated']['sequence']
        if len(hall_sequence) != protocol['binder']['length']:
            raise ValueError('Hallucinated binder length differs from protocol')
        reference = read_chains(design_dir / 'hallucinated.pdb', {'A': hall_sequence, 'B': original.sequence})
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        global_errors.append(f'AF2 reference: {error}')
    af2 = {}
    if reference is not None:
        try:
            af2.update(target_ca_rmsd_to_input_angstrom=aligned_rmsd(original.ca, reference['B'].ca),
                       binder_plddt_0_to_100=float(reference['A'].ca_plddt.mean()),
                       binder_ca_radius_of_gyration_angstrom=float(np.sqrt(np.mean(np.sum(
                           (reference['A'].ca-reference['A'].ca.mean(0))**2, axis=-1)))))
            af2['interface'] = interface_metrics(reference['A'], reference['B'],
                screen['interchain_heavy_atom_contact_cutoff_angstrom'],
                screen['interchain_minimum_heavy_atom_distance_min_angstrom'])
        except ValueError as error:
            global_errors.append(f'AF2 geometry: {error}')
    try:
        with np.load(design_dir / 'hard-output.npz', allow_pickle=False) as arrays:
            n, length = protocol['binder']['length'], protocol['target']['length']
            if str(arrays['sequence']) != by_name['hallucinated']['sequence'] + original.sequence:
                raise ValueError('Hard-output sequence mismatch')
            if not np.array_equal(arrays['asym_id'], [1]*n + [2]*length):
                raise ValueError('Hard-output chain order mismatch')
            pae, plddt = arrays['pae'], arrays['plddt']
            if pae.shape != (n+length, n+length) or plddt.shape != (n+length,):
                raise ValueError('Hard-output confidence shapes mismatch')
            if (not np.isfinite(pae).all() or not np.isfinite(plddt).all()
                    or np.any(pae < 0) or np.any((plddt < 0) | (plddt > 1))):
                raise ValueError('Nonfinite or out-of-range hard-output confidence')
            if reference is not None:
                pdb_plddt = np.concatenate([reference['A'].ca_plddt, reference['B'].ca_plddt])
                # PDB stores two decimal places; allow rounding plus FP32 noise.
                if not np.allclose(pdb_plddt, plddt * 100, rtol=0, atol=0.0051):
                    raise ValueError('Hard-output pLDDT differs from rounded AF2 PDB confidence')
            af2.update(binder_to_target_pae_mean=float(pae[:n, n:].mean()),
                       target_to_binder_pae_mean=float(pae[n:, :n].mean()),
                       binder_raw_plddt_0_to_100=float(plddt[:n].mean()*100))
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        global_errors.append(f'AF2 confidence arrays: {error}')
    af2['hard_sequence_metrics'] = design.get('hard_diagnostics', {})
    rows = []
    for name in names:
        item = by_name.get(name, {})
        sequence = item.get('sequence', '')
        candidate_errors, seq_metrics = [], {}
        try:
            seq_metrics = sequence_metrics(sequence)
            if len(sequence) != protocol['binder']['length']:
                raise ValueError('Binder length differs from protocol')
            if item.get('target_sequence', original.sequence) != original.sequence:
                raise ValueError('Candidate target sequence mismatch')
        except ValueError as error:
            candidate_errors.append(str(error))
        found, file_errors = {}, []
        for path in sorted((refolds_dir / name).rglob('prediction.json')):
            try:
                metadata = json.loads(path.read_text())
                seed = metadata['seed']
                if seed not in seeds:
                    raise ValueError(f'Unexpected refold seed {seed}')
                found.setdefault(seed, []).append((path, metadata))
            except (ValueError, OSError, KeyError, TypeError) as error:
                file_errors.append(f'{path}: {error}')
        trials = []
        for seed in seeds:
            metrics = dict(seq_metrics)
            errors = list(global_errors) + candidate_errors + file_errors
            trial = {'seed': seed, 'metrics': metrics, 'errors': errors}
            files = found.get(seed, [])
            if len(files) != 1:
                errors.append(f'Expected exactly one prediction for seed {seed}; found {len(files)}')
            else:
                path, metadata = files[0]
                trial.update(prediction_json=str(path), prediction_sha256=sha256(path),
                             provenance={k: metadata.get(k) for k in (
                                 'model', 'revision', 'device', 'profile', 'precision', 'seed',
                                 'feature_sha256', 'input_sha256', 'inference_seconds', 'load_seconds',
                                 'total_model_seconds', 'pair_chains_iptm', 'ptm', 'msa_rows_used')})
                try:
                    for key, expected in [('model', folding['model']), ('revision', folding['revision']),
                        ('profile', folding['profile']), ('precision', folding['pair_precision'])]:
                        if metadata.get(key) != expected:
                            raise ValueError(f'Refold {key} differs from protocol')
                    chains = metadata['chains']
                    if [(c['id'], c['sequence']) for c in chains] != [('A', sequence), ('B', original.sequence)]:
                        raise ValueError('Refold metadata chains/sequence mismatch')
                except (ValueError, KeyError, TypeError) as error:
                    errors.append(str(error))
                try:
                    metrics['complex_iptm'] = safe_number(metadata, 'iptm', 0, 1)
                except ValueError as error:
                    errors.append(str(error))
                try:
                    # FastPLMs cif_writer defaults B-factors to100 when pLDDT is
                    # absent. Require raw pLDDT presence to reject that fallback.
                    safe_number(metadata, 'plddt_mean', 0, 1)
                    confidence_present = True
                except ValueError as error:
                    confidence_present = False
                    errors.append(str(error))
                try:
                    cif = path.with_suffix('.cif')
                    moving = read_chains(cif, {'A': sequence, 'B': original.sequence})
                    trial['prediction_cif_sha256'] = sha256(cif)
                    confidence = moving['A'].ca_plddt
                    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 100)):
                        raise ValueError('Invalid binder CIF pLDDT')
                    if confidence_present:
                        metrics['binder_plddt_0_to_100'] = float(confidence.mean())
                    metrics.update(interface_metrics(moving['A'], moving['B'],
                        screen['interchain_heavy_atom_contact_cutoff_angstrom'],
                        screen['interchain_minimum_heavy_atom_distance_min_angstrom']))
                    metrics['refold_target_ca_rmsd_to_input_angstrom'] = aligned_rmsd(original.ca, moving['B'].ca)
                    if reference is not None:
                        metrics.update(pose_metrics(reference, moving, original))
                except (ValueError, RuntimeError, OSError) as error:
                    errors.append(f'Refold structure: {error}')
            trial['checks'] = screen_metrics(metrics, screen)
            trial['failed_criteria'] = [key for key, check in trial['checks'].items() if not check['passed']]
            trial['passes_screen'] = not errors and not trial['failed_criteria']
            trials.append(trial)
        both_metrics = set(trials[0]['metrics']) & set(trials[1]['metrics'])
        seed_differences = {key: float(trials[1]['metrics'][key] - trials[0]['metrics'][key])
                            for key in sorted(both_metrics)
                            if isinstance(trials[0]['metrics'][key], (int, float))
                            and isinstance(trials[1]['metrics'][key], (int, float))}
        rows.append({'name': name, 'sequence': sequence, 'sequence_metrics': seq_metrics,
                     'trials': trials, 'seed_metric_differences_second_minus_first': seed_differences,
                     'passing_seeds': sum(t['passes_screen'] for t in trials),
                     'passes_both_seeds': all(t['passes_screen'] for t in trials)})
    count = sum(r['passes_both_seeds'] for r in rows)
    return json_safe({'scope': 'Computational self-consistency screen only; no experimental evidence of binding.',
            'protocol': protocol, 'protocol_sha256': protocol_digest,
            'original_target_sha256': target_digest,
            'design_summary_sha256': sha256(design_dir / 'summary.json'),
            'alignment': 'C-alpha by residue position; proper Kabsch target-only transform applied to binder.',
            'plddt_source': 'Per-residue CA B-factors in FastPLMs CIF (0–100); raw pLDDT presence required.',
            'af2_diagnostics': af2, 'global_errors': global_errors, 'candidates': rows,
            'passing_candidates': count,
            'interpretation': ('Computationally consistent candidate(s), pending experimental validation.' if count else
                               'No candidate met the predefined computational screen.')})


def json_safe(value):
    """Keep invalid optional diagnostic values visible as null in valid JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('design', type=Path)
    parser.add_argument('refolds', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.design, args.refolds, args.protocol, args.target)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'passing_candidates': report['passing_candidates'],
                      'global_errors': report['global_errors'],
                      'candidates': [{'name': c['name'], 'passing_seeds': c['passing_seeds']} for c in report['candidates']]}, indent=2))


if __name__ == '__main__':
    main()
