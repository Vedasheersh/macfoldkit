"""CPU regression tests for binder pose evaluation (NumPy + Gemmi only)."""
import json
from pathlib import Path
import tempfile
import unittest

import gemmi
import numpy as np

from evaluate_binder import (Chain, SIDECHAINS, aligned_rmsd, evaluate, interface_metrics,
                             pose_metrics, proper_transform, read_chains, screen_metrics,
                             sequence_metrics, sha256)


BACKBONE = np.array([[0., 0., 0.], [3., 0., 0.], [2., 3., 0.], [1., 2., 3.]])


def chain_at(ca, sequence='AGSV'):
    return Chain(sequence, ca.copy(), ca.copy(), np.arange(len(ca)), np.full(len(ca), 95.))


def write_structure(path, chains):
    structure, model = gemmi.Structure(), gemmi.Model('1')
    names = {'A':'ALA', 'G':'GLY', 'S':'SER', 'V':'VAL'}
    for name, sequence, positions in chains:
        chain = gemmi.Chain(name)
        for i, (aa, position) in enumerate(zip(sequence, positions)):
            residue = gemmi.Residue()
            residue.name, residue.seqid = names[aa], gemmi.SeqId(i+1, ' ')
            for j, atom_name in enumerate(('N CA C O '+SIDECHAINS[aa]).split()):
                atom = gemmi.Atom()
                atom.name, atom.element = atom_name, gemmi.Element(atom_name[0])
                atom.pos = gemmi.Position(*(position + [0., .12*j, .08*j]))
                atom.occ, atom.b_iso = 1., 95.
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    if path.suffix == '.pdb':
        structure.write_pdb(str(path))
    else:
        structure.make_mmcif_document().write_file(str(path))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.reference = {'A': chain_at(BACKBONE+[0., 0., 5.]), 'B': chain_at(BACKBONE)}
        self.rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.translation = np.array([11., -23., 7.])

    def test_rigid_transform_target_alignment_preserves_binder_pose(self):
        moving = {key: chain_at(chain.ca @ self.rotation + self.translation)
                  for key, chain in self.reference.items()}
        metrics = pose_metrics(self.reference, moving, self.reference['B'])
        for value in metrics.values():
            self.assertLess(value, 1e-12)
        r, _ = proper_transform(self.reference['B'].ca, moving['B'].ca)
        self.assertAlmostEqual(np.linalg.det(r), 1.)

    def test_shifted_binder_passes_fold_but_fails_pose(self):
        moving = {'A': chain_at(self.reference['A'].ca+[10., 0., 0.]), 'B': self.reference['B']}
        metrics = pose_metrics(self.reference, moving, self.reference['B'])
        self.assertLess(metrics['binder_self_aligned_ca_rmsd_angstrom'], 1e-12)
        self.assertAlmostEqual(metrics['target_aligned_binder_ca_rmsd_angstrom'], 10.)
        self.assertGreater(metrics['target_aligned_binder_ca_rmsd_angstrom'], 3.)

    def test_reflection_is_not_allowed_as_rotation(self):
        mirrored = BACKBONE * [-1., 1., 1.]
        self.assertGreater(aligned_rmsd(BACKBONE, mirrored), .1)

    def test_contacts_count_distinct_residues_and_pairs(self):
        a = Chain('AA', np.zeros((2, 3)), np.array([[0.,0.,0.],[0.,.1,0.],[10.,0.,0.]]),
                  np.array([0, 0, 1]), np.ones(2))
        b = Chain('AA', np.zeros((2, 3)), np.array([[0.,0.,2.],[0.,.1,2.],[30.,0.,0.]]),
                  np.array([0, 0, 1]), np.ones(2))
        result = interface_metrics(a, b)
        self.assertEqual(result['interchain_heavy_atom_contacts'], 4)
        self.assertEqual(result['distinct_contacting_residue_pairs'], 1)
        self.assertEqual(result['distinct_contacting_binder_residues'], 1)
        self.assertEqual(result['interchain_heavy_atom_clashes'], 0)
        self.assertAlmostEqual(result['interchain_minimum_heavy_atom_distance_angstrom'], 2.)

    def test_missing_nonfinite_or_degenerate_coords_rejected(self):
        for moving in [BACKBONE[:-1], BACKBONE * np.nan, np.zeros_like(BACKBONE)]:
            with self.assertRaises(ValueError):
                proper_transform(BACKBONE, moving)

    def test_sequence_bias_and_homopolymer(self):
        metrics = sequence_metrics('AAAAAGSV')
        self.assertEqual(metrics['maximum_homopolymer_run'], 5)
        self.assertAlmostEqual(metrics['max_amino_acid_fraction'], 5/8)
        with self.assertRaises(ValueError):
            sequence_metrics('')


class StructureTests(unittest.TestCase):
    def test_strict_sequence_and_heavy_atom_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.pdb'
            write_structure(path, [('A', 'AGSV', BACKBONE)])
            self.assertEqual(read_chains(path, {'A':'AGSV'})['A'].sequence, 'AGSV')
            with self.assertRaisesRegex(ValueError, 'sequence mismatch'):
                read_chains(path, {'A':'AGSA'})
            original = path.read_text()
            path.write_text('\n'.join(line for line in original.splitlines()
                                      if not (line.startswith('ATOM') and line[12:16].strip() == 'CA'))+'\n')
            with self.assertRaisesRegex(ValueError, 'Missing/unexpected heavy atoms'):
                read_chains(path, {'A':'AGSV'})
            path.write_text('\n'.join(line for line in original.splitlines()
                                      if not (line.startswith('ATOM') and line[12:16].strip() == 'OG'))+'\n')
            with self.assertRaisesRegex(ValueError, 'Missing/unexpected heavy atoms'):
                read_chains(path, {'A':'AGSV'})

    def test_nonfinite_coordinates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.pdb'
            write_structure(path, [('A', 'AGSV', BACKBONE)])
            text = path.read_text()
            lines = text.splitlines()
            i = next(i for i, line in enumerate(lines) if line.startswith('ATOM'))
            lines[i] = lines[i][:30] + '     nan' + lines[i][38:]
            path.write_text('\n'.join(lines)+'\n')
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                read_chains(path, {'A':'AGSV'})


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.design, self.refolds = self.root/'design', self.root/'refolds'
        self.design.mkdir()
        self.target, self.protocol = self.root/'target.cif', self.root/'protocol.json'
        write_structure(self.target, [('A', 'AGSV', BACKBONE)])
        write_structure(self.design/'hallucinated.pdb',
                        [('A', 'AGSV', BACKBONE+[0.,0.,5.]), ('B','AGSV',BACKBONE)])
        np.savez(self.design/'hard-output.npz', sequence='AGSVAGSV', asym_id=[1]*4+[2]*4,
                 plddt=np.full(8,.95), pae=np.ones((8,8)))
        names = ['hallucinated','mpnn_1','mpnn_2']
        # Synthetic fixture thresholds deliberately broad, not the live protocol.
        from evaluate_binder import SCREEN_FIELDS
        screen = {key: (0 if direction == 'min' else 100) for key, direction in SCREEN_FIELDS.values()}
        screen.update(both_boltz_seeds_must_pass=True, interchain_heavy_atom_contact_cutoff_angstrom=5.)
        protocol = {'target':{'chain':'A','length':4}, 'binder':{'length':4},
                    'redesign':{'candidates':names}, 'screen':screen,
                    'independent_refold':{'seeds':[7,11], 'model':'fixture', 'revision':'test',
                                         'profile':'balanced', 'pair_precision':'float32'}}
        self.protocol.write_text(json.dumps(protocol))
        (self.design/'summary.json').write_text(json.dumps({
            'candidates':[{'name':n, 'sequence':'AGSV'} for n in names],
            'protocol_sha256':sha256(self.protocol), 'target':{'input_sha256':sha256(self.target)}}))
        for name in names:
            for seed in (7,11):
                folder = self.refolds/name/str(seed)
                folder.mkdir(parents=True)
                write_structure(folder/'prediction.cif', [('A','AGSV',BACKBONE+[0.,0.,5.]),('B','AGSV',BACKBONE)])
                metadata = {'seed':seed,'model':'fixture','revision':'test','profile':'balanced',
                            'precision':'float32', 'iptm':.9, 'plddt_mean':.95,
                            'chains':[{'id':'A','sequence':'AGSV'},{'id':'B','sequence':'AGSV'}]}
                (folder/'prediction.json').write_text(json.dumps(metadata))

    def tearDown(self):
        self.tmp.cleanup()

    def report(self):
        report = evaluate(self.design, self.refolds, self.protocol, self.target)
        json.dumps(report, allow_nan=False)
        return report

    def test_both_seeds_required_and_failures_retained(self):
        self.assertEqual(self.report()['passing_candidates'], 3)
        (self.refolds/'hallucinated'/'11'/'prediction.json').unlink()
        report = self.report()
        self.assertEqual(report['passing_candidates'], 2)
        candidate = report['candidates'][0]
        self.assertEqual(len(candidate['trials']), 2)
        self.assertEqual(candidate['passing_seeds'], 1)
        self.assertFalse(candidate['passes_both_seeds'])
        self.assertTrue(candidate['trials'][1]['errors'])

    def test_missing_or_mismatched_provenance_fails_globally(self):
        path = self.design/'summary.json'
        baseline = json.loads(path.read_text())
        for key in ('protocol_sha256', 'target'):
            for value in (None, 'incorrect-sha256'):
                with self.subTest(key=key, value=value):
                    metadata = dict(baseline)
                    metadata[key] = ({'input_sha256':value} if key == 'target' else value)
                    path.write_text(json.dumps(metadata))
                    report = self.report()
                    self.assertEqual(report['passing_candidates'], 0)
                    self.assertTrue(report['global_errors'])
                    self.assertEqual(len(report['candidates']), 3)
                    for candidate in report['candidates']:
                        for trial in candidate['trials']:
                            self.assertFalse(trial['passes_screen'])
                            self.assertEqual(trial['metrics']['complex_iptm'], .9)
        path.write_text(json.dumps(baseline))

    def test_af2_raw_confidence_matches_rounded_pdb(self):
        np.savez(self.design/'hard-output.npz', sequence='AGSVAGSV', asym_id=[1]*4+[2]*4,
                 plddt=np.full(8,.95004), pae=np.ones((8,8)))
        self.assertEqual(self.report()['passing_candidates'], 3)
        np.savez(self.design/'hard-output.npz', sequence='AGSVAGSV', asym_id=[1]*4+[2]*4,
                 plddt=np.full(8,.99), pae=np.ones((8,8)))
        report = self.report()
        self.assertEqual(report['passing_candidates'], 0)
        self.assertTrue(any('rounded AF2 PDB confidence' in error for error in report['global_errors']))

    def test_absent_plddt_never_uses_cif_default_100(self):
        path = self.refolds/'hallucinated'/'7'/'prediction.json'
        metadata = json.loads(path.read_text())
        metadata.update(plddt_mean=None, ptm=float('nan'))
        path.write_text(json.dumps(metadata))
        trial = self.report()['candidates'][0]['trials'][0]
        self.assertFalse(trial['passes_screen'])
        self.assertIsNone(trial['checks']['binder_plddt_0_to_100']['value'])
        self.assertIsNone(trial['provenance']['ptm'])

    def test_sequence_mismatch_fails_closed_without_hiding_other_metrics(self):
        path = self.refolds/'hallucinated'/'7'/'prediction.json'
        metadata = json.loads(path.read_text())
        metadata['chains'][0]['sequence'] = 'AAAA'
        path.write_text(json.dumps(metadata))
        trial = self.report()['candidates'][0]['trials'][0]
        self.assertFalse(trial['passes_screen'])
        self.assertEqual(trial['metrics']['complex_iptm'], .9)
        self.assertIn('Refold metadata chains/sequence mismatch', trial['errors'])


if __name__ == '__main__':
    unittest.main()
