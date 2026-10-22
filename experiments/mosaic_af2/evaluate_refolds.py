"""Compare independent Boltz refolds to a de novo AF2-generated backbone."""
import argparse
from collections import Counter
import json
from pathlib import Path
import gemmi
import numpy as np


def read_ca(path):
    st = gemmi.read_structure(str(path))
    st.remove_ligands_and_waters()
    st.remove_alternative_conformations()
    st.remove_empty_chains()
    if len(st) != 1 or len(st[0]) != 1:
        raise ValueError(f'Expected one model and chain: {path}')
    residues = list(st[0][0])
    ca = np.array([[r.sole_atom('CA').pos.x, r.sole_atom('CA').pos.y,
                    r.sole_atom('CA').pos.z] for r in residues])
    if not np.isfinite(ca).all():
        raise ValueError('Nonfinite coordinates')
    return gemmi.one_letter_code([r.name for r in residues]), ca


def rmsd(a, b):
    if a.shape != b.shape:
        raise ValueError('Residue counts differ')
    a, b = a-a.mean(0), b-b.mean(0)
    u, _, vh = np.linalg.svd(b.T @ a)
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(u @ vh)
    return float(np.sqrt(np.mean(np.sum((b @ (u @ fix @ vh)-a)**2, axis=-1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('design', type=Path)
    parser.add_argument('refolds', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    design = json.loads((args.design/'summary.json').read_text())
    _, reference = read_ca(args.design/'hallucinated.pdb')
    rows = []
    for item in design['candidates']:
        files = list((args.refolds/item['name']).rglob('prediction.json'))
        if len(files) != 1:
            raise ValueError(f'Expected one prediction for {item["name"]}, got {files}')
        metadata = json.loads(files[0].read_text())
        sequence, coords = read_ca(files[0].with_suffix('.cif'))
        if sequence != item['sequence']:
            raise ValueError('Refolded sequence mismatch')
        r = rmsd(reference, coords)
        plddt = metadata['plddt_mean']*100
        ptm = metadata['ptm']
        fractions = np.asarray(list(Counter(sequence).values()))/len(sequence)
        rows.append(dict(name=item['name'], sequence=sequence, ca_rmsd_angstrom=r,
            plddt_0_to_100=plddt, ptm=ptm, inference_seconds=metadata['inference_seconds'],
            max_amino_acid_fraction=float(fractions.max()),
            sequence_entropy_bits=float(-np.sum(fractions*np.log2(fractions))),
            passes_example_screen=plddt>=80 and ptm>=0.6 and r<=2.0))
    report = dict(scope='In-silico monomer self-consistency only; no binding or experimental validation.',
        alignment='Every C-alpha by residue position, proper-rotation Kabsch',
        example_screen={'plddt_min':80, 'ptm_min':0.6, 'ca_rmsd_max_angstrom':2.0},
        design_loss_initial=design['initial_loss'], design_loss_best=design['best_soft_loss'],
        candidates=rows)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
