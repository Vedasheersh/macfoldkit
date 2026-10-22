"""Plot a completed binder run, preserving target-only alignment of all poses.

Requires NumPy, Gemmi and Matplotlib. Inputs are the original design directory,
refold root and evaluator JSON; no model runtime or GPU is used.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_binder import read_chains, proper_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('design', type=Path)
    parser.add_argument('refolds', type=Path)
    parser.add_argument('validation', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = json.loads(args.validation.read_text())
    history = json.loads((args.design/'trajectory.json').read_text())
    reference = read_chains(args.design/'hallucinated.pdb')
    fig = plt.figure(figsize=(11, 8), layout='constrained')
    ax = fig.add_subplot(221)
    losses = np.asarray([r['loss'] for r in history])
    ax.plot(losses, color='#456987', lw=1, alpha=.65, label='Evaluated loss')
    ax.plot(np.minimum.accumulate(losses), color='#123b58', lw=2, label='Best evaluated loss')
    ax.set(xlabel='Sequence evaluation', ylabel='Composite objective', title='150 Mosaic optimization steps')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.15)
    for i, candidate in enumerate(report['candidates']):
        ax = fig.add_subplot(2, 2, i+2, projection='3d')
        ax.plot(*reference['B'].ca.T, color='#777777', lw=2, label='AF2 target')
        ax.plot(*reference['A'].ca.T, color='#123b58', lw=2, label='AF2 designed binder')
        all_coords = [reference['A'].ca, reference['B'].ca]
        lines = []
        for trial, color in zip(candidate['trials'], ['#df8138', '#20a29c'], strict=True):
            files = [p for p in (args.refolds/candidate['name']).rglob('prediction.json')
                     if json.loads(p.read_text())['seed'] == trial['seed']]
            if len(files) != 1:
                raise ValueError('Expected exactly one refold per seed')
            moving = read_chains(files[0].with_suffix('.cif'))
            r, t = proper_transform(reference['B'].ca, moving['B'].ca)
            xyz = moving['A'].ca @ r + t
            all_coords.append(xyz)
            ax.plot(*xyz.T, color=color, lw=1.6, label=f"Boltz seed {trial['seed']}")
            metric = trial['metrics']
            lines.append(f"{trial['seed']}: iPTM {metric['complex_iptm']:.2f}, pose RMSD {metric['target_aligned_binder_ca_rmsd_angstrom']:.1f} Å")
        ax.set_title(candidate['name'] + '\n' + ' | '.join(lines), fontsize=9)
        ax.set_box_aspect(np.maximum(np.ptp(np.vstack(all_coords), axis=0), 1))
        ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"M5 Pro · ubiquitin binder workflow · {report['passing_candidates']}/3 pass the frozen screen\n"
                 'Backbone traces; Boltz binders aligned by the target only. Computational evidence, not binding validation.', fontsize=11)
    fig.savefig(args.output, dpi=180)


if __name__ == '__main__':
    main()
