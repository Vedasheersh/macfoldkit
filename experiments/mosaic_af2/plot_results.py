"""Render the recorded experimental trajectory and independent refold overlay."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_refolds import read_ca

root = Path(__file__).resolve().parent/'results'
trajectory = json.loads((root/'joint-trajectory.json').read_text())
_, a = read_ca(root/'joint-backbone.pdb')
_, b = read_ca(root/'joint-mpnn_2-boltz.cif')
a -= a.mean(0); b -= b.mean(0)
u, _, vh = np.linalg.svd(b.T @ a)
fix = np.eye(3); fix[-1,-1] = np.linalg.det(u @ vh)
b = b @ (u @ fix @ vh)
fig = plt.figure(figsize=(10,4), layout='constrained')
ax = fig.add_subplot(121)
ax.plot(range(len(trajectory)), [v['loss'] for v in trajectory], color='#305d9a', lw=1.5)
ax.set(xlabel='Sequence update', ylabel='Composite design loss', title='Mosaic: 150 AF2 + ProteinMPNN updates')
ax.spines[['top','right']].set_visible(False)
ax.grid(alpha=.15)
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(*a.T, color='#305d9a', lw=2, label='AF2-designed backbone')
ax2.plot(*b.T, color='#cf6730', lw=1.5, alpha=.8, label='Independent Boltz2 refold')
ax2.set_title('MPNN candidate 2: 1.00 Å Cα RMSD')
ax2.set_box_aspect(np.ptp(np.vstack([a,b]),axis=0))
ax2.set_axis_off();ax2.legend(loc='lower center', fontsize=8)
fig.suptitle('Apple M5 Pro · 64-residue de novo monomer · computational demonstration', fontsize=11)
fig.savefig(root/'workflow.png',dpi=180)
