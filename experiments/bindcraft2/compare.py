"""Compare two probe runs elementwise: forward values and every gradient array."""
import json
import sys

import numpy as np


def rel(a, b):
    a = a.astype('float64')
    b = b.astype('float64')
    denom = max(np.abs(a).max(), np.abs(b).max(), 1e-30)
    return float(np.abs(a - b).max() / denom)


def main():
    cpu_json, mps_json = sys.argv[1], sys.argv[2]
    cpu = json.load(open(cpu_json))
    mps = json.load(open(mps_json))
    ca = np.load(cpu_json.replace('.json', '') + '-arrays.npz')
    ma = np.load(mps_json.replace('.json', '') + '-arrays.npz')

    gate = 0.005
    worst = 0.0
    rows = []
    for c, m in zip(cpu['checks'], mps['checks']):
        assert c['label'] == m['label']
        if 'error' in c or 'error' in m:
            rows.append((c['label'], 'ERROR', c.get('error') or m.get('error')))
            continue
        fwd = rel(ca[f"{c['label']}::forward"], ma[f"{c['label']}::forward"])
        grad_rels = {}
        for name in c['grad_names']:
            key = f"{c['label']}::grad::{name}"
            grad_rels[name] = rel(ca[key], ma[key])
        gmax = max(grad_rels.values())
        norm_rel = max(abs(x - y) / max(abs(x), 1e-30)
                       for x, y in zip(c['grad_norms'], m['grad_norms']))
        zero_match = c['grad_zero_fraction'] == m['grad_zero_fraction']
        worst = max(worst, fwd, gmax)
        rows.append(dict(
            label=c['label'],
            forward_rel=fwd,
            grad_rel_max=gmax,
            grad_rel_worst_tensor=max(grad_rels, key=grad_rels.get),
            grad_norm_rel_max=norm_rel,
            zero_fraction_identical=zero_match,
            cpu_zero_fracs=c['grad_zero_fraction'],
            mps_zero_fracs=m['grad_zero_fraction'],
            cpu_zero_rows=c['activation_zero_rows'],
            mps_zero_rows=m['activation_zero_rows'],
            cpu_seconds=c['seconds'], mps_seconds=m['seconds'],
            speedup=c['seconds'] / m['seconds'] if m['seconds'] else None,
            passes_half_percent_gate=bool(fwd < gate and gmax < gate and zero_match),
        ))
    out = dict(gate=gate, worst_relative=worst,
               all_pass=all(r['passes_half_percent_gate'] for r in rows if isinstance(r, dict)),
               rows=rows)
    print(json.dumps(out, indent=2))
    with open(sys.argv[3], 'w') as h:
        json.dump(out, h, indent=2)


if __name__ == '__main__':
    main()
