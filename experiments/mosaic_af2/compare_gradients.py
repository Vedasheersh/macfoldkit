"""Compare identical CPU/MPS raw probes, including the simplex tangent space."""
import argparse
import json
from pathlib import Path
import numpy as np


def metrics(reference, actual):
    norm = np.linalg.norm(reference)
    other_norm = np.linalg.norm(actual)
    return dict(relative_l2=float(np.linalg.norm(actual-reference)/norm),
        cosine=float(np.sum(reference*actual)/(norm*other_norm)),
        max_absolute_error=float(np.max(np.abs(actual-reference))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cpu', type=Path)
    parser.add_argument('mps', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--relative-tolerance', type=float, default=0.005)
    parser.add_argument('--loss-tolerance', type=float, default=1e-4)
    args = parser.parse_args()
    cpu = np.load(args.cpu/'initial-gradient.npz')
    mps = np.load(args.mps/'initial-gradient.npz')
    cpu_info = json.loads((args.cpu/'probe.json').read_text())
    mps_info = json.loads((args.mps/'probe.json').read_text())
    if cpu_info['platform'] != 'cpu' or mps_info['platform'] != 'mps':
        raise ValueError('Expected a CPU reference and an MPS probe')
    for key in ['model', 'precision', 'forward_passes', 'length', 'seed',
                'objective_seed', 'mpnn_weight', 'mpnn_samples', 'feature_sha256',
                'loss_terms', 'protocol_sha256']:
        if cpu_info[key] != mps_info[key]:
            raise ValueError(f'Probe settings differ: {key}')
    np.testing.assert_array_equal(cpu['sequence'], mps['sequence'])
    for d in [cpu, mps]:
        if d['gradient'].shape != d['sequence'].shape or d['loss'].shape != ():
            raise ValueError('Expected a sequence-shaped gradient and scalar loss')
        for key in ['sequence', 'loss', 'gradient']:
            if not np.isfinite(d[key]).all():
                raise ValueError(f'Nonfinite {key}')
    a, b = cpu['gradient'].astype('float64'), mps['gradient'].astype('float64')
    report = dict(raw=metrics(a, b),
        simplex_tangent=metrics(a-a.mean(-1,keepdims=True), b-b.mean(-1,keepdims=True)),
        loss_absolute_error=float(abs(cpu['loss']-mps['loss'])),
        relative_tolerance=args.relative_tolerance, loss_tolerance=args.loss_tolerance,
        cpu=cpu_info, mps=mps_info)
    report['passed'] = (report['raw']['relative_l2'] <= args.relative_tolerance
        and report['simplex_tangent']['relative_l2'] <= args.relative_tolerance
        and report['loss_absolute_error'] <= args.loss_tolerance)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['cpu','mps']}, indent=2))
    if not report['passed']:
        raise SystemExit('CPU/MPS numerical gate failed')


if __name__ == '__main__':
    main()
