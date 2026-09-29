"""Cost of a Mosaic design gradient through Boltz2, versus the measured AF2 baseline.

AF2 reference on this machine, 124-residue complex: 2.08 s per gradient step, 2.66 GB peak.

Boltz2's forward is a trunk with recycling plus N diffusion sampling steps, all
differentiated. Mosaic defaults to sampling_steps=25. Sweeping it shows whether diffusion
dominates and therefore whether reducing it makes Boltz2 design affordable.

Reports OOM and other failures rather than dying, so a failure is still a result.
"""
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def main():
    checkpoint, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    binder_length = int(sys.argv[3])
    sweep = [int(s) for s in sys.argv[4].split(',')]

    # torch >= 2.6 defaults torch.load to weights_only=True, and this checkpoint carries
    # an omegaconf DictConfig, which lightning's loader then refuses. Allowlisting the
    # config classes did not suffice, so weights_only is forced off for this one file.
    # Justification: it was fetched from Boltz's own model gateway over HTTPS and its
    # length matched the advertised content-length exactly. This is a deliberate, scoped
    # trust decision about a specific local artifact, not a general default.
    import torch
    _original_torch_load = torch.load

    def _load_trusted(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _load_trusted

    from mosaic.models.boltz2 import Boltz2
    from mosaic.structure_prediction import TargetChain
    import mosaic.losses.structure_prediction as sp

    UBIQUITIN = ('MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLED'
                 'GRTLSDYNIQKESTLHLVLRLRGG')
    report = {'binder_length': binder_length, 'target_length': len(UBIQUITIN),
              'total_length': binder_length + len(UBIQUITIN),
              'device': str(jax.devices()[0]), 'runs': []}

    start = time.perf_counter()
    model = Boltz2(checkpoint)
    report['load_seconds'] = time.perf_counter() - start
    print('loaded in %.1f s' % report['load_seconds'], flush=True)

    start = time.perf_counter()
    features, _writer = Boltz2.binder_features(
        binder_length, [TargetChain(UBIQUITIN, use_msa=False)])
    report['feature_seconds'] = time.perf_counter() - start
    print('features in %.1f s' % report['feature_seconds'], flush=True)

    x = jnp.asarray(np.full((binder_length, 20), 0.05, dtype=np.float32))
    terms = sp.PLDDTLoss() + 0.25 * sp.IPTMLoss()

    for steps in sweep:
        row = {'sampling_steps': steps}
        try:
            loss = model.build_loss(loss=terms, features=features,
                                    recycling_steps=1, sampling_steps=steps)
            grad = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))
            key = jax.random.key(7)
            begin = time.perf_counter()
            (value, _), g = jax.block_until_ready(grad(x, key=key))
            row['compile_and_first_seconds'] = time.perf_counter() - begin
            times = []
            for _ in range(2):
                begin = time.perf_counter()
                jax.block_until_ready(grad(x, key=key))
                times.append(time.perf_counter() - begin)
            row['warm_gradient_seconds'] = min(times)
            row['loss'] = float(value)
            row['gradient_finite'] = bool(np.isfinite(np.asarray(g)).all())
            stats = jax.devices()[0].memory_stats()   # None on the CPU backend
            if stats:
                row['peak_gb'] = stats['peak_bytes_in_use'] / 1e9
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {str(error)[:220]}'
        report['runs'].append(row)
        print(json.dumps(row), flush=True)
        out_path.write_text(json.dumps(report, indent=2) + '\n')

    out_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
