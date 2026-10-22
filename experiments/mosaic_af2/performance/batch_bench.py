"""Does batching independent designs improve GPU occupancy?

Triangle multiplication reaches only 26% of this machine's peak at N=124 but 55% at
N=231, so the small-complex case underutilizes 20 GPU cores. Independent design
trajectories are embarrassingly parallel, and the v3 campaign showed several seeds are
needed anyway. vmap over B initializations should raise utilization, making B designs
cost materially less than B sequential ones.

Reports seconds per design, which is the number that matters.
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / 'outputs/macfoldkit/experiments/mosaic_af2'))
import run_design as R  # noqa: E402


def main():
    weights, target_path, length = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    out = Path(sys.argv[4])
    batches = [int(b) for b in sys.argv[5].split(',')]
    structure, target = R.load_target(target_path)
    spec = R.loss_specification(True, 0.0, 4)     # AF2 terms only, to isolate the model
    model = R.SingleAF2(weights, use_remat=True)
    from mosaic.structure_prediction import TargetChain
    chains = [TargetChain(target['sequence'], use_msa=False, template_chain=structure[0][0])]
    features, _ = model.binder_features(length, chains=chains)
    features = jax.tree.map(jnp.asarray, features)
    loss = R.DesignLoss(model, features, R.build_terms(spec, None))

    scalar = lambda v, k: loss(v, key=k)[0]
    single = jax.jit(jax.value_and_grad(scalar))
    batched = jax.jit(jax.vmap(jax.value_and_grad(scalar), in_axes=(0, None)))

    rng = np.random.default_rng(7)
    report = {'length': length, 'total_length': length + target['length'], 'runs': []}
    for b in batches:
        x = jnp.asarray(rng.random((b, length, 20), dtype=np.float32))
        x = x / x.sum(-1, keepdims=True)
        key = jax.random.key(7)
        fn = (lambda: single(x[0], key)) if b == 1 else (lambda: batched(x, key))
        try:
            jax.block_until_ready(fn())
            times = []
            for _ in range(3):
                start = time.perf_counter()
                jax.block_until_ready(fn())
                times.append(time.perf_counter() - start)
            seconds = min(times)
            mem = jax.devices()[0].memory_stats()['peak_bytes_in_use'] / 1e9
            row = dict(batch=b, seconds=seconds, seconds_per_design=seconds / b,
                       peak_gb=mem)
        except Exception as error:
            row = dict(batch=b, error=repr(error)[:200])
        report['runs'].append(row)
        print(json.dumps(row), flush=True)
    base = next((r['seconds_per_design'] for r in report['runs']
                 if r.get('batch') == 1 and 'seconds_per_design' in r), None)
    if base:
        for r in report['runs']:
            if 'seconds_per_design' in r:
                r['speedup_per_design'] = base / r['seconds_per_design']
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
