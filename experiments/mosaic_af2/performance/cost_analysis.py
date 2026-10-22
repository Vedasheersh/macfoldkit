"""Authoritative FLOP count for the AF2 design gradient, via XLA cost analysis.

Divides the compiled cost by the measured step time to get achieved throughput, which is
then comparable to the machine's empirical peak from tri_roofline.py.
"""
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / 'outputs/macfoldkit/experiments/mosaic_af2'))
import run_design as R  # noqa: E402


def main():
    weights, target_path, length = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    out = Path(sys.argv[4])
    structure, target = R.load_target(target_path)
    spec = R.loss_specification(True, 0.0, 4)        # AF2 terms only; MPNN measured separately
    model = R.SingleAF2(weights, use_remat=True)
    from mosaic.structure_prediction import TargetChain
    chains = [TargetChain(target['sequence'], use_msa=False, template_chain=structure[0][0])]
    features, _ = model.binder_features(length, chains=chains)
    features = jax.tree.map(jnp.asarray, features)
    terms = R.build_terms(spec, None)
    loss = R.DesignLoss(model, features, terms)
    x = jnp.asarray(np.full((length, 20), 0.05, dtype=np.float32))
    key = jax.random.key(7)

    scalar = lambda v, k: loss(v, key=k)[0]
    fwd = jax.jit(scalar)
    grad = jax.jit(jax.value_and_grad(scalar))

    report = {'binder_length': length, 'target_length': target['length'],
              'total_length': length + target['length']}
    for name, fn in [('forward', fwd), ('value_and_grad', grad)]:
        lowered = fn.lower(x, key)
        try:
            cost = lowered.compile().cost_analysis()
            cost = cost[0] if isinstance(cost, list) else cost
            flops = float(cost.get('flops', 0.0))
            bytes_accessed = float(cost.get('bytes accessed', 0.0))
        except Exception as error:                     # backend may not expose it
            flops, bytes_accessed, cost = 0.0, 0.0, {'error': repr(error)}
        import time
        jax.block_until_ready(fn(x, key))
        times = []
        for _ in range(3):
            start = time.perf_counter(); jax.block_until_ready(fn(x, key))
            times.append(time.perf_counter() - start)
        seconds = min(times)
        report[name] = dict(seconds=seconds, flops=flops, bytes_accessed=bytes_accessed,
                            gflops_achieved=flops / seconds / 1e9 if flops else None,
                            arithmetic_intensity=flops / bytes_accessed if bytes_accessed else None)
        print(name, json.dumps(report[name]), flush=True)
    if report['forward']['seconds']:
        report['backward_multiplier'] = (report['value_and_grad']['seconds']
                                         / report['forward']['seconds'])
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
