"""Memory-bandwidth roofline for this machine, to place the design gradient on it.

The matmul roofline (tri_roofline.py) bounds the compute side at ~7.4 TFLOPS. The AF2
design gradient runs at ~9% of that, and is a long chain of low-intensity operations, so
the bound that actually applies is bandwidth, not FLOPs. This measures achievable
bandwidth with streaming kernels at sizes far beyond any cache, so "are we near the limit"
can be answered against the right roofline.

  copy   : read N, write N              -> 2N bytes
  scale  : read N, write N              -> 2N bytes
  triad  : read 2N, write N             -> 3N bytes
  reduce : read N                       -> 1N bytes

Run under each environment separately:
  <mosaic python>  bandwidth_roofline.py jax out.json
  <mlx venv>       bandwidth_roofline.py mlx out.json
"""
import json
import sys
import time

SIZES_MB = [64, 256, 1024]
REPEATS, WARMUP = 10, 3


def bench(fn, sync):
    for _ in range(WARMUP):
        sync(fn())
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        sync(fn())
        times.append(time.perf_counter() - start)
    return min(times)


def run_jax():
    import jax, jax.numpy as jnp
    sync = jax.block_until_ready
    out = {'backend': 'jax-mps', 'version': jax.__version__,
           'device': str(jax.devices()[0]), 'kernels': []}
    key = jax.random.key(0)
    for mb in SIZES_MB:
        n = mb * 1_000_000 // 4
        a = jax.random.normal(key, (n,), jnp.float32)
        b = a + 1.0
        jax.block_until_ready((a, b))
        for name, fn, bytes_moved in [
            ('copy', jax.jit(lambda x: x * 1.0), 2 * n * 4),
            ('scale', jax.jit(lambda x: x * 2.5), 2 * n * 4),
            ('triad', jax.jit(lambda x, y: x + 2.5 * y), 3 * n * 4),
            ('reduce', jax.jit(lambda x: x.sum()), 1 * n * 4),
        ]:
            call = (lambda: fn(a, b)) if name == 'triad' else (lambda: fn(a))
            seconds = bench(call, sync)
            out['kernels'].append(dict(kernel=name, mb=mb, seconds=seconds,
                                       gb_per_s=bytes_moved / seconds / 1e9))
    return out


def run_mlx():
    import mlx.core as mx
    sync = mx.eval
    out = {'backend': 'mlx', 'version': mx.__version__,
           'device': str(mx.default_device()), 'kernels': []}
    for mb in SIZES_MB:
        n = mb * 1_000_000 // 4
        a = mx.random.normal((n,), dtype=mx.float32)
        b = a + 1.0
        mx.eval(a, b)
        for name, fn, bytes_moved in [
            ('copy', mx.compile(lambda x: x * 1.0), 2 * n * 4),
            ('scale', mx.compile(lambda x: x * 2.5), 2 * n * 4),
            ('triad', mx.compile(lambda x, y: x + 2.5 * y), 3 * n * 4),
            ('reduce', mx.compile(lambda x: mx.sum(x)), 1 * n * 4),
        ]:
            call = (lambda: fn(a, b)) if name == 'triad' else (lambda: fn(a))
            seconds = bench(call, sync)
            out['kernels'].append(dict(kernel=name, mb=mb, seconds=seconds,
                                       gb_per_s=bytes_moved / seconds / 1e9))
    return out


if __name__ == '__main__':
    result = {'jax': run_jax, 'mlx': run_mlx}[sys.argv[1]]()
    best = max(k['gb_per_s'] for k in result['kernels'])
    result['peak_gb_per_s'] = best
    print(json.dumps(result, indent=2))
    if len(sys.argv) > 2:
        json.dump(result, open(sys.argv[2], 'w'), indent=2)
