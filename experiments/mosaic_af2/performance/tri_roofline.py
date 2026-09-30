"""Backend comparison on AF2's dominant Evoformer op, plus an empirical matmul roofline.

Triangle multiplication is einsum('ikc,jkc->ijc'), which is a batched matmul over the
channel axis: A(c,N,N) @ B(c,N,N)^T, costing 2*c*N^3 FLOPs. It dominates AF2/Boltz pair
updates, so its achieved throughput against the machine's own peak matmul throughput
bounds what any kernel work can win.

Run under each backend separately; compare the JSON files.
  <mosaic python>  tri_roofline.py jax   out.json
  <mlx venv>       tri_roofline.py mlx   out.json
  <boltz python>   tri_roofline.py torch out.json
"""
import json
import sys
import time

N_PAIR = [124, 231]      # ubiquitin complex and EGFR domain III complex
C_PAIR = 128             # AF2 pair channels
ROOFLINE = [1024, 2048, 4096]
WARMUP, REPEATS = 2, 5


def bench(fn, sync, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        sync(fn())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        sync(fn())
        times.append(time.perf_counter() - start)
    return min(times)


def run_jax():
    import jax, jax.numpy as jnp
    key = jax.random.key(0)
    sync = lambda x: jax.block_until_ready(x)
    out = {'backend': 'jax', 'version': jax.__version__, 'device': str(jax.devices()[0])}
    mm = jax.jit(lambda a, b: a @ b)
    tri = jax.jit(lambda a, b: a @ jnp.swapaxes(b, -1, -2))
    out['roofline'] = []
    for n in ROOFLINE:
        a = jax.random.normal(key, (n, n), jnp.float32); b = a + 1
        t = bench(lambda: mm(a, b), sync)
        out['roofline'].append(dict(n=n, seconds=t, gflops=2 * n ** 3 / t / 1e9))
    out['triangle'] = []
    for n in N_PAIR:
        a = jax.random.normal(key, (C_PAIR, n, n), jnp.float32); b = a + 1
        t = bench(lambda: tri(a, b), sync)
        out['triangle'].append(dict(n=n, c=C_PAIR, seconds=t,
                                    gflops=2 * C_PAIR * n ** 3 / t / 1e9))
    return out


def run_mlx():
    import mlx.core as mx
    sync = lambda x: mx.eval(x)
    out = {'backend': 'mlx', 'version': mx.__version__, 'device': str(mx.default_device())}
    out['roofline'] = []
    for n in ROOFLINE:
        a = mx.random.normal((n, n), dtype=mx.float32); b = a + 1
        mx.eval(a, b)
        t = bench(lambda: a @ b, sync)
        out['roofline'].append(dict(n=n, seconds=t, gflops=2 * n ** 3 / t / 1e9))
    out['triangle'] = []
    for n in N_PAIR:
        a = mx.random.normal((C_PAIR, n, n), dtype=mx.float32); b = a + 1
        mx.eval(a, b)
        t = bench(lambda: a @ mx.swapaxes(b, -1, -2), sync)
        out['triangle'].append(dict(n=n, c=C_PAIR, seconds=t,
                                    gflops=2 * C_PAIR * n ** 3 / t / 1e9))
    return out


def run_torch():
    import torch
    dev = torch.device('mps')
    sync = lambda x: torch.mps.synchronize()
    out = {'backend': 'torch', 'version': torch.__version__, 'device': str(dev)}
    out['roofline'] = []
    for n in ROOFLINE:
        a = torch.randn(n, n, device=dev); b = a + 1
        t = bench(lambda: a @ b, sync)
        out['roofline'].append(dict(n=n, seconds=t, gflops=2 * n ** 3 / t / 1e9))
    out['triangle'] = []
    for n in N_PAIR:
        a = torch.randn(C_PAIR, n, n, device=dev); b = a + 1
        t = bench(lambda: a @ b.transpose(-1, -2), sync)
        out['triangle'].append(dict(n=n, c=C_PAIR, seconds=t,
                                    gflops=2 * C_PAIR * n ** 3 / t / 1e9))
    return out


if __name__ == '__main__':
    result = {'jax': run_jax, 'mlx': run_mlx, 'torch': run_torch}[sys.argv[1]]()
    print(json.dumps(result, indent=2))
    if len(sys.argv) > 2:
        json.dump(result, open(sys.argv[2], 'w'), indent=2)
