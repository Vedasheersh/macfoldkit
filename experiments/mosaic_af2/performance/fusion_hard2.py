"""Re-test the 1.67x claim with matched inputs and a checksum guard.

The original test built wl and wg from the SAME jax key, so they were identical arrays and
XLA could common-subexpression the two matmuls into one, while MLX -- seeded separately --
computed both. That makes the comparison unfair, though in MLX's disfavour, so the
direction of the reported gap is not explained by it. Here both frameworks get bit-identical
numpy inputs and the outputs are checksummed.
"""
import json, sys, time
import numpy as np
N, C = 124, 128
REPEATS, WARMUP = 30, 5
rng = np.random.default_rng(0)
Z  = rng.standard_normal((N, N, C)).astype('float32')
WL = (rng.standard_normal((C, C)) * 0.05).astype('float32')
WG = (rng.standard_normal((C, C)) * 0.05).astype('float32')

def bench(fn, sync):
    for _ in range(WARMUP): sync(fn())
    t = []
    for _ in range(REPEATS):
        s = time.perf_counter(); sync(fn()); t.append(time.perf_counter() - s)
    return min(t), float(np.median(t))

def run_jax():
    import jax, jax.numpy as jnp
    z, wl, wg = map(jnp.asarray, (Z, WL, WG))
    def block(z, wl, wg):
        m = z.mean(-1, keepdims=True); v = z.var(-1, keepdims=True)
        n = (z - m) * jax.lax.rsqrt(v + 1e-5)
        a = jax.nn.sigmoid(n @ wg) * (n @ wl)
        return z + jnp.einsum('ikc,jkc->ijc', a, a)
    j = jax.jit(block)
    mn, md = bench(lambda: j(z, wl, wg), jax.block_until_ready)
    return {'framework': 'jax-mps', 'min_seconds': mn, 'median_seconds': md,
            'checksum': float(np.asarray(j(z, wl, wg), dtype='float64').sum())}

def run_mlx():
    import mlx.core as mx
    z, wl, wg = map(mx.array, (Z, WL, WG)); mx.eval(z, wl, wg)
    def block(z, wl, wg):
        m = mx.mean(z, axis=-1, keepdims=True); v = mx.var(z, axis=-1, keepdims=True)
        n = (z - m) * mx.rsqrt(v + 1e-5)
        a = mx.sigmoid(n @ wg) * (n @ wl)
        return z + mx.einsum('ikc,jkc->ijc', a, a)
    c = mx.compile(block)
    mn, md = bench(lambda: c(z, wl, wg), mx.eval)
    mne, mde = bench(lambda: block(z, wl, wg), mx.eval)
    return {'framework': 'mlx', 'min_seconds': mn, 'median_seconds': md,
            'eager_min_seconds': mne,
            'checksum': float(np.asarray(c(z, wl, wg), dtype='float64').sum())}

if __name__ == '__main__':
    r = {'jax': run_jax, 'mlx': run_mlx}[sys.argv[1]]()
    print(json.dumps(r, indent=2))
    if len(sys.argv) > 2: json.dump(r, open(sys.argv[2], 'w'), indent=2)
