"""Does jax-mps fusion hold up on an Evoformer-shaped block, not just an elementwise chain?

The simple chain showed jax-mps within 3% of mx.compile. Real Evoformer blocks mix
reductions, transposes, batched matmuls and gating, which are exactly the boundaries
fusion tends to break at. If jax-mps degrades here relative to MLX, that localizes the
27x gap to the real model's op graph.
"""
import json, sys, time
N, C = 124, 128
REPEATS, WARMUP = 10, 3

def bench(fn, sync):
    for _ in range(WARMUP): sync(fn())
    t = []
    for _ in range(REPEATS):
        s = time.perf_counter(); sync(fn()); t.append(time.perf_counter() - s)
    return min(t)

def run_jax():
    import jax, jax.numpy as jnp
    k = jax.random.key(0)
    z = jax.random.normal(k, (N, N, C), jnp.float32)
    wl = jax.random.normal(k, (C, C), jnp.float32)
    wg = jax.random.normal(k, (C, C), jnp.float32)
    sync = jax.block_until_ready
    def block(z, wl, wg):
        m = z.mean(-1, keepdims=True); v = z.var(-1, keepdims=True)
        n = (z - m) * jax.lax.rsqrt(v + 1e-5)
        a = jax.nn.sigmoid(n @ wg) * (n @ wl)
        t = jnp.einsum('ikc,jkc->ijc', a, a)
        return z + t
    j = jax.jit(block)
    return {'backend': 'jax-mps', 'jit_seconds': bench(lambda: j(z, wl, wg), sync)}

def run_mlx():
    import mlx.core as mx
    z = mx.random.normal((N, N, C), dtype=mx.float32)
    wl = mx.random.normal((C, C), dtype=mx.float32)
    wg = mx.random.normal((C, C), dtype=mx.float32)
    mx.eval(z, wl, wg)
    def block(z, wl, wg):
        m = mx.mean(z, axis=-1, keepdims=True); v = mx.var(z, axis=-1, keepdims=True)
        n = (z - m) * mx.rsqrt(v + 1e-5)
        a = mx.sigmoid(n @ wg) * (n @ wl)
        t = mx.einsum('ikc,jkc->ijc', a, a)
        return z + t
    c = mx.compile(block)
    return {'backend': 'mlx', 'eager_seconds': bench(lambda: block(z, wl, wg), mx.eval),
            'compiled_seconds': bench(lambda: c(z, wl, wg), mx.eval)}

if __name__ == '__main__':
    r = {'jax': run_jax, 'mlx': run_mlx}[sys.argv[1]]()
    print(json.dumps(r, indent=2))
    if len(sys.argv) > 2: json.dump(r, open(sys.argv[2], 'w'), indent=2)
