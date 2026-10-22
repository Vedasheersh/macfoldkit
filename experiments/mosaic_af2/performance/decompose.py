"""Where inside the block does jax-mps lose to MLX? Cumulative decomposition.

Two realistic composed blocks disagreed: one showed MLX 1.45x, the other tied. Building
the disagreeing block up one stage at a time localises where the gap opens instead of
attributing it by guesswork. Checksums are printed so the two frameworks are known to be
computing the same thing at every stage.
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
    return min(t)

def run_jax():
    import jax, jax.numpy as jnp
    z, wl, wg = map(jnp.asarray, (Z, WL, WG))
    def ln(z):
        m = z.mean(-1, keepdims=True); v = z.var(-1, keepdims=True)
        return (z - m) * jax.lax.rsqrt(v + 1e-5)
    stages = {
        's1_layernorm':      lambda z, wl, wg: ln(z),
        's2_one_matmul':     lambda z, wl, wg: ln(z) @ wl,
        's3_two_matmul':     lambda z, wl, wg: (ln(z) @ wg) + (ln(z) @ wl),
        's4_gating':         lambda z, wl, wg: jax.nn.sigmoid(ln(z) @ wg) * (ln(z) @ wl),
        's5_gating_einsum':  lambda z, wl, wg: jax_s5(z, wl, wg),
        's6_full_block':     lambda z, wl, wg: z + jax_s5(z, wl, wg),
    }
    def jax_s5(z, wl, wg):
        a = jax.nn.sigmoid(ln(z) @ wg) * (ln(z) @ wl)   # bound once, as in the real block
        return jnp.einsum('ikc,jkc->ijc', a, a)
    out = {'framework': 'jax-mps', 'stages': []}
    for name, fn in stages.items():
        j = jax.jit(fn)
        s = bench(lambda: j(z, wl, wg), jax.block_until_ready)
        out['stages'].append({'stage': name, 'seconds': s,
            'checksum': float(np.asarray(j(z, wl, wg), dtype='float64').sum())})
        print(json.dumps(out['stages'][-1]), flush=True)
    return out

def run_mlx():
    import mlx.core as mx
    z, wl, wg = map(mx.array, (Z, WL, WG)); mx.eval(z, wl, wg)
    def ln(z):
        m = mx.mean(z, axis=-1, keepdims=True); v = mx.var(z, axis=-1, keepdims=True)
        return (z - m) * mx.rsqrt(v + 1e-5)
    stages = {
        's1_layernorm':      lambda z, wl, wg: ln(z),
        's2_one_matmul':     lambda z, wl, wg: ln(z) @ wl,
        's3_two_matmul':     lambda z, wl, wg: (ln(z) @ wg) + (ln(z) @ wl),
        's4_gating':         lambda z, wl, wg: mx.sigmoid(ln(z) @ wg) * (ln(z) @ wl),
        's5_gating_einsum':  lambda z, wl, wg: mlx_s5(z, wl, wg),
        's6_full_block':     lambda z, wl, wg: z + mlx_s5(z, wl, wg),
    }
    def mlx_s5(z, wl, wg):
        a = mx.sigmoid(ln(z) @ wg) * (ln(z) @ wl)       # bound once, as in the real block
        return mx.einsum('ikc,jkc->ijc', a, a)
    out = {'framework': 'mlx', 'stages': []}
    for name, fn in stages.items():
        c = mx.compile(fn)
        s = bench(lambda: c(z, wl, wg), mx.eval)
        out['stages'].append({'stage': name, 'seconds': s,
            'checksum': float(np.asarray(c(z, wl, wg), dtype='float64').sum())})
        print(json.dumps(out['stages'][-1]), flush=True)
    return out

if __name__ == '__main__':
    r = {'jax': run_jax, 'mlx': run_mlx}[sys.argv[1]]()
    if len(sys.argv) > 2: json.dump(r, open(sys.argv[2], 'w'), indent=2)
