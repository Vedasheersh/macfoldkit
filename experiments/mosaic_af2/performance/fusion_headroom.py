"""Is jax-mps leaving operator fusion on the table?

jax-mps is MLX-backed, and MLX exposes its own fusing compiler via mx.compile. If a JAX
primitive chain is translated one-op-at-a-time into MLX, every intermediate round-trips
through memory, and on a bandwidth-limited machine that is exactly the ~9%-of-peak
behaviour the AF2 design gradient shows.

Times an Evoformer-shaped elementwise chain -- layer norm, gate, scale, residual -- over a
pair tensor, under jax-mps, MLX eager, and MLX compiled. The gap between jax-mps and MLX
compiled is the fusion headroom a better backend translation could recover.

Run under each environment separately:
  <mosaic python>  fusion_headroom.py jax out.json
  <mlx venv>       fusion_headroom.py mlx out.json
"""
import json
import sys
import time

N, C = 124, 128          # AF2 pair representation for the ubiquitin complex
REPEATS, WARMUP = 20, 3


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
    key = jax.random.key(0)
    z = jax.random.normal(key, (N, N, C), jnp.float32)
    g = jax.random.normal(key, (N, N, C), jnp.float32)
    scale = jnp.ones((C,), jnp.float32)
    offset = jnp.zeros((C,), jnp.float32)
    sync = jax.block_until_ready

    def chain(z, g, scale, offset):
        mean = z.mean(-1, keepdims=True)
        var = z.var(-1, keepdims=True)
        normed = (z - mean) * jax.lax.rsqrt(var + 1e-5) * scale + offset
        gate = jax.nn.sigmoid(g)
        return z + normed * gate

    jitted = jax.jit(chain)
    out = {'backend': 'jax-mps', 'version': jax.__version__,
           'device': str(jax.devices()[0]),
           'jit_seconds': bench(lambda: jitted(z, g, scale, offset), sync)}
    # Unjitted, to show what per-primitive dispatch costs without any fusion at all.
    out['eager_seconds'] = bench(lambda: chain(z, g, scale, offset), sync)
    return out


def run_mlx():
    import mlx.core as mx
    z = mx.random.normal((N, N, C), dtype=mx.float32)
    g = mx.random.normal((N, N, C), dtype=mx.float32)
    scale = mx.ones((C,), dtype=mx.float32)
    offset = mx.zeros((C,), dtype=mx.float32)
    mx.eval(z, g, scale, offset)
    sync = mx.eval

    def chain(z, g, scale, offset):
        mean = mx.mean(z, axis=-1, keepdims=True)
        var = mx.var(z, axis=-1, keepdims=True)
        normed = (z - mean) * mx.rsqrt(var + 1e-5) * scale + offset
        gate = mx.sigmoid(g)
        return z + normed * gate

    compiled = mx.compile(chain)
    return {'backend': 'mlx', 'version': mx.__version__,
            'device': str(mx.default_device()),
            'eager_seconds': bench(lambda: chain(z, g, scale, offset), sync),
            'compiled_seconds': bench(lambda: compiled(z, g, scale, offset), sync)}


if __name__ == '__main__':
    result = {'jax': run_jax, 'mlx': run_mlx}[sys.argv[1]]()
    result['tensor_mb'] = N * N * C * 4 / 1e6
    print(json.dumps(result, indent=2))
    if len(sys.argv) > 2:
        json.dump(result, open(sys.argv[2], 'w'), indent=2)
