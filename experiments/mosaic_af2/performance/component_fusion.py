"""Where does jax-mps lose to MLX, and does the backward path fuse?

Two questions, both open:

 1. A composed Evoformer-shaped block showed MLX 1.67x faster than jax-mps, while an
    isolated triangle multiplication tied. Somewhere between "one op" and "one block" the
    gap appears. This measures each component separately, then the composition, so the
    1.67x can be attributed rather than guessed at.

 2. Design needs gradients. A fused forward with an unfused backward would buy little, so
    every component is measured forward AND backward under both frameworks.

Design notes, learned from getting earlier measurements wrong:
 - Both implementations are checked to compute the same function before timing is trusted;
   an unverified pair of implementations is not a comparison.
 - Timing takes the minimum of several repeats after warmup, with explicit synchronisation
   (block_until_ready / mx.eval), since MLX is lazy and would otherwise time nothing.
 - Component shapes are the real AF2 ones for a 124-residue complex.
 - Results are written per component so a partial run is still usable.

Run under each environment separately:
  <mosaic python>  component_fusion.py jax out.json
  <mlx venv>       component_fusion.py mlx out.json
"""
import json
import sys
import time

N, C, HEADS, HEAD_DIM = 124, 128, 4, 32
REPEATS, WARMUP = 10, 3
SEED = 0


def bench(fn, sync):
    for _ in range(WARMUP):
        sync(fn())
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        sync(fn())
        times.append(time.perf_counter() - start)
    return min(times)


# ---------------------------------------------------------------- jax


def run_jax():
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(SEED)
    z = jnp.asarray(rng.standard_normal((N, N, C)).astype('float32'))
    w1 = jnp.asarray(rng.standard_normal((C, C)).astype('float32') * 0.05)
    w2 = jnp.asarray(rng.standard_normal((C, 4 * C)).astype('float32') * 0.05)
    w3 = jnp.asarray(rng.standard_normal((4 * C, C)).astype('float32') * 0.05)
    qkv = jnp.asarray(rng.standard_normal((N, N, HEADS, HEAD_DIM)).astype('float32'))
    bias = jnp.asarray(rng.standard_normal((N, 1, 1, N)).astype('float32'))
    sync = jax.block_until_ready

    def layernorm(z):
        mean = z.mean(-1, keepdims=True)
        var = z.var(-1, keepdims=True)
        return (z - mean) * jax.lax.rsqrt(var + 1e-5)

    def gating(z, w1):
        return jax.nn.sigmoid(z @ w1) * z

    def triangle_mult(z, w1):
        a = z @ w1
        return jnp.einsum('ikc,jkc->ijc', a, a)

    def triangle_attention(qkv, bias):
        q, k, v = qkv, qkv, qkv
        logits = jnp.einsum('bqhc,bkhc->bhqk', q, k) * (HEAD_DIM ** -0.5) + bias
        return jnp.einsum('bhqk,bkhc->bqhc', jax.nn.softmax(logits), v)

    def transition(z, w2, w3):
        return jax.nn.relu(z @ w2) @ w3

    def block(z, w1, w2, w3):
        n = layernorm(z)
        z = z + triangle_mult(n, w1)
        return z + transition(layernorm(z), w2, w3)

    components = [
        ('layernorm', layernorm, (z,)),
        ('gating', gating, (z, w1)),
        ('triangle_mult', triangle_mult, (z, w1)),
        ('triangle_attention', triangle_attention, (qkv, bias)),
        ('transition', transition, (z, w2, w3)),
        ('block_composed', block, (z, w1, w2, w3)),
    ]

    out = {'framework': 'jax-mps', 'version': jax.__version__,
           'device': str(jax.devices()[0]), 'components': []}
    for name, fn, args in components:
        row = {'component': name}
        try:
            forward = jax.jit(fn)
            row['forward_seconds'] = bench(lambda: forward(*args), sync)
            value = np.asarray(forward(*args), dtype='float64')
            row['forward_checksum'] = float(value.sum())
            row['forward_abs_mean'] = float(abs(value).mean())
            row['output_shape'] = list(value.shape)
            scalar = jax.jit(jax.grad(lambda *a: (fn(*a) ** 2).sum()))
            row['backward_seconds'] = bench(lambda: scalar(*args), sync)
            grad = np.asarray(scalar(*args), dtype='float64')
            row['grad_checksum'] = float(grad.sum())
            row['grad_abs_mean'] = float(abs(grad).mean())
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {str(error)[:160]}'
        out['components'].append(row)
        print(json.dumps(row), flush=True)
    return out


# ---------------------------------------------------------------- mlx


def run_mlx():
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(SEED)
    z = mx.array(rng.standard_normal((N, N, C)).astype('float32'))
    w1 = mx.array((rng.standard_normal((C, C)) * 0.05).astype('float32'))
    w2 = mx.array((rng.standard_normal((C, 4 * C)) * 0.05).astype('float32'))
    w3 = mx.array((rng.standard_normal((4 * C, C)) * 0.05).astype('float32'))
    qkv = mx.array(rng.standard_normal((N, N, HEADS, HEAD_DIM)).astype('float32'))
    bias = mx.array(rng.standard_normal((N, 1, 1, N)).astype('float32'))
    mx.eval(z, w1, w2, w3, qkv, bias)
    sync = mx.eval

    def layernorm(z):
        mean = mx.mean(z, axis=-1, keepdims=True)
        var = mx.var(z, axis=-1, keepdims=True)
        return (z - mean) * mx.rsqrt(var + 1e-5)

    def gating(z, w1):
        return mx.sigmoid(z @ w1) * z

    def triangle_mult(z, w1):
        a = z @ w1
        return mx.einsum('ikc,jkc->ijc', a, a)

    def triangle_attention(qkv, bias):
        q, k, v = qkv, qkv, qkv
        logits = mx.einsum('bqhc,bkhc->bhqk', q, k) * (HEAD_DIM ** -0.5) + bias
        return mx.einsum('bhqk,bkhc->bqhc', mx.softmax(logits, axis=-1), v)

    def transition(z, w2, w3):
        return mx.maximum(z @ w2, 0.0) @ w3

    def block(z, w1, w2, w3):
        n = layernorm(z)
        z = z + triangle_mult(n, w1)
        return z + transition(layernorm(z), w2, w3)

    components = [
        ('layernorm', layernorm, (z,)),
        ('gating', gating, (z, w1)),
        ('triangle_mult', triangle_mult, (z, w1)),
        ('triangle_attention', triangle_attention, (qkv, bias)),
        ('transition', transition, (z, w2, w3)),
        ('block_composed', block, (z, w1, w2, w3)),
    ]

    out = {'framework': 'mlx', 'version': mx.__version__,
           'device': str(mx.default_device()), 'components': []}
    for name, fn, args in components:
        row = {'component': name}
        try:
            row['forward_eager_seconds'] = bench(lambda: fn(*args), sync)
            forward = mx.compile(fn)
            row['forward_seconds'] = bench(lambda: forward(*args), sync)
            value = np.asarray(forward(*args), dtype='float64')
            row['forward_checksum'] = float(value.sum())
            row['forward_abs_mean'] = float(abs(value).mean())
            row['output_shape'] = list(value.shape)
            scalar = mx.compile(mx.grad(lambda *a: mx.sum(fn(*a) ** 2)))
            row['backward_seconds'] = bench(lambda: scalar(*args), sync)
            grad = np.asarray(scalar(*args), dtype='float64')
            row['grad_checksum'] = float(grad.sum())
            row['grad_abs_mean'] = float(abs(grad).mean())
            eager_grad = mx.grad(lambda *a: mx.sum(fn(*a) ** 2))
            row['backward_eager_seconds'] = bench(lambda: eager_grad(*args), sync)
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {str(error)[:160]}'
        out['components'].append(row)
        print(json.dumps(row), flush=True)
    return out


if __name__ == '__main__':
    result = {'jax': run_jax, 'mlx': run_mlx}[sys.argv[1]]()
    result['shapes'] = dict(N=N, C=C, heads=HEADS, head_dim=HEAD_DIM,
                            pair_tensor_mb=N * N * C * 4 / 1e6)
    if len(sys.argv) > 2:
        json.dump(result, open(sys.argv[2], 'w'), indent=2)
    print(json.dumps({'written': sys.argv[2] if len(sys.argv) > 2 else None}))
