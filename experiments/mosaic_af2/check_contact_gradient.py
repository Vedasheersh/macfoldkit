"""Analytic regression for the MPS sort VJP; run on CPU and MPS separately."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from metal_losses import sorted_top_k_mean


def main():
    rows = []
    for shape in [(4,), (2, 4), (3, 8), (32, 32), (2, 3, 8)]:
        for tied in [False, True]:
            x = np.random.default_rng(7).normal(size=shape).astype('float32')
            if tied:
                x = np.round(x)
            weight = np.arange(1, np.prod(shape[:-1]) + 1, dtype='float32').reshape(shape[:-1])
            for k in [1, 2, shape[-1], shape[-1] + 1]:
                count = min(k, shape[-1])
                expected = np.zeros_like(x)
                selected = np.argsort(x, axis=-1, kind='stable')[..., -count:]
                np.put_along_axis(expected, selected,
                    np.broadcast_to(weight[..., None] / count, selected.shape), axis=-1)
                values = {}
                for name, fn in [('upstream', lambda v: jnp.sort(v, axis=-1, descending=True)[..., :k].mean(-1)),
                                 ('corrected', lambda v: sorted_top_k_mean(v, k))]:
                    loss = lambda v: (fn(v) * jnp.asarray(weight)).sum()
                    value, grad = jax.block_until_ready(jax.jit(jax.value_and_grad(loss))(jnp.asarray(x)))
                    err = float(np.max(np.abs(np.asarray(grad) - expected)))
                    values[name] = float(value)
                    rows.append(dict(shape=shape, tied=tied, k=k, method=name,
                        value=float(value), max_gradient_error=err,
                        nonzeros=int(np.count_nonzero(grad))))
                    if name == 'corrected' or jax.default_backend() == 'cpu':
                        np.testing.assert_allclose(grad, expected, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(values['upstream'], values['corrected'], rtol=0, atol=0)
    print(json.dumps(dict(platform=jax.default_backend(), cases=rows), indent=2))


if __name__ == '__main__':
    main()
