"""OuterProductMean: does the contraction order matter at MSA depth 1?

AF2's comment says the current order is "faster", which it is when the MSA is deep. This
experiment runs a single sequence, where contracting the sequence axis first buys nothing
and materialises a large intermediate. Checks the two orders agree before comparing them.
"""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np

N_SEQ, N_RES, C_OUT, C_Z = 1, 124, 32, 128
rng = np.random.default_rng(0)
L = rng.standard_normal((N_SEQ, N_RES, C_OUT)).astype('float32')
R = rng.standard_normal((N_SEQ, N_RES, C_OUT)).astype('float32')
W = (rng.standard_normal((C_OUT, C_OUT, C_Z)) * 0.05).astype('float32')

def current(left, right, w):
    left = jnp.transpose(left, [0, 2, 1])
    act = jnp.einsum('acb,ade->dceb', left, right)
    act = jnp.einsum('dceb,cef->dbf', act, w)
    return jnp.transpose(act, [1, 0, 2])

def reordered(left, right, w):
    # Fold output_w into right_act first: the big [d,c,e,b] intermediate never exists.
    tmp = jnp.einsum('ade,cef->adcf', right, w)
    act = jnp.einsum('abc,adcf->bdf', left, tmp)
    return act

def bench(fn, *a):
    j = jax.jit(fn)
    jax.block_until_ready(j(*a))
    t = []
    for _ in range(20):
        s = time.perf_counter(); jax.block_until_ready(j(*a)); t.append(time.perf_counter()-s)
    return min(t), np.asarray(j(*a), dtype='float64')

l, r, w = map(jnp.asarray, (L, R, W))
t1, v1 = bench(current, l, r, w)
t2, v2 = bench(reordered, l, r, w)
out = {
    'backend': jax.default_backend(),
    'shapes': {'n_seq': N_SEQ, 'n_res': N_RES, 'c_out': C_OUT, 'c_z': C_Z},
    'current_seconds': t1, 'reordered_seconds': t2, 'speedup': t1/t2,
    'same_shape': list(v1.shape) == list(v2.shape),
    'max_abs_diff': float(np.abs(v1-v2).max()),
    'rel_l2': float(np.linalg.norm(v1-v2)/np.linalg.norm(v1)),
}
print(json.dumps(out, indent=1))
if len(sys.argv) > 1: json.dump(out, open(sys.argv[1], 'w'), indent=1)
