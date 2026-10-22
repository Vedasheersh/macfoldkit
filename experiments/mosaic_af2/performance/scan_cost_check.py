"""Does XLA cost analysis count a scan body once, or once per trip?

If once, every FLOP and byte figure derived from cost_analysis.py is undercounted by the
scan trip count -- and AF2's Evoformer is a 48-iteration hk.scan.
"""
import json
import jax, jax.numpy as jnp

def body(c, _):
    return jnp.tanh(c @ c), None

def scan_version(x, count):
    return jax.lax.scan(body, x, None, length=count)[0]

def loop_version(x, count):
    for _ in range(count):
        x = jnp.tanh(x @ x)
    return x

x = jnp.eye(64, dtype=jnp.float32) * 0.5
rows = []
for count in [1, 2, 4, 8, 16, 48]:
    row = {'count': count}
    for name, fn in [('scan', scan_version), ('loop', loop_version)]:
        c = jax.jit(lambda v, f=fn, n=count: f(v, n)).lower(x).compile().cost_analysis()
        c = c[0] if isinstance(c, list) else c
        row[name + '_flops'] = float(c.get('flops', 0))
        row[name + '_bytes'] = float(c.get('bytes accessed', 0))
    g = jax.jit(jax.grad(lambda v, n=count: scan_version(v, n).sum())).lower(x).compile().cost_analysis()
    g = g[0] if isinstance(g, list) else g
    row['grad_scan_flops'] = float(g.get('flops', 0))
    rows.append(row)
print(json.dumps(rows, indent=1))
