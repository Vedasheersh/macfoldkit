# Descending-sort gradient on the tested MPS backend

On JAX/jaxlib 0.11.2 and jax-mps 0.11.0, descending batched sort followed by
a top-k mean produces the correct scalar loss but can lose the first row of
gradients. With a 32×32 contact-score matrix and k=8, CPU has 256 nonzero
derivatives; MPS has 248. The eight missing entries are all in row zero.
The contact entropy derivative without sorting agrees to relative L2 7.8e-7.
This accounts for most of the earlier approximately 3% full AF2 gradient error.

A minimal uncorrected reproducer needs only JAX:

```python
import jax
import jax.numpy as jnp
x = jnp.array([[0., 1., 2., 3.], [4., 5., 6., 7.]])
f = lambda x: jnp.sort(x, axis=-1, descending=True)[:, :2].mean()
print(jax.jit(jax.grad(f))(x))
# Expected on both devices: [[0, 0, .25, .25], [0, 0, .25, .25]].
# Tested MPS backend instead returns all zeros in the first row.
```

`check_contact_gradient.py` tests distinct and tied inputs, unbatched/2D/3D
shapes, weighted cotangents, k=1, k=2, full rows, and oversized slices. Of 40
cases, upstream passes 40/40 on CPU and 20/40 on this MPS backend. The adapter
passes 40/40 on both, with zero error against the analytical selected gradient
and exactly equal forward values. `check_contact_losses.py` additionally checks
complete within-binder and binder-target losses, including paratope/epitope
selection. These checks require the Mosaic runtime, not the lightweight package.

```bash
export PYTHONPATH="$MACFOLDKIT_HOME/sources/mosaic/src"
JAX_PLATFORMS=cpu "$MACFOLDKIT_HOME/runtimes/mosaic/bin/python" \
  experiments/mosaic_af2/check_contact_gradient.py
JAX_PLATFORMS=mps MLX_ENABLE_TF32=0 JAX_MPS_ASYNC_DISPATCH=0 \
  "$MACFOLDKIT_HOME/runtimes/mosaic/bin/python" \
  experiments/mosaic_af2/check_contact_gradient.py
```

For the complete contact losses, run `check_contact_losses.py cpu.npz` under
`JAX_PLATFORMS=cpu`, then `check_contact_losses.py mps.npz --reference cpu.npz`
under MPS with the same environment. The latter asserts corrected-gradient
relative L2 ≤1e-5 and reports the original derivative as a negative control.

## Workaround

`metal_losses.sorted_top_k_mean` keeps upstream's descending sort and mean in
the forward pass. Its custom VJP constructs a Boolean selection mask from a
stable ascending argsort's last k entries, then multiplies by the cotangent
divided by the selected count. This avoids scatter in reverse mode. It preserves
upstream's tie subgradient: descending value sort reverses ascending stable
order, preferring the highest original indices at a selection-boundary tie.
The separate `lax.top_k` paratope selection is retained, including its different
tie convention. No global JAX operations or installed libraries are patched.

This is an experiment-local reverse-mode workaround, not a general fix to the
backend, a forward-mode implementation, or a validation of every Mosaic loss.

## Source-level explanation

The inspected MLX 0.32.0 source in `mlx/backend/metal/kernels/indexing/scatter.h`
contains a matching signedness defect:

```cpp
auto upd_idx = ind_idx * static_cast<LocT>(upd_size) + gid.x;
```

`ind_idx` and `gid.x` are unsigned. When `LocT=int`, this expression infers an
unsigned offset. The following assignment of a negative-stride `elem_to_loc`
result therefore wraps to a large positive offset. Sort backward reverses a
padded cotangent; the JAX-MPS reverse handler preserves negative float strides.
For the observed case, row-zero offsets are negative, while later rows' total
offsets are nonnegative. The index-offset conditional in the same kernel also
mixes unsigned and signed operands.

This source analysis precisely fits the observed pattern. We have not rebuilt
the installed backend with a native fix, so it is not a tested native repair.
An upstream repair would explicitly type/cast both offsets as signed `LocT`;
a plugin workaround could materialize contiguous scatter inputs. Either needs
broader scatter regression coverage and measurement before replacing the wheel.

The older trajectory artifacts remain historical runs using the defective
derivative. Correcting the adapter does not retroactively validate them.
