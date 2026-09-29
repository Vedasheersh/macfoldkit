# BindCraft2 on Apple Silicon: feasibility, not support

Audited September 28, 2026 at upstream commit
[`e6d30f6`](https://github.com/PacesaLab/BindCraft2/tree/e6d30f6ea2e5bbc2f62ae7fa722f183da6c6c29f).
BindCraft2 is not an installed MacFoldKit backend in this release.

The source is promising for a Mac port: it declares JAX 0.11, matching the
JAX 0.11.2 / jax-mps 0.11.0 stack tested here. CUDA/cuEquivariance are optional;
attention and triangle multiplication have ordinary JAX implementations.
ProteinMPNN also uses JAX. Structural scoring uses Biotite/NumPy/JAX; no
PyRosetta dependency was found. This suggests a backend adapter and targeted
fixes, rather than a complete MLX rewrite.

CPU checks with JAX 0.11.2 passed for stock versus chunked attention forward and
backward (q, k, v and pair bias; maximum absolute discrepancy 3.73e-9), model and
filter imports, public ubiquitin structural scoring and rigid alignment. The
audit did not run any BindCraft2 model on MPS or complete a design campaign.
Full campaign imports also need its own complete dependency environment.

## Update, September 29 2026: first MPS execution

The audit above was CPU-only. BindCraft2's attention primitive has now been run on the
GPU, forward and backward, using the existing jax-mps 0.11.2 runtime. `experiments/
bindcraft2/mps_probe.py` reproduces it; BindCraft2 is read from a separate local checkout
and is still neither vendored nor redistributed.

| | CPU | MPS | agreement |
|---|---|---|---|
| `attend` stock, forward | 9162.173828 | 9162.173828 | bit-identical |
| `attend` chunked, forward | 9162.173828 | 9162.173828 | bit-identical |
| gradient norms, all five inputs | — | — | 2e-08 to 7e-08 relative |
| zero-gradient fractions | all 0.0 | all 0.0 | identical |
| backward wall time, stock | 0.0519 s | 0.0089 s | **5.8x faster** |
| backward wall time, chunked | 0.1289 s | 0.0213 s | **6.0x faster** |

Three things this establishes. The dependency gap is small: jax-mps and dm-haiku are
already installed, and only `optax` and `biotite` had to be added. `supported_attention_
backend('auto')` correctly selects `stock` on this machine and `cuequivariance_available()`
is false, so the CUDA path declines cleanly rather than failing. And the descending-sort
backward defect that required a custom VJP in the Mosaic experiment does **not** affect
this path: zero-gradient fractions match CPU exactly.

`fused_triangle_multiplicative_update` raises `ModuleNotFoundError: cuequivariance_jax`,
which is correct — it is the CUDA-only fused variant, and BindCraft2 carries an ordinary
JAX triangle multiplication to fall back to. That fallback has not been exercised yet.

Source inspection also revises one expected hazard downward: the contact ranking at
`bindcraft/loss.py:412` is wrapped in `jax.lax.stop_gradient`, so no gradient flows through
that `argsort`. The remaining sort-shaped exposure is `top_k=30` in the ProteinMPNN k-nearest
-neighbour graph (`bindcraft/mpnn/modules.py`), and only if gradients reach it.

This is one primitive, not the model. Nothing below is retracted.

Remaining work:

- Replace the CUDA-oriented installer with an isolated Mac environment.
- Bypass NVIDIA-only device discovery and multiprocessing; use one MPS worker.
  CUDA coupling is confined to five files: `design_workers.py`, `selfcheck.py`, `cli.py`,
  `af/accel.py` and `af/alphafold/model/modules.py`.
- Exercise the non-fused triangle multiplication fallback on MPS.
- Check whether gradients reach the ProteinMPNN `top_k` graph, and if so gate it.
- Compare full AF2 confidence/structure losses and raw sequence gradients against
  CPU, including ProteinMPNN's sorting/sampling operations.
- Run one bounded trajectory, sequence redesign and unchanged structural filters;
  measure memory and iteration time before attempting a campaign.

A single successful ColabFold prediction does not establish full design support:
sequence optimization executes repeated forward/backward passes. Mosaic's own
full-gradient experiment exposed a single-iteration scan slowdown and a roughly
3% CPU/MPS gradient discrepancy, so those paths deserve explicit testing here.

The upstream [license](https://github.com/PacesaLab/BindCraft2/blob/e6d30f6ea2e5bbc2f62ae7fa722f183da6c6c29f/LICENSE)
is named **BindCraft2 Source-Available License (Hosting-Restricted)** and states
that it is not OSI-approved. Its terms differ from MacFoldKit's Apache-2.0 code.
An integration should download it separately, preserve its notices and explain
its terms; this audit does not relicense or redistribute BindCraft2.

Relevant source: [dependencies](https://github.com/PacesaLab/BindCraft2/blob/e6d30f6ea2e5bbc2f62ae7fa722f183da6c6c29f/pyproject.toml),
[attention](https://github.com/PacesaLab/BindCraft2/blob/e6d30f6ea2e5bbc2f62ae7fa722f183da6c6c29f/bindcraft/af/accel.py),
[AF2 gradient pipeline](https://github.com/PacesaLab/BindCraft2/blob/e6d30f6ea2e5bbc2f62ae7fa722f183da6c6c29f/bindcraft/af2.py),
[worker discovery](https://github.com/PacesaLab/BindCraft2/blob/e6d30f6ea2e5bbc2f62ae7fa722f183da6c6c29f/bindcraft/design_workers.py).
