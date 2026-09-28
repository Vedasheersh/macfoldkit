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

Remaining work:

- Replace the CUDA-oriented installer with an isolated Mac environment.
- Bypass NVIDIA-only device discovery and multiprocessing; use one MPS worker.
- Disable CUDA accelerators and verify ordinary JAX attention gradients on MPS.
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
