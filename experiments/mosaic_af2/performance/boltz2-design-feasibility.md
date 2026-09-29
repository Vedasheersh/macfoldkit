# Mosaic design with Boltz2 instead of AF2

Investigated September 29 2026. **The backend exists and now imports on this machine. It is
not yet run, and the expected cost is roughly an order of magnitude worse than AF2.**

## It is a first-class Mosaic backend

Mosaic ships `mosaic/models/boltz2.py` and `mosaic/losses/boltz2.py`, exposing `Boltz2`,
`Boltz2Loss` and `MultiSampleBoltz2Loss`, with `binder_features`, `build_loss`,
`build_multisample_loss`, `model_output` and `predict` — the same shape of interface the AF2
adapter uses. It bridges PyTorch to JAX through `joltz.from_torch(torch_model)`.

So switching the design step from AF2 to Boltz2 is not a porting project. It is an
environment and cost question.

## Environment: assembled, and it imports

The obstacle was that the pieces live in separate isolated runtimes: the mosaic runtime has
jax-mps 0.11.2, equinox and torch but no `boltz`/`joltz`; the boltz runtime has `boltz` but
no jax. They can be combined by path rather than by reinstalling anything:

```
PYTHONPATH="$MACFOLDKIT_HOME/sources/mosaic/src:\
$MACFOLDKIT_HOME/sources/boltz/src:\
<workspace>/work/mosaic-joltz-source/src:\
<workspace>/work/mosaic-boltz2-deps"
python = "$MACFOLDKIT_HOME/runtimes/mosaic/bin/python"
```

`work/mosaic-boltz2-deps` is a `uv pip install --target` side directory holding the
transitive dependencies the mosaic runtime lacked, about 1.0 GB:

> click, pytorch_lightning, fairscale, rdkit, numba, llvmlite, mashumaro, requests, urllib3,
> pandas, python-dateutil, pytz, tzdata, six, chembl_structure_pipeline, scikit-learn,
> joblib, threadpoolctl, cloudpickle, narwhals, ihm, modelcif, einx, frozendict

Install them with `--no-deps` and **delete the bundled `torch`** from the target directory:
`uv` pulls torch 2.14 as a `pytorch_lightning` dependency, and because PYTHONPATH precedes
site-packages it would shadow the runtime's own torch.

With that, `import mosaic.models.boltz2` succeeds.

## It runs. Measured on CPU

A Mosaic design gradient through Boltz2 now executes. On CPU, 48-residue binder against
76-residue ubiquitin, `sampling_steps=1`:

| stage | seconds |
|---|---|
| model load, torch checkpoint through `joltz.from_torch` | 11.1 |
| feature build | 0.51 |
| **warm gradient step** | **8.73** |

Loss -0.6201, gradient finite. AF2's CPU baseline for the same complex is 6.51 s per step,
and AF2 gets 3.1x from the GPU, so if Boltz2 scales similarly a `sampling_steps=1` step
would land near 2.8 s on MPS. The default is 25, so the scaling with sampling steps is what
actually decides affordability and is being measured next.

### The checkpoint was already here

`boltz2_conf.ckpt` (2,286,561,469 bytes) was already in `work/boltz-cache/`, alongside
`boltz2_aff.ckpt`, `mols/` and `mols.tar`. It was re-downloaded before that was checked, and
the redundant copy has been deleted. The correct mechanism is the `MOSAIC_CACHE_DIR`
environment variable that `mosaic/cache.py` reads:

```
export MOSAIC_CACHE_DIR=<workspace>/work/mosaic-cache   # containing boltz -> work/boltz-cache
```

Without it, `resolve_cache` points at `~/.cache/mosaic/boltz` and both the checkpoint and
the 45,227-file CCD molecule directory would be downloaded again.

### Two compatibility fixes were required

1. **`torch.load` weights_only.** torch >= 2.6 defaults `weights_only=True`, and the
   checkpoint carries an `omegaconf.DictConfig`, so lightning's loader refuses it.
   Allowlisting the config classes via `add_safe_globals` did not suffice. The probe forces
   `weights_only=False` for this one file — a scoped trust decision about an artifact
   fetched from Boltz's own gateway with a verified length, not a general default.
2. **antlr4 pin.** `omegaconf` requires `antlr4-python3-runtime==4.9.*`; installing with
   `--no-deps` pulls a newer runtime and the grammar fails with
   `Could not deserialize ATN with version 3 (expected 4)`.

## Measured on MPS. The earlier estimate in this document was wrong

| sampling_steps | CPU warm | MPS warm | speedup | MPS reported peak | status |
|---|---|---|---|---|---|
| 1 | 8.75 s | — | — | — | **fails**, jax-mps SVD bug |
| 5 | 9.02 s | 3.02 s | 2.98x | 44.4 GB | finite |
| **25 (Mosaic default)** | 10.74 s | **3.68 s** | 2.92x | 44.4 GB | finite |

Against the measured AF2 baseline of **2.08 s/step and 2.66 GB**, a Boltz2 design step at
the default 25 sampling steps costs **1.77x the time** — not the 7-10x this document
previously estimated. A 125-step design would be about **7.7 minutes**, against 4.4 for AF2.
The MPS speedup, 2.9-3.0x, matches AF2's 3.1x, so Boltz2 is not unusually badly served by
this backend.

**Diffusion is not the cost driver, which is what the estimate got wrong.** On CPU the
marginal cost is 0.083 s per sampling step; going from 1 to 25 steps adds only 2.07 s of a
10.74 s total. The implied trunk-and-fixed cost is 8.67 s, about 81% of the step. Reducing
`sampling_steps` is therefore *not* the affordability lever it looked like.

### Two real problems

**1. Memory.** The reported peak is 44.4 GB against a 24.48 GB device limit. It is identical
at 5 and 25 sampling steps, which suggests it may be an allocator high-water mark rather
than live residency, so the figure needs interpreting before it is quoted as a hard number —
process RSS was not recorded and should be. Either way it is an order of magnitude above
AF2's 2.66 GB and is the thing most likely to prevent a real campaign, especially at EGFR
scale where AF2 already reaches 5.42 GB.

**2. A jax-mps backend bug at `sampling_steps=1`:**

```
JaxRuntimeError: INTERNAL: Output count mismatch: expected 5, got 0
  (eval: svd_impl: sgesvdx_ failed with code -4)
```

An SVD in the MLX/MPS backend fails with a LAPACK illegal-argument code. It affects only
`sampling_steps=1`; 5 and 25 both run. This is a backend defect, not a Boltz2 or Mosaic one,
and is worth reporting upstream.

## The estimate that this superseded

Retained for honesty; the measurements above replace it.

| | AF2 design step | Boltz2 design step |
|---|---|---|
| forward structure | one pass, no diffusion | trunk with recycling **plus 25 diffusion sampling steps** (`sampling_steps=25` default) |
| measured, 124-residue complex | **2.08 s/step** | not yet measured |
| Boltz2 inference for comparison | — | ~4.7 s per prediction, diffusion included |
| estimated gradient step | — | **~15–20 s** (inference x 3–4 for backward) |
| 125-step design | **4.4 min** | **~35–40 min estimated** |

Differentiating through 25 diffusion steps is the cost driver. Memory is the other risk:
the AF2 design gradient peaks at 2.66 GB of 24.48 GB, and a backward pass through a
diffusion trajectory could be far larger. Whether it fits is unknown and is the first thing
to check.

The performance study in [README.md](README.md) applies here too: there is no MLX headroom
to recover on this path either, so a slower model simply costs more.

## The methodological cost, which is easy to miss

Every result in `ubiquitin/`, `ubiquitin/v2/` and `ubiquitin/v3/` rests on AF2 doing the
design and Boltz2 doing an **independent** refold. That independence is what makes the
screen mean anything at all.

Designing with Boltz2 destroys it. The validator would have to become AF2, or Protenix
(also present in this workspace), and every published threshold would need re-deriving
against the new validator. This is a larger change than swapping a model.

## What would settle it

1. Download `boltz2_conf.ckpt` and load it through `joltz.from_torch`.
2. One probe-only gradient at 124 residues: measure seconds per step and peak memory.
3. Compare against 2.08 s and 2.66 GB.

If a step costs 15–20 s and fits in memory, Boltz2 design is usable but expensive and
belongs behind a flag. If it does not fit, the answer is no on this hardware.
