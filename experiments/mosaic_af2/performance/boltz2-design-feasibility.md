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

## Remaining cost question

This is the part that decides whether to pursue it.

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
