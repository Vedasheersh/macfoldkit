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

### Memory: there is no problem to optimize. The 44.4 GB was a measurement artifact

An earlier revision of this document reported a 44.4 GB peak against a 24.48 GB device
limit and called it the likely campaign blocker. That figure was wrong, and the tell was
already visible: it was byte-identical at 5 and 25 sampling steps and exceeded the device
limit, which is internally impossible.

The cause was the sweep running `sampling_steps` 1, 5 and 25 **in one process**.
`peak_bytes_in_use` is a process-lifetime high-water mark, so it accumulated across a failed
run and three separate compilations. Measured one config per process:

| config | tokens | atoms | sampling_steps | warm gradient | device peak | pool bytes |
|---|---|---|---|---|---|---|
| ubiquitin + 48-mer | 124 | 864 | 5 | 2.91 s | **5.25 GB** | 10.21 GB |
| ubiquitin + 48-mer | 124 | 864 | 25 | 3.57 s | **5.25 GB** | 10.21 GB |
| EGFR domain III + 60-mer | 231 | — | 25 | 12.92 s | **10.17 GB** | 18.10 GB |
| EGFR domain III + 150-mer | 321 | — | 25 | 30.41 s | **18.30 GB** | 18.10 GB |

Device limit 24.48 GB. AF2 for comparison: 2.66 GB at 124 tokens, 5.42 GB at 231.

Boltz2 design costs roughly **2x AF2's memory and 1.4x its time**, and fits comfortably at
both ubiquitin and EGFR scale. Nothing needs optimizing at the sizes that matter.

**`sampling_steps` has no effect on memory at all** — 5.25 GB at both 5 and 25. Joltz runs
the diffusion loop as a `jax.lax.scan` under `@jax.checkpoint`, so activations do not
accumulate across sampling steps. Joltz already applies `@jax.checkpoint` at ten or more
sites, and `boltz2_trunk` wraps each recycling iteration in `jax.lax.stop_gradient`. The
obvious memory levers are already pulled.

### Where the real ceiling is, and it is time not memory

Memory scales about `N^1.7` in tokens (231 -> 321 tokens, a 1.39x increase, took 10.17 ->
18.30 GB, 1.80x). Extrapolating to the 24.48 GB limit puts the memory ceiling near **370
tokens**.

Time scales worse, about `N^2.6` (12.92 -> 30.41 s over the same range). At 321 tokens a
single gradient step is 30.41 s, so a 125-step design is **about 63 minutes**. Time becomes
prohibitive well before memory does, and the performance study in [README.md](README.md)
found no backend headroom to recover — all three backends sit at the same roofline.

So the practical envelope for Boltz2 design on this machine is roughly **up to EGFR scale**:
231 tokens at 12.92 s/step is about 27 minutes for a 125-step design, against 19 minutes for
AF2. Beyond ~300 tokens it stops being sensible on time grounds, not memory grounds.

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
