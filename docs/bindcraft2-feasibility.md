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

## Update, September 29 2026: MPS execution

The audit above was CPU-only. BindCraft2 primitives, a full AF2 forward pass and an AF2
backward pass have now been run on the GPU. Probes and JSON results are in
`experiments/bindcraft2/`; `results-summary.json` consolidates them. BindCraft2 is read from
a separate local checkout, which was left untouched (still at `e6d30f6`, clean working
tree). No BindCraft2 source is vendored here.

The repository's 0.5% relative gate is used throughout.

### Attention: PASS

`accel.attend`, stock and chunked, forward and backward. CPU vs MPS forward values
bit-identical; gradient norms for all five inputs agree to 2e-08..7e-08; zero-gradient
fractions identical. Backward 5.8x (stock) and 6.0x (chunked) faster than CPU.

### Non-fused triangle multiplication: PASS

The CUDA kernel is reachable only through the guard at `modules.py:1050`
(`gc.get('use_cueq', False) and accel.cuequivariance_available()`); `use_cueq` is not a key
in the stock global config, so the ordinary-JAX fallback runs unconditionally. Both JAX
paths (`_fused_triangle_multiplication`'s fallback and `_triangle_multiplication`) were run
across outgoing/incoming and full/padded masks, forward and backward.

| size | worst relative | zero-gradient rows | MPS backward |
|---|---|---|---|
| N=48, C=32 | 5.83e-07 | none, identical | 0.47x–1.73x (dispatch-bound) |
| N=192, C=128 (evoformer scale) | 8.06e-07 | none, identical | **2.16x–6.82x** |

No scatter/sort backward signature anywhere.

### Full AF2 forward: PASS in float32, FAILS at BindCraft2's bfloat16 default

Two things the audit expected to be blockers were not. **The ColabFold weights work as-is**:
`get_model_haiku_params` probes `<data_dir>/params/params_<model>.npz` and
`work/colabfold-weights/params/` already holds every `params_model_{1..5}{,_ptm,_multimer_v3}.npz`;
93,237,338 parameters load in 0.22 s with no conversion. **No CUDA bypass was needed**:
constructing `bindcraft.af2.AlphaFoldDesignModel` directly never imports `cli.py`,
`design_workers.py` or `selfcheck.py`.

| comparison | model_1_ptm | multimer_v3 | gate |
|---|---|---|---|
| CPU fp32 vs MPS fp32 | 5.21e-04 | 4.68e-04 | **PASS** |
| CPU bf16 vs MPS bf16 (the default) | 3.43e-01 | 1.44e+00 | FAIL |
| CPU bf16 vs CPU fp32 (no backend change) | 3.97e-01 | 1.38e+00 | FAIL |

The third row is the important one: **the bfloat16 failure is not an MPS defect.** Changing
only precision on a single backend moves the answer as much as changing the backend does.
AF2 is bf16-chaotic on this input (random sequence, no MSA, pLDDT ~0.31) and the structure
module amplifies it. The float32 "worst" is entirely the `astype(jnp.float16)` on returned
coordinates at `af2.py:345` — one fp16 ULP, 0.0156 A. The metrics agree far tighter: pLDDT
2.4e-05, PAE 6.5e-07, pTM 1.9e-07, distogram 5.9e-07.

**Actionable:** `_alphafold_runner` (`af2.py:252`) hardcodes `bfloat16 = True`. On MPS that is
*slower* than float32 (0.633 s vs 0.573 s for a 100-residue multimer) while on CPU it is
faster (1.507 s vs 2.155 s). A Mac adapter should turn it off and gain both speed and
numerical agreement.

### AF2 backward: PASSES at a realistic input. The earlier failure was the test, not MPS

First measured with `Protein.empty` -- `0.01 * normal` logits, an essentially uniform
sequence that AF2 predicts at pLDDT 0.31. CPU and MPS disagreed by **1.756%** relative L2
against a 0.5% gate, and that was recorded here as the blocker for a design campaign.

It was not. Every individual primitive had already passed cleanly (attention 7e-08, triangle
multiplication 8e-07), so the 1.756% was accumulation over 48 blocks, and a near-uniform
sequence at pLDDT 0.31 is exactly the ill-conditioned regime where such accumulation is
worst. Re-running the identical gradient path with one variable changed -- a real ubiquitin
sequence in place of the noise, same flags, same padding, same `num_recycle=0`:

| input | pLDDT | gradient relative L2 | verdict |
|---|---|---|---|
| `Protein.empty` (uniform noise) | 0.31 | 1.756% | fails |
| real ubiquitin sequence | 0.54 | **0.288%** | **passes** |

Cosine similarity 0.9999991, sign agreement 100%, forward pLDDT agreeing to 0.0034% and pTM
to 7 significant figures. `af2_gradient_probe_real.py` reproduces it; `afg-real-*.json` hold
the numbers.

**So the AF2 design gradient is not blocked on MPS.** It passes this repository's own gate at
a realistic input, which is the only kind a design campaign encounters.

Scope, because one measurement is not a port: a single sequence, `model_1_ptm`, one loss
scalar (mean pLDDT), `num_recycle=0`, L=76, no MSA and no template. 0.288% passes but is
looser than the 0.026-0.159% Mosaic's full AF2 gradient reaches on a real target, and a real
trajectory would run with recycling and templates, which were not measured. The multimer
model and BindCraft2's actual composite loss registry remain untested.

### ProteinMPNN k-nearest-neighbour graph: not a hazard

An earlier revision of this document stated that "the remaining sort-shaped exposure is
`top_k=30` in the ProteinMPNN k-nearest-neighbour graph". That was wrong on three counts and
is retracted:

1. The primitive is `jax.lax.approx_min_k(D_masked, k, reduction_dimension=-1)[1]`
   (`mpnn/modules.py:200`), not `top_k` and not `argsort`. `top_k` is only a keyword name.
   The `[1]` keeps the int32 indices and discards the values, so the selection has no
   tangent space at all.
2. k is **48**, not 30. 30 is only `ProteinFeatures.__init__`'s default; `ProteinMPNN.__init__`
   overrides it from the checkpoint's `num_edges`, and the shipped `v_48_020.npz` gives 48.
3. **ProteinMPNN is never inside a gradient.** The only two gradient sites in `bindcraft/`
   are `af2.py:378` and `protein.py:146`; neither touches MPNN.

Measured: gradient through the indices is exactly zero on both backends, and jax-mps returns
the *exact* nearest neighbours — index arrays bit-identical to CPU and matching a brute-force
argsort reference at 1.0 on both. So the approximate primitive is not a forward-correctness
hazard on MPS either. A counterfactual gradient through `ProteinFeatures` agrees to 5.66e-06.

`loss.py:412`'s `jax.lax.stop_gradient(jnp.argsort(...))` was re-verified. There is no
sort-shaped gradient exposure left in BindCraft2.

### A full trajectory now runs on Apple Silicon

`run_trajectory` (trajectory.py:278), BindCraft2's real four-stage design machine, completed
on MPS. Driver, launcher and settings are in `experiments/bindcraft2/`
(`run_trajectory_mac.py`, `run-mac.sh`, `settings-ubiquitin-smoke.json`); the checkout is
still untouched and nothing is vendored.

Target ubiquitin (76 residues, chain A), de novo binder of 60, one AF2 preset
(`model_1_multimer_v3`), seed 7:

| | |
|---|---|
| device | MPS:0 |
| stages completed | screen, refine, anneal, harden — all four, `failure: None` |
| gradient rounds | 8 (3/2/2/1), against the default 125 |
| model load | 0.2 s |
| trajectory | 362.8 s |
| i_pTM over the run | 0.08 -> 0.13 |
| output | a 60-residue binder plus a per-round loss table |

**This is a plumbing result, not a design result.** Eight rounds is about 6% of the default
budget and every stage gate was nulled so nothing could abort, so the sequence it produced
means nothing. What it establishes is that the pipeline executes: settings parsing, target
preparation, the four optimiser stages, the loss registry, per-round recording and sequence
readout all work on this backend.

What was actually needed, all of it outside the checkout:

- **matplotlib.** `trajectory.py:8` imports `TrajectoryRecorder`, and
  `trajectory_output.py:12` imports matplotlib at module level, unguarded. Without it
  `import bindcraft.trajectory` fails before anything runs. Installed to a side directory;
  `MPLCONFIGDIR` is pointed at scratch so the font cache does not land in `$HOME`.
- **float32.** `_alphafold_runner` (af2.py:243-259) sets `bfloat16 = True` on a fresh
  deepcopy of the config, so there is nothing upstream to override. The driver wraps that
  method. Wrapping matters rather than mutating runners afterwards: runners are cached on
  `(model_family, subbatch_size, attention_backend, use_cueq)` (af2.py:62) and created
  lazily, so a post-hoc walk over `alphafold_runners` catches only the small probe runner
  built in `__init__` and leaves later ones in bfloat16.
- **`binder_lengths`** has no default and raises if omitted.
- **`min_iptm_final: 0.0`, never `null`.** settings.py:480-482 guards on key presence rather
  than on the value, so null reaches `float()` and raises.
- **One preset.** The full design pool loads roughly 1.9 GB of parameters.

**No CUDA bypass was needed.** Importing `bindcraft.trajectory`, `af2`, `protein` and
`settings` pulls in none of `cli.py`, `design_workers.py`, `selfcheck.py` or `preflight.py`,
and issues zero subprocess calls — verified by instrumenting `subprocess.run` during import,
not inferred from reading imports.

### Runtime: subbatching attention is a free 5x

The first trajectory ran at 42.9 s per gradient round (measured marginally, 8 rounds vs 24,
with compilation only 20 s of it), which puts the default 125 gradient + 15 mutation budget
near 100 minutes. Mosaic's AF2 gradient at a comparable size is 2.08 s, so BindCraft2 was
roughly 20x slower for the same model on the same backend.

Ruled out by measurement, not assumption: MSA depth (both feed depth 1 -- af2.py:133 builds
`msa_feat` as `(1, length, 49)` and `num_extra_msa` is 1), gradient checkpointing (af2.py:251
already sets `use_remat = True`, and both projects run the 48 blocks the same way, via
`layer_stack(48)` wrapping `hk.remat(evoformer_fn)`), design model count (pinned to 1 and
confirmed by `design_model_count()`), length bucketing (af2.py:56 pads the *design* chain
only, so a 60-mer becomes 64 and the target is untouched), recycling (`design_recycles: 0`
gives 1.04x) and dropout (`design_dropout: false` gives 1.00x).

The cause is memory. Scanning binder length at a fixed 76-residue target:

| binder | padded complex | s / round | device peak | under the 24.48 GB limit? |
|---|---|---|---|---|
| 20 | 108 | **5.3** | 16.68 GB | yes |
| 40 | 140 | 36.2 | 27.85 GB | **no** |
| 60 | 140 | 46.9 | 27.85 GB | **no** |

(40 and 60 pad to the same 64-residue binder, which is why their memory is identical.) The
apparent scaling exponent of 6.2 is not algorithmic -- it is the cliff of falling out of
unified memory and swapping.

`resolve_subbatch_size` (af2.py:203-206) returns `None` below 384 residues, so AF2's
attention is never chunked at the sizes a binder trajectory actually uses. Passing an
explicit `subbatch_size` chunks it:

| subbatch_size | s / round | device peak | speedup |
|---|---|---|---|
| `auto` (no chunking) | 46.9 | 27.85 GB | 1.00x |
| **64** | **9.4** | **15.23 GB** | **5.00x** |
| 32 | 9.2 | 15.29 GB | 5.08x |

**The designed sequence is byte-identical across all three.** Chunked and stock attention
already agreed to 3.7e-9 in the earlier CPU audit, so this is the same computation; getting
back under the memory ceiling is pure win. A full 125+15 round trajectory drops from roughly
100 minutes to about 22.

`run_trajectory_mac.py --subbatch 64` sets it. This is a Mac-specific default worth reporting
upstream alongside the bfloat16 finding: BindCraft2's 384-residue threshold is presumably
tuned for GPUs with more memory than 24 GiB of shared unified memory.

### A full-budget trajectory with the real gates: 13 minutes, and it failed honestly

Settings with no `*_steps` and no `min_*` overrides, so BindCraft2's own budget and its own
stage gates apply, plus `--subbatch 64`. Target ubiquitin chain A, 60-residue de novo binder,
one design model, seed 7. `settings-ubiquitin-full.json`, results in
`trajectory-full-report.json` and `trajectory-full-losses.csv`.

| stage | rounds | i_pTM start -> end | best | gate | outcome |
|---|---|---|---|---|---|
| screen | 50 | 0.08 -> 0.14 | 0.20 | pLDDT 0.60 | passed, pLDDT 0.80 |
| refine | 25 | 0.11 -> 0.16 | 0.19 | pLDDT 0.60 | passed, pLDDT 0.79 |
| anneal | 45 | 0.12 -> 0.14 | 0.24 | i_pTM 0.50 | **failed** |
| harden | 5 | — | — | — | not reached |
| mutate | 15 | — | — | — | not reached |

120 of 140 rounds in 678.3 s, i.e. **5.7 s per round and 11.3 minutes**, at 15.24 GB of the
24.48 GB limit. A trajectory that completes all 140 rounds is about **13 minutes** — compile
amortises further over a longer run than the 9.4 s/round measured from two rounds.

**The trajectory failed, and that is the correct outcome to report.** i_pTM never exceeded
0.24 against the 0.50 the anneal stage requires, so BindCraft2's own quality gate rejected
the design and stopped. The pipeline is faithful including its rejection behaviour, which is
more reassuring than a pass would have been at this stage. The sequence it reached,
`GAKRRMRMYQKRSQKAWWMHQNAPRNMRRGERRRFRNRAYRMFERYWPMAWKEFEMMENG`, is conspicuously
arginine-rich — the charge-driven non-specific-binder failure mode.

Nothing about design quality follows from this. It is one trajectory; BindCraft2's defaults
assume `max_trajectories` in the thousands and keep the few that survive. It also ran with
`design_models=1` rather than the 5-model pool, against a promiscuous hub protein, with no
epitope hotspots specified. What it establishes is that the machinery, the budget and the
gates all work on Apple Silicon at a usable speed.

### What is still not established

### What is still not established

### What is still not established

### What is still not established

- **No AF2 gradient meets the 0.5% gate on MPS.** Only the forward does, and only in float32.
- One loss scalar (mean pLDDT) was differentiated. BindCraft2's composite loss registry, its
  structure and contact terms and the full `DesignLoss` plumbing were not exercised.
- Everything ran at `num_recycle=0`. Recycling is `stop_gradient`-wrapped per iteration
  (`af2.py:140`) but was not measured.
- All inputs were random sequences with no MSA and no real template, a low-confidence and
  precision-sensitive regime. The bf16 numbers are an upper bound on the discrepancy, not a
  typical one.
- Sizes were 40-100 residues padded to 64/128. Nothing at campaign scale; memory headroom
  there is unknown (peak RSS here 1.2-3.0 GiB of 24).
- No trajectory, no optimization loop, no MPNN redesign stage, no filters, no campaign.
  ProteinMPNN's forward pass was never run with real weights, only its featuriser.
- `fused_triangle_multiplicative_update` (cuEquivariance) remains unavailable and untested.

Remaining work:

- Replace the CUDA-oriented installer with an isolated Mac environment.
- Bypass NVIDIA-only device discovery and multiprocessing; use one MPS worker. CUDA coupling
  is confined to five files: `design_workers.py`, `selfcheck.py`, `cli.py`, `af/accel.py`,
  `af/alphafold/model/modules.py`.
- **Resolve the backward-pass discrepancy**, which is the actual blocker.
- Turn off the hardcoded `bfloat16 = True` for Mac.

The upstream [license](https://github.com/PacesaLab/BindCraft2/blob/e6d30f6ea2e5bbc2f62ae7fa722f183da6c6c29f/LICENSE)
is named **BindCraft2 Source-Available License (Hosting-Restricted)** and states
that it is not OSI-approved. Its terms differ from MacFoldKit's Apache-2.0 code.
An integration should download it separately, preserve its notices and explain
its terms; this audit does not relicense or redistribute BindCraft2.

Relevant source: [dependencies](https://github.com/PacesaLab/BindCraft2/blob/e6d30f6ea2e5bbc2f62ae7fa722f183da6c6c29f/pyproject.toml),
[attention](https://github.com/PacesaLab/BindCraft2/blob/e6d30f6ea2e5bbc2f62ae7fa722f183da6c6c29f/bindcraft/af/accel.py),
[AF2 gradient pipeline](https://github.com/PacesaLab/BindCraft2/blob/e6d30f6ea2e5bbc2f62ae7fa722f183da6c6c29f/bindcraft/af2.py),
[worker discovery](https://github.com/PacesaLab/BindCraft2/blob/e6d30f6ea2e5bbc2f62ae7fa722f183da6c6c29f/bindcraft/design_workers.py).
