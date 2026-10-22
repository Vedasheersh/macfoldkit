# Lessons from building this: what the measurements taught us

A retrospective on the work behind MacFoldKit, September 26–30 2026. Hardware
throughout: Apple M5 Pro, 20 GPU cores, 18 CPU cores, 24 GiB unified memory,
macOS 26.6.2.

This file records *how* the numbers elsewhere in this repository were arrived
at, including the ones that were published and later withdrawn. The individual
results live in [validation.md](validation.md),
[bindcraft2-feasibility.md](bindcraft2-feasibility.md) and
[the Mosaic AF2 performance study](../experiments/mosaic_af2/performance/README.md).
Earlier inference reports and raw benchmark logs are in the surrounding
workspace, outside this repository.

---

## How performance evolved

Four phases: *does it run at all* → *make it fast* → *package it* → *make it
design things*, with a running fifth theme of discovering that the measurements
had been lying.

### Phase 1 — feasibility

The starting question was only whether ESMFold2 and Boltz2 could run on a Mac.

- **Boltz-2** via the `colbyford` MPS fork worked with **no source changes**.
  76-residue ubiquitin, 12.4 s cached, 4.7 GiB RSS, pLDDT 92.9, pTM 0.923.
- **Protenix-v2** needed a small MPS compatibility patch: explicit MPS/FP32
  selection, CPU checkpoint staging with restricted `weights_only` loading, and
  a dtype fix where the confidence code allocated unsupported float64.
- **FastPLMs Boltz2** was best of the lot — 5.25 s, 3.46 GiB, pLDDT 93.1.
- **ESMFold2-Fast was a trap.** It reduces the folding trunk but keeps the 6B
  ESMC backbone: ~25 GB of weights, 18.89 GiB footprint, and it scored *worse*
  (82.5) than the small 300/600 variants. Loading dominated its runtime.

### Phase 2 — complexes and the optimization push

The standing benchmark became TEM-1 β-lactamase / BLIP, 427 residues, PDB 1JTG.

**Boltz2 (FastPLMs, PyTorch MPS):**

| Config | Inference | Peak footprint | Complex Cα RMSD |
|---|---:|---:|---:|
| Full reference, 3 recycles / 200 steps | 101.5 s | 7.92 GiB | 0.62 Å |
| Balanced 1 / 100 | 53.8 s | 7.95 GiB | 0.81 Å |
| + compiled pair update (Inductor MPS) | 33.5 s | 9.02 GiB | 0.72 Å |
| + BF16 pair work | 27.6 s | 7.02 GiB | 0.72 Å |
| Final general runner | 22.27 s | 6.94 GiB | 0.723 Å |

One compiled graph is reused across the 64 trunk pairformer blocks, 4 MSA pair
blocks and 8 confidence pairformer blocks. This combines fewer recycles and
sampling steps *with* implementation gains; it is not an equivalent-settings
3.7x kernel speedup.

**ColabFold 1.6.3 (AlphaFold2, community `jax-mps` backend):**

| Config | 427-residue process time | Peak | Numerically exact? |
|---|---:|---:|---|
| CPU (JAX 0.11.2) | 173.3 s | 2.55 GiB | baseline |
| Stock GPU | ~88.7 s | 10.43 GiB | — |
| + bounded Evoformer batching | 68.4 s | 10.81 GiB | **yes**, byte-identical |
| + rebuilt NAX-enabled backend | 54.8–56.1 s | 10.81 GiB | no |
| + three-pass native Metal attention | 39.8–42.3 s | ~10.37 GiB | no |

Only the batching step preserved outputs exactly. Everything past it changed
geometry and stayed explicitly opt-in — which is why none of it is in the
installer. See [limitations.md](limitations.md).

Apple's own `jax-metal` never worked: it detects the M5 Pro but fails on basic
tensor execution in **both** JAX versions ColabFold accepts. A detected GPU is
not evidence of executable model support.

### Phase 3 — packaging

Alpha `0.1.0a1`: one CLI, three isolated pinned backend environments, 29 tests,
CI. Separate runtimes because the Torch / JAX / NumPy requirements of the three
backends are mutually incompatible. The shipped version deliberately uses
ordinary MPS and JAX-MPS paths.

### Phase 4 — design

A full de novo workflow: random sequence → AF2 confidence/structure objective →
150 Mosaic steps → ProteinMPNN redesign → independent offline Boltz2 refold →
frozen screen. 150 steps in 91.3 s at 64 residues; warm full gradient 0.599 s;
1.46 GiB peak RSS. Binder campaigns ran v1 0/3, v2 0/3, v3 3/18. BindCraft2 went
from "probably too hard to port" to a complete 140-round trajectory in 13
minutes.

---

## Lessons

### 1. Three claims were measured, published, and retracted

All within the performance study, on the same question.

- *"MLX shows no win"* — drawn from an isolated triangle multiplication, where
  all three backends tie because all three dispatch the same matmul. Too narrow
  to support the claim.
- *"1.67x demonstrably recoverable"* — drawn from a **single** measurement that
  caught jax-mps in a bad state. Replicated across four independent processes:
  **1.03x**. The 1208 µs figure behind the claim was an outlier; the same code
  measures 816–868 µs across four clean runs.

jax-mps's run-to-run spread is wide enough to manufacture a 1.5x "finding" out
of noise — its *median* ranged 964–1431 µs against MLX's stable 869–911 µs, for
the same best case.

> **On this machine a single timing is not evidence.**

Where MLX genuinely is faster — isolated layer norm (2.4x), elementwise and
reduction work — the advantage vanishes the moment a compute-heavy einsum is in
the block, and every real Evoformer block has several. Worse, jax-mps is
*faster* at "layer norm + one matmul" than at layer norm alone, because it fuses
the norm into the matmul when the result is not the output. Isolated-op
benchmarks systematically misrepresent it.

### 2. The instrument was broken, and the contradiction should have been the tell

A published claim that the design gradient ran at **3.7% of roofline** came from
XLA's `HloCostAnalysis`, which counts a `scan` body **once regardless of trip
count**. AF2's Evoformer is `layer_stack(48)` lowered to a single HLO `while`,
so every FLOP and byte figure was off by roughly 48x.

| quantity | reported | corrected |
|---|---:|---:|
| forward FLOPs | 56.4 GFLOP | ~1060–1150 GFLOP |
| gradient bytes | 21.5 GB | ~409–460 GB |
| fraction of bandwidth roofline | 3.7% | **~72–78%** |

Driving it to 100% would be about **1.3x**, not 27x.

Two artifact-only checks caught the undercount without the probe: the reported
4.748 GB for a whole forward is *smaller than one of its own components*, and a
hand FLOP count per block disagreed by a factor near 19–21.

> **The giveaway was already in the document.** Ten independent optimization
> levers had all failed to help. That is exactly what a workload at ~75% of its
> roofline looks like, and is very hard to explain at 3.7%.

### 3. Twice, the failure was the test — not the backend

- **BindCraft2's AF2 backward pass** "failed" the 0.5% gate at **1.756%** and
  was recorded as the blocker for a design campaign. But the input was
  `Protein.empty` — uniform noise at pLDDT 0.31, the worst possible
  conditioning, and every individual primitive had already passed cleanly
  (attention 7e-08, triangle multiplication 8e-07). Re-run with **one variable
  changed**, a real ubiquitin sequence: **0.288%**, passes.
- **A 44.4 GB memory peak** "explained" why Boltz2-as-design-backend was
  infeasible. It was a process-lifetime high-water mark accumulated across three
  configs and a failed run in one process. Measured one config per process:
  **5.25 GB**. That figure was also *internally impossible* — it exceeded the
  24.48 GB device limit, and was byte-identical at 5 and 25 sampling steps.

### 4. A real upstream bug existed, and was found by refusing to accept 3%

Mosaic's full AF2 gradients disagreed CPU-vs-MPS by ~3% — tempting to write off
as accumulation over 48 blocks. Isolated, it was a **signedness defect in MLX's
`scatter.h`**: `ind_idx * static_cast<LocT>(upd_size) + gid.x` infers an
unsigned offset, so a negative-stride `elem_to_loc` result wraps to a large
positive offset. Descending-sort backward drops row zero's cotangents.

A four-line pure-JAX reproducer demonstrates it:

```python
import jax, jax.numpy as jnp
x = jnp.array([[0., 1., 2., 3.], [4., 5., 6., 7.]])
f = lambda x: jnp.sort(x, axis=-1, descending=True)[:, :2].mean()
print(jax.jit(jax.grad(f))(x))
# both devices should give [[0,0,.25,.25],[0,0,.25,.25]]
# tested MPS backend returns all zeros in the first row
```

Upstream fails 20/40 regression cases on MPS. The custom-VJP workaround passes
40/40 on both devices and took full-model gradient error from ~3% to **0.149%**
(monomer) and **0.0479%** (124-residue complex). Details in
[SCATTER_BACKWARD_BUG.md](../experiments/mosaic_af2/SCATTER_BACKWARD_BUG.md).

### 5. Memory cliffs masquerade as algorithmic scaling

BindCraft2 appeared to scale at exponent **6.2** with binder length. It was
falling out of unified memory and swapping.

| binder | padded complex | s / round | device peak | fits in 24.48 GB? |
|---|---:|---:|---:|---|
| 20 | 108 | 5.3 | 16.68 GB | yes |
| 40 | 140 | 36.2 | 27.85 GB | **no** |
| 60 | 140 | 46.9 | 27.85 GB | **no** |

`resolve_subbatch_size` returns `None` below 384 residues, so attention is never
chunked at the sizes a binder trajectory actually uses — presumably tuned for
GPUs with more memory than 24 GiB shared. Passing `subbatch_size=64` gives a
**5.00x** speedup at 15.23 GB, and **the designed sequence is byte-identical**.
A full 125+15 round trajectory drops from ~100 minutes to about 22.

### 6. CUDA optimization kits don't transfer, but some of their ideas do

Anthropic's Boltz-2 kit depends on CUDA graphs, Triton/CUDA kernels, NVRTC and
cuEquivariance — none reusable under PyTorch MPS. Its platform-independent ideas
were checked individually: FastPLMs already omits the template module and
computes diffusion conditioning outside the sampler loop, and skipping
identity-valued dropout multiplications gave byte-identical coordinates with no
speedup, so it was discarded.

Apple's SimpleFold MLX code supplied useful *patterns* (packed QKV projections,
native MLX SDPA) but no reusable kernel — its architecture omits Boltz2's
triangle attention and pair biases entirely.

### 7. Biology-side lessons

- **Paired MSA depth was the difference between unusable and excellent.** On
  barnase/barstar, an empty MSA gave iPTM 0.091 and 14.98 Å complex RMSD; a
  paired MSA gave **iPTM 0.960 and 0.53 Å**. Raising recycling from 1 to 3 and
  sampling from 50 to 200 did *not* recover the interface.
- **But MSA depth is a poor speed lever.** Going 512 → 1024 rows was effectively
  free and improved every interface metric except contact recall. An early
  random subsample was replaced because it discarded most paired rows, and a
  256-row prefix produced a wrong interface *at high confidence*.
- **Seed variance swamped every protocol effect tested.** The v3 campaign's
  between-seed spread was ~30 pLDDT points and 2–34 Å of pose. All 3 passing
  candidates of 18 came from one seed, in both length arms. This is why v3
  pre-registered its seeds before any run, and why `run_design.py` rejects a
  seed outside the declared set. That is what separates a campaign from
  seed-shopping.
- **Outcomes were close to bimodal:** a design either converges to something
  self-consistent (pLDDT 91–94, iPTM 0.86–0.95, pose 0.5–2.4 Å) or fails
  clearly. There is little middle ground.

### 8. The structural insight that nearly got lost

Switching the design model from AF2 to Boltz2 would **destroy the validation**.

Every screen in this project rests on AF2 doing the design and Boltz2 doing an
*independent* refold. That independence is the only thing making the pass/fail
column mean anything. Designing with Boltz2 would force the validator to become
AF2 or Protenix, and every published threshold would need re-deriving against
the new validator. This is a much larger change than swapping a model — and it
is invisible if the swap is evaluated purely on cost.

The cost, for the record, turned out fine: a Boltz2 design step at the default
25 sampling steps is 1.77x AF2's time and ~2x its memory, not the 7–10x
originally estimated. Diffusion is not the cost driver — the trunk is ~81% of
the step, so reducing `sampling_steps` is not the affordability lever it looked
like. Time becomes prohibitive near ~300 tokens, well before memory does.

### 9. Negative results are worth packaging

Several things were measured, found useless, and kept as documented negative
results rather than quietly dropped: diffusion bookkeeping changes, whole
score-network compilation, FP16 (faster in isolation, failed in SVD alignment),
PyTorch SDPA (not faster than compiled mathematical triangle attention), bf16
Evoformer for design (1.109x for 23.3% gradient error against a 0.5% gate),
vmap batching of 2–4 designs (GPU already saturated at batch 1), and
`use_remat=False` (compile exceeded 28 minutes).

The full-budget BindCraft2 trajectory is the clearest case: it **failed** at the
anneal stage, i_pTM never exceeding 0.24 against the required 0.50, and produced
a conspicuously arginine-rich sequence — the charge-driven non-specific-binder
failure mode. Reporting that is more useful than a pass would have been, because
it shows the pipeline is faithful *including its rejection behaviour*.

### 10. Append-grown documents rot at the summary first

[bindcraft2-feasibility.md](bindcraft2-feasibility.md) accumulated its findings
by appending dated sections, and its closing "what is still not established"
list was left behind: it continued to assert that no AF2 gradient met the gate
on MPS and that no trajectory had run, both disproved by sections directly above
it in the same file. The header had also been duplicated four times.

When a finding is overturned, the summary sections need rereading along with the
body — they are the parts most likely to contradict the evidence beneath them,
and the parts a reader is most likely to trust.
