# Can the design workflow go faster on an M5 Pro?

**Not by switching backends, and not by configuration.** Ten levers were measured; none
helped. On realistic composed blocks jax-mps and MLX are **within 3% of each other** when
measurements are replicated. The design gradient sits far from an idealised roofline, but
that gap is not reachable through any backend or flag tested here.

**This file has now been wrong twice on the same question, in opposite directions.** Both
errors are documented below under *MLX versus jax-mps*, because the way they happened is
more useful than the numbers: the first generalised an isolated-operation tie to all code,
the second built a "1.67x recoverable" claim on a single unreplicated measurement that had
caught jax-mps in a bad state. Replication, not reasoning, settled it.

Measurements, not estimates. Target is the Mosaic AF2 design gradient, the heaviest GPU
consumer here: 2.08 s per step at 124 residues, 9.25 s at 231.

Machine: Apple M5 Pro, 20 GPU cores, 24 GiB unified memory, macOS 26.6.2.
jax-mps 0.11.2 / MLX 0.32.0 / torch-mps 2.14.

## Result table

| lever | speed vs baseline | why it fails |
|---|---|---|
| MLX kernels for the dominant pair op | **0.91–0.99x** | MLX is not faster than jax-mps; all backends reach hardware peak |
| `JAX_MPS_ASYNC_DISPATCH=1` | **0.998x** | not dispatch-bound; loss and gradient bit-identical |
| `use_remat=False` | **impractical** | compile exceeded 28 min at N=124 vs 5.0 s |
| remat policy `dots_no_batch` | **0.778x** | peak memory 2.66 -> 20.61 GB |
| remat policy `dots` | **0.040x** | peak memory 26.30 GB, over the 24.48 GB limit |
| `bfloat16` Evoformer | **1.109x** | 23.3% gradient error against a 0.5% gate |
| batching 2 or 4 designs (vmap) | **1.00x / 0.97x** per design | GPU already saturated at batch 1 |

## The machine's peak, and whether any backend misses it

Square FP32 matmul, `tri_roofline.py`:

| N | jax-mps | MLX | torch-mps |
|---|---|---|---|
| 1024 | 1129 | 1734 | 2399 |
| 2048 | 6636 | 6755 | 5117 |
| 4096 | **7404** | **7406** | **7431** |

**Peak is ~7.4 TFLOPS FP32 and all three backends reach it.** No backend is leaving
large-matmul throughput on the table. The N=1024 spread is launch overhead that amortizes
away.

## MLX versus jax-mps: they tie on real work. Two retracted claims

**Replicated result, 4 independent processes per framework**, same Evoformer-shaped block,
bit-identical inputs and bit-identical output checksums:

| | run 1 | run 2 | run 3 | run 4 | spread |
|---|---|---|---|---|---|
| jax-mps, min | 868 | 816 | 818 | 854 us | 6% |
| MLX compiled, min | 799 | 792 | 808 | 797 us | 2% |

**Best-of-4 ratio 1.03x — a tie.** jax-mps has visibly worse run-to-run variance (its
*median* ranged 964-1431 us against MLX's stable 869-911) but the same best case.

### Where MLX genuinely is faster, and why it does not matter

Cumulative decomposition of that block, all checksums matching:

| stage | jax-mps | MLX | MLX gain |
|---|---|---|---|
| layer norm alone | 759 us | 318 us | **2.39x** |
| + one matmul | 565 | 400 | 1.41x |
| + two matmuls | 821 | 518 | 1.58x |
| + gating | 767 | 521 | 1.47x |
| + triangle einsum | 746 | 748 | **1.00x** |
| + residual (full block) | 832 | 836 | **1.00x** |

MLX is genuinely 1.4-2.4x faster on **isolated memory-bound elementwise and reduction
work**. That advantage disappears completely the moment a compute-heavy einsum is present —
and every real Evoformer block contains several. Note also that jax-mps is *faster* at
"layer norm + one matmul" (565 us) than at layer norm alone (759 us): it materialises the
result when it is the output and fuses it into the matmul when it is not, so isolated-op
benchmarks systematically misrepresent it.

A six-component sweep at real AF2 shapes (`component-*.json`) agrees: forward and backward,
jax-mps and MLX tie on triangle multiplication, triangle attention, transition and the
composed block. The only large gaps are isolated layer norm (MLX 2.8x on forward) and
isolated layer-norm backward (jax-mps 1.9x).

### The two retractions

1. *"MLX shows no win"* — drawn from an isolated triangle multiplication, where all three
   backends tie because all three dispatch the same matmul. Too narrow to support the claim.
2. *"1.67x demonstrably recoverable"* — drawn from one measurement of one block. Replicating
   it four times gives 1.03x. The 1208 us jax figure behind that claim was an outlier;
   the same code measures 816-868 us across four clean runs.

The lesson worth keeping: on this machine a single timing is not evidence. jax-mps's
run-to-run spread is large enough to manufacture a 1.5x "finding" from noise.

## How far from the limit: about 70-78% of the bandwidth roofline

**This section previously claimed 3.7%. That was wrong, and it was wrong because the
measuring instrument was broken.**

`cost_analysis.py` used XLA's `HloCostAnalysis`. XLA counts a `while`/`scan` body **once,
regardless of trip count**. AF2's Evoformer is `layer_stack(48)` (modules_multimer.py:477,
`evoformer_num_block: 48`), implemented as `hk.scan` (layer_stack.py:156), so it lowers to a
single HLO `while`. The extra-MSA stack (4 blocks) and template pair stack (2 blocks) are
scans too. Every FLOP and byte figure derived from that call undercounted by roughly the
trip count.

Verified directly (`scan_cost_check.py`, CPU-only, no weights):

| trip count | scan FLOPs | loop FLOPs |
|---|---|---|
| 1 | 524,288 | 524,288 |
| 48 | **524,290** | **25,165,824** |
| ratio | **1.00x** | **48.00x** |

Scan cost is flat from 2 to 48 while the unrolled equivalent scales exactly 48x. Gradient
cost is flat too. The instrument truncates the model to roughly one block.

Two artifact-only checks confirm the undercount without relying on that probe. The reported
4.748 GB for a whole forward is **smaller than one of its own components**: the
triangle-attention logits tensor is (124, 4, 124, 124) float32 = 30.5 MB, written and read
at minimum, twice per block over 48 blocks = 5.86 GB. And a hand FLOP count per Evoformer
block at N=124, c_z=128 gives roughly 22-24 GFLOP, so 48 blocks is about 1060-1150 GFLOP
against XLA's reported 56.4 GFLOP for the forward -- a factor near 19-21.

### Corrected position

| quantity | corrected |
|---|---|
| forward FLOPs | ~1060-1150 GFLOP (not 56.4) |
| gradient FLOPs, at the measured 4.13x multiplier | ~4400 GFLOP |
| achieved throughput | ~2200 GFLOPS, **~30% of the 7404 GFLOPS matmul peak** |
| gradient bytes | ~409-460 GB (not 21.5) |
| bandwidth-bound lower bound | ~1.44-1.61 s |
| **measured** | **1.994 s** |
| **fraction of bandwidth roofline** | **~72-78%** |

**The design gradient runs at roughly three quarters of this machine's memory-bandwidth
roofline.** Driving it to 100% would be about a **1.3x** speedup, not 27x.

This is also the only reading consistent with the rest of this document. Ten independent
levers failing to help is exactly what a workload at ~75% of its roofline looks like, and is
very hard to explain at 3.7%. The internal contradiction should have been the tell.

### What still has headroom

Being near the bandwidth roofline bounds what *scheduling* can win, not what *doing less
work* can win. Two concrete reductions were identified and are not yet tested:

- **OuterProductMean contraction order** (modules.py compute_chunk). At the single-sequence
  MSA depth this experiment uses, the contraction is performed in an order roughly 13-14x
  more expensive than necessary; the term is about 16% of model FLOPs.
- **The 4-block extra-MSA stack** runs full-width pair operations on an all-zero, fully
  masked MSA -- about 7.4% of model FLOPs for no information.

Both change the arithmetic, so both need the CPU/MPS gradient gate re-run, and the second
needs the design output re-screened rather than only re-timed.

## Where the time actually goes

| component | seconds | share |
|---|---|---|
| forward pass | 0.483 | — |
| full value_and_grad | 1.994 | 100% |
| backward multiplier | **4.13x** | signature of full recomputation |
| MPNN inverse-folding term | 0.072 | 3.4% |
| all triangle multiplications | ~0.072 | ~3.5% |

**Correction.** This section previously said "nothing dominates" and put triangle
multiplication at 3.5% of the step, extrapolated from an isolated-op timing multiplied by a
hand-counted call count. That was wrong by roughly 9x. By FLOPs, four pair operations are
about 97% of the trunk, and the trunk is about 91% of the model: triangle multiplication
~29%, triangle attention ~29%, OuterProductMean's output projection ~16%, pair transition
~16%. The MPNN term really is 3.4%, measured by ablation rather than extrapolation.

The 4.13x backward multiplier is full gradient checkpointing recomputing the whole forward,
which suggested spending idle memory to avoid it. That trade is **inverted on unified
memory**: keeping matmul outputs took peak memory from 2.66 GB to 20–26 GB and made things
0.78x and 0.04x, because activations compete for the same bandwidth the arithmetic needs.
Recomputing is cheaper than storing here. Both policies produced **bit-identical
gradients**, confirming the measurements are pure performance.

But the workload is not purely bandwidth-bound either: halving activation bytes with
bfloat16 bought only 11%, and barely moved peak memory (2.66 -> 2.61 GB). No single
resource is the limit, which is why no single lever helps.

## The structural number

| | seconds per gradient step, 124 residues |
|---|---|
| CPU (jax cpu) | 6.51 |
| MPS (jax-mps) | 2.08 |
| **GPU speedup** | **3.1x** |

A 3.1x GPU speedup, at roughly 30% of matmul peak, is the honest characterization: this workload
suits the GPU poorly. It is not idle — batching and async dispatch both fail to help, so it
is genuinely busy — it is simply executing many low-intensity operations. That is a
property of the Evoformer's op mix under reverse-mode AD, not of jax-mps.

## Where optimization does pay: inference, not design

This project's earlier Metal/MLX work won real speedups on **inference**: a native Boltz
Metal attention kernel took 427-residue attention from 4.728 ms to 0.951 ms and the full
forward from 20.61 s to 18.93 s, and a rebuilt jax-mps runtime took 427-residue ColabFold
from 88.7 s to 54.8 s. Both remain opt-in because of accuracy deltas.

Those wins rest on fused low-precision attention. Design cannot use them: it needs float32
gradients, and bfloat16 here costs 23% gradient error. This is the distinction to carry
forward — **fused Metal kernels and reduced precision pay off for folding, and structurally
cannot pay off for design.**

## Recommendation

Do not invest in MLX kernels for the design path. The measured ceiling is the op mix, and
every backend is already at it. If design throughput matters, the effective lever is fewer
or cheaper gradient steps — a shorter schedule, a cheaper objective, or a smaller binder —
not faster kernels.

## Reproduce

```bash
MODEL_HOME="${MACFOLDKIT_HOME:-$HOME/Library/Caches/macfoldkit}"
EXP=experiments/mosaic_af2
P="$EXP/performance"

# backend roofline, run under each environment
JAX_PLATFORMS=mps MLX_ENABLE_TF32=0 JAX_MPS_ASYNC_DISPATCH=0 \
  PYTHONPATH="$MODEL_HOME/sources/mosaic/src" \
  "$MODEL_HOME/runtimes/mosaic/bin/python" "$P/tri_roofline.py" jax roofline-jax.json
<mlx env>/bin/python                        "$P/tri_roofline.py" mlx   roofline-mlx.json
"$MODEL_HOME/runtimes/boltz/bin/python"     "$P/tri_roofline.py" torch roofline-torch.json

# forward vs gradient split, and batching
"$MODEL_HOME/runtimes/mosaic/bin/python" "$P/cost_analysis.py" \
  "$MODEL_HOME/weights/colabfold" "$EXP/ubiquitin/ubiquitin-target.cif" 48 cost.json
"$MODEL_HOME/runtimes/mosaic/bin/python" "$P/batch_bench.py" \
  "$MODEL_HOME/weights/colabfold" "$EXP/ubiquitin/ubiquitin-target.cif" 48 batch.json 1,2,4

# configuration levers, via run.sh
bash "$EXP/run.sh" mps out --length 48 --steps 125 --seed 7 --mpnn-weight 5 \
  --optimizer bindcraft --remat-policy dots --bfloat16 off \
  --target "$EXP/ubiquitin/ubiquitin-target.cif" --probe-only
```

Timings take the minimum of 3–5 repeats after warmup, with explicit GPU synchronization.
`--remat`, `--remat-policy` and `--bfloat16` all default to the published settings.
