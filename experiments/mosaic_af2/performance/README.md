# Can the design workflow go faster on an M5 Pro?

**Yes, substantially — but not through configuration.** The design gradient runs at **3.7%
of this machine's bandwidth roofline**, about 27x off. Eight configuration- and
backend-level levers were measured and none helped. The remaining headroom is in operator
fusion of the real model's op graph, and at least **1.67x of it is demonstrably recoverable**.

An earlier revision of this file concluded "not materially, and not with MLX". That was
wrong, and the correction is recorded below under *MLX versus jax-mps*: the no-win result
came from benchmarking an **isolated** operation, which is not where the difference lives.

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

## MLX versus jax-mps: isolated ops tie, composed blocks do not

**Isolated triangle multiplication**, `A(128,N,N) @ B(128,N,N)^T`:

| N | jax-mps | MLX | torch-mps | MLX vs jax-mps | % of peak |
|---|---|---|---|---|---|
| 124 | 1926 | 1755 | 1906 | 0.91x | 26% |
| 231 | 4072 | 4012 | 3568 | 0.99x | 55% |

All three tie, because all three dispatch the same underlying matmul. Concluding from this
that "MLX offers no win" was a mistake — a single op is not where a compiler differs.

**A composed Evoformer-shaped block** — layer norm, gate, two matmuls, the triangle einsum,
residual — at N=124, C=128:

| implementation | time | vs jax-mps |
|---|---|---|
| jax-mps, jitted | 1415 us | 1.00x |
| MLX, eager | 935 us | **1.51x** |
| MLX, `mx.compile` | **849 us** | **1.67x** |

**MLX compiled is 1.67x faster than jax-mps jitted, and MLX eager beats jax-mps jitted.**
That is real, recoverable headroom, and it only appears once operations are composed.

A trivially fusable elementwise chain (layer norm, sigmoid gate, residual) shows no gap —
jax-mps 421 us against `mx.compile` 409 us, within 3% — so jax-mps does fuse simple chains
competently. The gap opens at the boundaries a realistic block introduces: reductions,
transposes, batched matmuls and einsums.

## How far from the limit, precisely

XLA cost analysis of the same computation (`cost_analysis.py`, run on the CPU backend since
jax-mps does not expose it), 124-residue complex, one `value_and_grad`:

| quantity | value |
|---|---|
| FLOPs | 167.7 GFLOP |
| bytes accessed | 21.5 GB |
| arithmetic intensity | 7.78 FLOP/byte |

This machine's measured rooflines: **7.4 TFLOPS** compute, **285 GB/s** bandwidth (both
backends agree; see `bandwidth-*.json`). The ridge point is 7404/285 = 26 FLOP/byte, and at
7.78 FLOP/byte this workload is firmly **bandwidth-bound**.

| bound | time |
|---|---|
| compute-bound: 167.7 GFLOP / 7.4 TFLOPS | 0.023 s |
| **bandwidth-bound: 21.5 GB / 285 GB/s** | **0.076 s** |
| **measured** | **2.068 s** |

**The design gradient achieves 3.7% of its roofline.** Caveats worth stating: cost analysis
is approximate, and the byte count comes from XLA's fused HLO, so it describes an idealised
fused execution rather than what jax-mps actually issues — the real traffic is higher and
the true gap correspondingly smaller. But it is not a factor of 27 smaller, and no
interpretation of these numbers puts this workload near its limit.

## Where the time actually goes

| component | seconds | share |
|---|---|---|
| forward pass | 0.483 | — |
| full value_and_grad | 1.994 | 100% |
| backward multiplier | **4.13x** | signature of full recomputation |
| MPNN inverse-folding term | 0.072 | 3.4% |
| all triangle multiplications | ~0.072 | ~3.5% |

Nothing dominates. The dominant op is 3.5% of the step; the MPNN term another 3.4%. The
time is spread across a long chain of medium-sized, low-arithmetic-intensity operations —
layer norms, gating, softmaxes, transposes over the pair tensor.

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

A 3.1x GPU speedup, at ~9% of peak FLOPs, is the honest characterization: this workload
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
