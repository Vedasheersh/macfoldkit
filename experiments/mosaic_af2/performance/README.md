# Where the design workflow's time goes on an M5 Pro

Measurements, not estimates. Target of optimization is the Mosaic AF2 design gradient,
which is the heaviest GPU consumer in this repository: 2.08 s per step for a 124-residue
complex, 9.25 s for 231 residues.

Machine: Apple M5 Pro, 20 GPU cores, 24 GiB unified memory, macOS 26.6.2.

## Hardware peak, and whether any backend misses it

Empirical roofline from square FP32 matmuls (`tri_roofline.py`):

| N | jax-mps 0.11.2 | MLX 0.32.0 | torch-mps 2.14 |
|---|---|---|---|
| 1024 | 1129 | 1734 | 2399 |
| 2048 | 6636 | 6755 | 5117 |
| 4096 | **7404** | **7406** | **7431** |

**Peak is ~7.4 TFLOPS FP32 and all three backends reach it.** No backend is leaving
large-matmul throughput on the table. The spread at N=1024 is launch overhead, which
amortizes away by N=2048.

## MLX versus jax-mps on AF2's dominant pair operation

Triangle multiplication is `einsum('ikc,jkc->ijc')`, i.e. a batched matmul over the channel
axis, `A(128,N,N) @ B(128,N,N)^T`, costing `2*c*N^3` FLOPs. It is the characteristic
Evoformer/pairformer operation.

| N | jax-mps | MLX | torch-mps | MLX vs jax-mps | % of 7.4 TFLOPS peak |
|---|---|---|---|---|---|
| 124 (ubiquitin complex) | 1926 | 1755 | 1906 | **0.91x** | 26% |
| 231 (EGFR complex) | 4072 | 4012 | 3568 | **0.99x** | 55% |

**MLX is not faster than jax-mps on this operation — it is slightly slower.** Replacing it
with an MLX kernel would gain nothing. The 26-55% of peak is a shape and occupancy limit:
these are 128 batched N x N x N matmuls with small N, which do not saturate 20 GPU cores.
It is not a backend deficiency, and it is the same on all three frameworks.

This is the direct answer to "is there an MLX-level optimization here": **not at this
operation.** Prior MLX/Metal kernel wins in this project were on *inference* attention in
Boltz (427-residue attention 4.728 -> 0.951 ms) where the shapes and the fused softmax
made a difference. That work does not transfer to design, which needs backward passes.

## Two configuration levers, both measured, both negative

| change | result |
|---|---|
| `use_remat=False` (spend idle memory to avoid recomputation) | **impractical** — compilation exceeded 28 minutes at N=124 versus 5.0 s with remat on, and was killed. Gradient checkpointing is load-bearing for compile tractability on jax-mps, not only for memory. |
| `JAX_MPS_ASYNC_DISPATCH=1` (backend suggests it for dispatch-bound work) | **no change** — 2.080 s vs 2.076 s, a 0.998x ratio, with the loss and gradient **bit-identical**. |

The async result is the more informative of the two: this workload is **compute-bound, not
dispatch-bound**, so there is no win available from reducing launch overhead. Combined with
the roofline above, the arithmetic is already running near what the shapes allow.

## The open question

288 triangle multiplications (48 Evoformer blocks x 2 directions, forward plus
remat-recomputed forward plus backward) at 0.25 ms each is roughly 72 ms — about **3.5%**
of the 2076 ms gradient step. Triangle multiplication is therefore *not* the bottleneck in
this workload, and neither is dispatch overhead. Where the other ~96% goes is not yet
established and is the next thing to measure before any kernel work is justified.

## Reproduce

```bash
MODEL_HOME="${MACFOLDKIT_HOME:-$HOME/Library/Caches/macfoldkit}"
EXP=experiments/mosaic_af2
JAX_PLATFORMS=mps MLX_ENABLE_TF32=0 JAX_MPS_ASYNC_DISPATCH=0 \
  PYTHONPATH="$MODEL_HOME/sources/mosaic/src" \
  "$MODEL_HOME/runtimes/mosaic/bin/python" "$EXP/performance/tri_roofline.py" jax roofline-jax.json
<an mlx environment>/bin/python "$EXP/performance/tri_roofline.py" mlx   roofline-mlx.json
"$MODEL_HOME/runtimes/boltz/bin/python"    "$EXP/performance/tri_roofline.py" torch roofline-torch.json
```

Timings take the minimum of 5 repeats after 2 warmups, with explicit GPU synchronization
(`block_until_ready`, `mx.eval`, `torch.mps.synchronize`).
