# Ubiquitin binder design, v2: hardened optimization schedule

> **Follow-up:** [`../v3/`](../v3/README.md) runs a pre-registered 3-seed campaign at two
> binder lengths. It produced this project's first screen passes (3/18 candidates) and
> shows that the single-seed result below is one draw, not a verdict on the protocol:
> seed-to-seed spread is ~30 pLDDT points, larger than any protocol effect measured here.

Computational workflow experiment. **No candidate passed the screen, and this is not
evidence of binding.** The v1 run in [`../`](../) is preserved unchanged.

v2 changes **exactly one variable** against v1: the optimization schedule. Target, binder
length, seed, all ten loss terms and their weights, the MPNN redesign, the independent
Boltz refolds and the **screen thresholds** are byte-identical to v1
(`protocol-v2.json` carries the same `screen` and `loss` blocks as `protocol.json`).

## Why the schedule was changed

Measured from the v1 artifacts, not inferred:

- The v1 run selected a **delocalized** PSSM: mean per-residue entropy 1.987 bits, mean
  max-probability 0.538, 19/48 residues below 0.5 and 11/48 below 0.3 (lowest 0.179).
  About half the binder positions were not close to any single amino acid.
- v1 ran one soft `simplex_APGM` stage and then a bare `argmax`. Hardening that iterate
  moved the objective **5.647 -> 8.082**, a gap of 2.435, of which **91.4% (2.227) sat in
  the four interface terms** — the signature of soft-sequence exploitation at the interface.
- Unannealed argmax export was unreliable in *every* run in this repository: the bare
  argmax candidate also failed the screen in both earlier monomer runs (2.60 and 3.10 A
  against a 2.0 A gate); only MPNN rescue passed.
- v1's selection rule took the global minimum over an oscillating trajectory (36.7% of
  steps increased the loss; last-50 std 0.847), so the argmin was a noise trough.

v2 therefore uses upstream's own schedule, already present in the pinned Mosaic source
(`b94b9d4`) and previously unused here: `mosaic.optimizers.bindcraft_design`, four
`colabdesign_stage` stages (50 logits / 25 logits / 45 soft-anneal / 5 hard), lr 0.1,
`norm_seq_grad`. **Its final stage evaluates a straight-through one-hot**, so selection is
restricted to hard-stage evaluations and the exported sequence is the sequence that was
evaluated.

## Pre-run numerical checks

The schedule changes the model input path, so it was gated before any long GPU run.

1. **Transform-level CPU/MPS parity** (`check_colabdesign_transform.py`; no weights,
   transform only): 12 cases covering all four stages' `(soft, temp, hard)` settings at two
   logit scales. Zero-gradient row counts agree on every case, and in the regime this run
   operates in (`bindcraft_init_0.01`) the gradient magnitudes agree **exactly** at every
   stage, including the temp-0.01 and hard stages. The straight-through path (`argmax`,
   `one_hot`, `stop_gradient`, `softmax` at temp 0.01 and its VJP) introduces no new MPS
   divergence there. The only difference appears at `evolved_logits_x30` with temp 0.01,
   where the softmax is fully saturated, 44/48 rows are exactly zero on both platforms and
   the gradient magnitude is 1.6e-04: the platforms differ by 1.3e-06 absolute, which is
   float32 noise on a numerically-zero quantity, in a regime v2 avoids by construction
   (see 2).
2. **Initialization scale.** At temp 0.01 the softmax saturates and the gradient vanishes
   on a logit-scale-dependent fraction of residues — measured, identically on CPU and MPS,
   so it is a property of the method and not the backend:

   | logit std | zero-gradient rows / 48 |
   |---|---|
   | 0.01 (BindCraft init) | 0 |
   | 0.5 | 0 |
   | 0.628 (v1's `gumbel*0.5`) | 8 |
   | 4.0 | 30 |
   | 16.0 | 44 |

   Carrying v1's initialization into this schedule would have silently zeroed the gradient
   on 8/48 positions before the run started. v2 uses BindCraft's `0.01 * normal`,
   row-centered. The run records `zero_grad_rows` per evaluation as a guard; it was
   **0 at every one of the 125 evaluations**.
3. **Full-model CPU/MPS gradient parity** at the **unchanged 0.5% gate**, probed at the
   stage-1 pseudo-sequence the schedule actually evaluates (row sums 0.018, off the
   simplex) rather than at a simplex point:

   | | v1 (simplex point) | v2 (stage-1 pseudo-sequence) |
   |---|---|---|
   | raw relative L2 | 0.0479% | **0.0259%** |
   | simplex-tangent relative L2 | 0.0494% | **0.0259%** |
   | loss absolute error | — | 1.9e-06 |

   The probe point is computed in host NumPy so both platforms evaluate a bit-identical
   input; computing it on device made CPU and MPS differ by 1.9e-09 and the harness
   correctly refused the comparison.
4. **v1 regression.** Re-running the v1 path after the code changes reproduces the
   published v1 probe **bit-identically** (sequence, loss and gradient all exactly equal).

## Result

Design + MPNN: 272.3 s (optimization 263.4 s), peak device memory 2.66 GB.
Six Boltz refolds, offline, empty MSA, template-free, balanced profile, FP32 pair ops.

**The relaxed-to-hard gap is eliminated by construction:**

| | v1 | v2 |
|---|---|---|
| selected objective | 5.647 (soft) | 6.382 (hard one-hot) |
| objective of the exported sequence | 8.082 | 6.382 |
| **relaxation gap** | **2.435** | **0.000000** |

AF2 design-side diagnostics (same model the objective was optimized against, so circular
— reported as diagnostics only): binder pLDDT 51.44 -> 60.03, iPTM 0.139 -> 0.385.

### Independent Boltz refolds against the frozen screen

Gates: binder pLDDT >= 80, complex iPTM >= 0.65, binder self-aligned CA RMSD <= 2.0 A,
target-aligned binder CA RMSD <= 3.0 A. Both seeds must pass.

| candidate | seed | pLDDT v1 -> v2 | iPTM v1 -> v2 | self-RMSD (fold) v1 -> v2 | target-aligned (dock) v1 -> v2 |
|---|---|---|---|---|---|
| hallucinated | 7 | 66.41 -> 70.27 | 0.211 -> 0.371 | 3.88 -> **1.68** | 19.37 -> 7.84 |
| hallucinated | 11 | 64.68 -> 69.57 | 0.342 -> 0.248 | 4.25 -> **1.50** | 14.63 -> 25.75 |
| mpnn_1 | 7 | 61.83 -> 76.11 | 0.538 -> 0.561 | 3.31 -> **1.55** | 16.32 -> **2.50** |
| mpnn_1 | 11 | 66.67 -> 75.16 | 0.499 -> 0.644 | 1.81 -> **1.40** | 17.52 -> **2.53** |
| mpnn_2 | 7 | 56.21 -> 69.50 | 0.354 -> 0.354 | 5.02 -> **1.80** | 14.92 -> **2.50** |
| mpnn_2 | 11 | 60.28 -> 74.51 | 0.281 -> 0.452 | 10.59 -> 2.24 | 14.06 -> 4.28 |

Bold marks values inside the gate.

- **Fold reproducibility improved sharply**: 5/6 refolds now meet the 2.0 A self-RMSD gate,
  against 1/6 in v1.
- **Pose reproducibility improved sharply**: 3/6 refolds now meet the 3.0 A target-aligned
  gate, against 0/6 in v1, where every value was 14-19 A. `mpnn_1` meets it on **both**
  seeds (2.50 and 2.53 A).
- Epitope agreement between the design and the independent refolds also rose: `mpnn_1`
  recall 0.56 -> 0.83/0.89, cross-seed Jaccard 0.88 -> 0.94.

### Still 0/3 — what failed

Every candidate failed on the two confidence gates:

| candidate | failed criteria |
|---|---|
| hallucinated | binder pLDDT, complex iPTM, target-aligned RMSD |
| mpnn_1 | binder pLDDT (76.11 / 75.16 vs 80), complex iPTM (0.561 / **0.644** vs 0.65), max single-AA fraction (0.354 vs 0.30) |
| mpnn_2 | binder pLDDT, complex iPTM, homopolymer run (5 vs 4), and seed 11 also fold and pose |

`mpnn_1` is the closest: it satisfies both structural reproducibility gates on both seeds
and misses iPTM by 0.006 on seed 11. Its composition failure is glutamate-rich sequence
from low-temperature MPNN with no composition constraint — a known failure mode that
upstream controls for and this protocol does not.

## Honest reading

The mechanism identified from the v1 data was real, and correcting it produced large
improvements in exactly the predicted directions: the relaxation gap went to zero, binder
folds became reproducible, and binder poses moved from 14-19 A to 2.5 A for one candidate.

It did **not** make the screen pass. That is consistent with a separate v1 measurement:
across all 151 v1 iterates, **0 reached soft pLDDT >= 0.80 and 0 reached soft iPTM >= 0.65**
— the best anywhere in the trajectory was 0.752 and 0.568. The configuration's optimum
sits below the confidence gates, so hardening the schedule cannot by itself reach them.
Remaining candidates for the next bounded experiment, in the order the evidence supports:
the confidence terms carry little weight in the objective (pLDDT 5.7%, iPTM 0.8% of the
objective magnitude, against 59% for the two contact terms); the protocol specifies no
epitope; binder length is 48 against upstream's own ubiquitin example at 70; and MPNN
redesign has no composition constraint. Each should be changed one at a time and
re-measured, not bundled.

## Reproduce

From the repository root, with environments and weights prepared as in the
[parent instructions](../../README.md). Use new output directories and check every exit
status.

```bash
EXP=experiments/mosaic_af2
INPUT="$EXP/ubiquitin"
MODEL_HOME="${MACFOLDKIT_HOME:-$HOME/Library/Caches/macfoldkit}"

bash "$EXP/run.sh" cpu v2-cpu-probe --length 48 --steps 125 --seed 7 --mpnn-weight 5 \
  --optimizer bindcraft --target "$INPUT/ubiquitin-target.cif" \
  --protocol "$INPUT/protocol-v2.json" --probe-only
bash "$EXP/run.sh" mps v2-mps-probe --length 48 --steps 125 --seed 7 --mpnn-weight 5 \
  --optimizer bindcraft --target "$INPUT/ubiquitin-target.cif" \
  --protocol "$INPUT/protocol-v2.json" --probe-only
"$MODEL_HOME/runtimes/mosaic/bin/python" "$EXP/compare_gradients.py" \
  v2-cpu-probe v2-mps-probe v2-gradient-parity.json
# Continue only if the gate passes.

bash "$EXP/run.sh" mps v2-design --length 48 --steps 125 --seed 7 --mpnn-weight 5 \
  --optimizer bindcraft --target "$INPUT/ubiquitin-target.cif" \
  --protocol "$INPUT/protocol-v2.json"
for candidate in hallucinated mpnn_1 mpnn_2; do
  for seed in 7 11; do
    macfoldkit fold "v2-design/$candidate.yaml" "v2-refolds/$candidate" \
      --backend boltz -- --offline --profile balanced --precision float32 --seed "$seed"
  done
done
"$MODEL_HOME/runtimes/boltz/bin/python" "$EXP/evaluate_binder.py" \
  v2-design v2-refolds v2-validation.json \
  --protocol "$INPUT/protocol-v2.json" --target "$INPUT/ubiquitin-target.cif"
```

`--optimizer simplex_apgm` (the default) still reproduces v1 exactly. A `--smoke N` flag
runs N steps per stage for a path-only test; it is refused together with `--protocol` so
the frozen schedule cannot be shortened by accident.
