# Ubiquitin binder design, v3: pre-registered seed set at two binder lengths

Computational workflow experiment. **3 of 18 candidates passed the frozen screen — the
first passes in this project. This is computational self-consistency, not evidence of
binding.** No affinity, specificity, expression or stability is claimed or measured.

v1 ([`../`](../README.md)) and v2 ([`../v2/`](../v2/README.md)) are preserved unchanged.

## Design of the experiment

Two arms, `protocol-v3-len48.json` and `protocol-v3-len70.json`, differing **only** in
binder length (48 vs 70). Length 70 is upstream's own choice for this exact target in
`examples/esmfold_minibinder.py`. Everything else is inherited from v2: the
`bindcraft_design` schedule, all ten loss terms and weights, the target, the MPNN
redesign, the Boltz refolds, and a **byte-identical screen**.

Both protocols **pre-register seeds 7, 8 and 9** before any run, and every outcome is
reported here, passing or failing. `run_design.py` enforces this: a seed outside the
declared set is rejected. This is what separates a campaign from seed-shopping.

Six designs, 18 candidates, 36 independent Boltz refolds. CPU/MPS gradient parity was
re-measured for the new length-70 model path before any long run: **0.159% raw / 0.154%
simplex-tangent** against the unchanged 0.5% gate (length 48 measured 0.0259%).

## The result the experiment actually produced

**Seed dominates. The length effect it was built to measure is not resolvable underneath it.**

Per design, median over that design's six refolds:

| binder length | seed 7 | seed 8 | seed 9 | passing |
|---|---|---|---|---|
| 48 | pLDDT 72.4, pose 3.40 Å | **pLDDT 92.5, pose 2.00 Å** | pLDDT 62.3, pose 20.7 Å | 1 / 9 |
| 70 | pLDDT 67.9, pose 21.8 Å | **pLDDT 90.9, pose 2.03 Å** | pLDDT 80.6, pose 33.7 Å | 2 / 9 |

Every pass came from **seed 8, in both arms**. Seeds 7 and 9 produced nothing in either
arm. The between-seed spread is roughly 30 pLDDT points and 2–34 Å of pose; any length
effect is smaller than that, and with three designs per arm it cannot be separated from
noise. The pooled per-refold medians (len48 pLDDT 75.6 / pose 3.40 Å; len70 pLDDT 80.6 /
pose 19.7 Å) should **not** be read as a length effect: 18 refolds per arm come from only
3 independent trajectories, so pooling overstates precision.

Outcomes are close to bimodal. A design either converges to something self-consistent
(pLDDT 91–94, iPTM 0.86–0.95, pose 0.5–2.4 Å) or fails clearly. There is little middle.

## The passing candidates

| design | candidate | refold seed | pLDDT | iPTM | self-RMSD | target-aligned |
|---|---|---|---|---|---|---|
| len48-seed8 | mpnn_2 | 7 / 11 | 93.57 / 93.57 | 0.951 / 0.947 | 0.46 / 0.41 | 1.16 / 0.54 |
| len70-seed8 | mpnn_1 | 7 / 11 | 92.82 / 92.90 | 0.892 / 0.878 | 0.40 / 0.42 | 1.65 / 1.18 |
| len70-seed8 | mpnn_2 | 7 / 11 | 90.56 / 91.22 | 0.858 / 0.875 | 1.21 / 1.04 | 2.40 / 2.42 |

Gates: pLDDT ≥ 80, iPTM ≥ 0.65, self-RMSD ≤ 2.0 Å, target-aligned ≤ 3.0 Å, both refold
seeds must pass, plus interface-contact, target-drift and composition criteria.

Checks run against these before reporting them:

- **They are not copies of the target.** Maximum ungapped identity to ubiquitin is 12.5%
  and 10.0%, i.e. background.
- **The epitope is the biophysically expected one.** Contacts fall on ubiquitin residues
  6–10, 42, 44, 47–49, 68–73 — the Ile44 hydrophobic patch and β-sheet face, the surface
  natural ubiquitin-binding domains use. 67–71% of contacted residues lie in that patch,
  and the independent refold agrees with the AF2 design epitope at Jaccard 0.80–0.94.
- **They are not three independent successes.** `len70-seed8` `mpnn_1` and `mpnn_2` are
  **64.3% identical** — two ProteinMPNN samples from one backbone. The three passing
  candidates represent **2 independent design trajectories**, both from seed 8.
- No interchain clashes; minimum heavy-atom distances 1.74–2.40 Å; target drift
  0.74–1.48 Å, inside the 2.0 Å gate.

## Is the pLDDT ≥ 80 gate fair?

Yes, and this was checked rather than assumed. Natural ubiquitin, predicted in the **same**
empty-MSA, template-free single-sequence mode as part of these very complexes, scores
**88.71 mean pLDDT** (84% of its residues ≥ 80). So 80 is reachable in this prediction
mode, and the v1/v2 binders at 56–76 were genuinely less confidently folded than a natural
protein — not victims of a strict measurement.

## What this does and does not establish

It establishes that the pipeline can produce designs that are **self-consistent under an
independent structure predictor**: a sequence designed against AF2 is refolded by Boltz2,
from sequence alone with no MSA and no template, into the same fold (0.4 Å) and the same
docked pose (0.5–2.4 Å) across two prediction seeds, on the expected epitope.

It does **not** establish binding. AF2 and Boltz2 are both deep-learning structure
predictors and may share biases, so agreement between them is necessary, not sufficient.
The success rate — 2 of 6 independent trajectories — rests on a single favourable seed and
is a very noisy estimate. Nothing here is experimental evidence.

## What this changes about how to run the next experiment

Single-seed comparisons are uninformative for this pipeline. The seed-to-seed spread
exceeds the effect size of every protocol change tested so far, which means v1's and v2's
n=1 outcomes could not have supported conclusions about their protocols either — and the
honest reading of v2's 0/3 is that it was one draw, not a verdict.

The next experiment should be a **campaign**, not another single-seed tweak: many
pre-registered initializations, all outcomes reported, success rate as the measured
quantity. Upstream already provides the machinery — `mosaic.optimizers.biohub_optimizer`
is batched over B parallel initializations for exactly this reason, and selects within a
tail window by a ranking metric rather than by the training loss. Protocol changes should
then be evaluated as a shift in success rate over a seed set, not as a single pass/fail.

## Reproduce

From the repository root, with environments and weights prepared as in the
[parent instructions](../../README.md). Use new output directories and check every exit
status. Length 70 has its own gradient gate because it is a different model path.

```bash
EXP=experiments/mosaic_af2
INPUT="$EXP/ubiquitin"
MODEL_HOME="${MACFOLDKIT_HOME:-$HOME/Library/Caches/macfoldkit}"

for LEN in 48 70; do
  bash "$EXP/run.sh" cpu "v3-len$LEN-cpu" --length "$LEN" --steps 125 --seed 7 \
    --mpnn-weight 5 --optimizer bindcraft --target "$INPUT/ubiquitin-target.cif" \
    --protocol "$INPUT/protocol-v3-len$LEN.json" --probe-only
  bash "$EXP/run.sh" mps "v3-len$LEN-mps" --length "$LEN" --steps 125 --seed 7 \
    --mpnn-weight 5 --optimizer bindcraft --target "$INPUT/ubiquitin-target.cif" \
    --protocol "$INPUT/protocol-v3-len$LEN.json" --probe-only
  "$MODEL_HOME/runtimes/mosaic/bin/python" "$EXP/compare_gradients.py" \
    "v3-len$LEN-cpu" "v3-len$LEN-mps" "v3-len$LEN-parity.json"
  # Continue only if the gate passes.
  for SEED in 7 8 9; do
    bash "$EXP/run.sh" mps "v3-len$LEN-seed$SEED-design" --length "$LEN" --steps 125 \
      --seed "$SEED" --mpnn-weight 5 --optimizer bindcraft \
      --target "$INPUT/ubiquitin-target.cif" \
      --protocol "$INPUT/protocol-v3-len$LEN.json"
    for C in hallucinated mpnn_1 mpnn_2; do
      for S in 7 11; do
        macfoldkit fold "v3-len$LEN-seed$SEED-design/$C.yaml" \
          "v3-len$LEN-seed$SEED-refolds/$C" --backend boltz -- \
          --offline --profile balanced --precision float32 --seed "$S"
      done
    done
    "$MODEL_HOME/runtimes/boltz/bin/python" "$EXP/evaluate_binder.py" \
      "v3-len$LEN-seed$SEED-design" "v3-len$LEN-seed$SEED-refolds" \
      "v3-len$LEN-seed$SEED-validation.json" \
      --protocol "$INPUT/protocol-v3-len$LEN.json" --target "$INPUT/ubiquitin-target.cif"
  done
done
```

Each design takes about 4.5 minutes at length 48 and 7 minutes at length 70 on an M5 Pro
(warm gradient 2.1 s and 3.25 s per step respectively); the whole campaign including
refolds and evaluation ran in 41 minutes.
