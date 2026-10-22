# What the binder screen actually measures

The ubiquitin binder experiments (`../ubiquitin/`) filter candidates with a fixed screen:
binder pLDDT >= 80, complex iPTM >= 0.65, and pose/fold RMSD gates, all computed from an
independent offline Boltz2 refold with an empty MSA and no template.

That screen had never been run against a control. These are the controls. **The short
version: it is a designability and self-consistency filter, not a binding predictor.**

## Setup

All predictions use the identical settings the screen uses — offline, empty MSA on both
chains, template-free, balanced profile, float32 pair operations, seeds 7 and 11 — against
the same 76-residue ubiquitin target as every design.

- **Positive control**: the HHR23A UBA(2) domain, a real, experimentally characterized
  ubiquitin-binding domain, taken from PDB **1P3Q** chain Q (selenomethionine mapped to
  methionine): `SSLIKKIEENERKDTLNTLQNMFPDMDPSLIEDVCIAAASGPCVD`.
- **Negative controls**: composition-preserving shuffles (seeded, deterministic) of that
  UBA sequence and of `v3 len48-seed8 mpnn_2`, the design that passed the screen.

## Result

| system | seed | binder pLDDT | complex iPTM | verdict vs gates |
|---|---|---|---|---|
| **positive** — real UBA(2) binder | 7 / 11 | 70.91 / 66.78 | 0.426 / 0.301 | **fails both** |
| negative — UBA shuffled | 7 / 11 | 44.01 / 47.26 | 0.321 / 0.217 | fails both |
| negative — design shuffled | 7 / 11 | 52.24 / 55.28 | 0.284 / 0.285 | fails both |
| reference — our passing design | 7 | 93.57 | 0.951 | passes |

Against the experimental complex (1P3Q chain Q paired with chain **V** — verified by
heavy-atom contact count, since chain Q contacts V with 113 contacts and U with only 40):

| seed | ubiquitin aligned | UBA fold, self-aligned | UBA pose vs experiment |
|---|---|---|---|
| 7 | 0.83 Å | 3.23 Å | **11.49 Å** |
| 11 | 0.63 Å | 4.05 Å | **17.00 Å** |

## Reading

1. **The screen has real discriminating power against nonsense.** Shuffled sequences score
   44–55 pLDDT against the passing design's 93.6. It is not vacuous.
2. **Confidence tracks correctness.** Boltz got ubiquitin itself right (0.6–0.8 Å) but
   misdocked the UBA domain by 11–17 Å, and reported iPTM 0.30–0.43 for it. The gates
   correctly rejected a wrong prediction; they did not reject a correct one.
3. **Passing does not mean binding.** Our designed binder outscores a genuine
   ubiquitin-binding domain by a wide margin (iPTM 0.951 vs 0.426). That gap reflects how
   confidently structure predictors handle idealized designed proteins versus natural,
   weak, transient interfaces — not relative affinity.
4. **Failing is weak evidence about a design**, because the evaluator demonstrably fails to
   recover a real complex in this regime. Candidates rejected by this screen have not been
   shown not to bind.

## Limits of these controls

UBA–ubiquitin is a deliberately hard case: isolated UBA domains bind ubiquitin weakly and
transiently, which is exactly the regime single-sequence structure prediction handles
worst. One failed positive control does not establish that the evaluator fails in general.

The empty-MSA, template-free setting is a deliberate choice — it keeps the screen
reproducible, offline, and free of alignment leakage from the target's known partners. It
is also the most likely cause of the positive control's failure. Whether supplying a real
MSA recovers the 1P3Q complex is an obvious and untested follow-up; it would trade
reproducibility for accuracy.

## Reproduce

```bash
MODEL_HOME="${MACFOLDKIT_HOME:-$HOME/Library/Caches/macfoldkit}"
for N in pos_uba_real neg_uba_scrambled neg_design_scrambled; do
  for S in 7 11; do
    macfoldkit fold "experiments/mosaic_af2/screen-controls/$N.yaml" "controls/$N" \
      --backend boltz -- --offline --profile balanced --precision float32 --seed "$S"
  done
done
```

The structural comparison uses PDB 1P3Q (fetched from RCSB); pair chain Q with chain V,
match residues positionally, align on ubiquitin, then measure the binder.
