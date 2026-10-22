# Does the design pipeline run at a realistic target size on this Mac?

Every design experiment here so far used ubiquitin, a 76-residue target giving a 124–146
residue complex. That is small. This records whether the same pipeline is usable against a
realistic therapeutic target on an M5 Pro with 24 GiB unified memory.

**Target: EGFR domain III**, extracted from PDB **1YY9** chain A, residues 310–480 (the
cetuximab-binding domain), 171 residues, complete backbone, no gaps, no duplicate residue
identifiers. Stored here as `egfr-domainIII.cif`.

This is a **feasibility probe only** — one compile plus two gradient evaluations
(`--probe-only`). No optimization was run and no binder was designed.

## Result: it fits, comfortably

| complex | residues | s / gradient step | 125-step design | peak GPU memory |
|---|---|---|---|---|
| ubiquitin (v2) | 124 | 2.10 | 4.4 min | 2.66 GB |
| ubiquitin, 70-mer binder (v3) | 146 | 3.25 | 6.8 min | — |
| **EGFR domain III + 60-mer binder** | **231** | **9.25** | **19.3 min** | **5.42 GB of 24.48 GB** |

Compile plus first gradient: 13.1 s. Host peak RSS: 1.62 GB.

**Memory is not the constraint; time is.** A 1.9x increase in residue count cost 4.4x in
time but only 2.0x in memory. At 5.4 GB of 24.5 GB there is substantial headroom — memory
would allow considerably larger complexes, but at roughly quadratic time cost a much
larger target would push a single design past an hour.

A full EGFR binder design under the v3 protocol would therefore take about 20 minutes per
seed on this hardware, which is practical.

## Gap this exposed

Preparing the target was entirely manual: fetch 1YY9, select chain A, slice residues
310–480, strip ligands, waters and alternate conformations, then verify backbone
completeness and unique residue numbering against what `run_design.py`'s `load_target`
requires. There is no MacFoldKit command for this, and it is the first thing any user
aiming at a real target will need.

## Scope

No binder was designed against EGFR and none is claimed. This measures execution
feasibility — speed and memory — for the design workflow at a realistic target size, on
one machine. It says nothing about whether designs against this target would be any good.

## Reproduce

```bash
EXP=experiments/mosaic_af2
bash "$EXP/run.sh" mps egfr-probe --length 60 --steps 125 --seed 7 --mpnn-weight 5 \
  --optimizer bindcraft --target "$EXP/egfr-feasibility/egfr-domainIII.cif" --probe-only
```

`probe.json` records timings, `probe_device_memory_stats` and `probe_max_rss_bytes`, so a
target's feasibility on a given machine can be checked before committing to a full run.
