# Release validation: 0.1.0a1

Tested September 28, 2026 on an Apple M5 Pro (20 GPU cores, 18 CPU cores),
24 GiB unified memory, macOS 26.6.2. This is a single-machine alpha validation,
not a claim of compatibility with every Apple Silicon Mac.

## Fresh backend installations

All three `macfoldkit setup` commands succeeded in a new runtime root. Each
backend used newly created Python 3.12 environments and its packaged dependency
snapshot; source checkouts matched the pinned commits. The Mosaic patch also
passed a reverse `git apply --check` against the patched checkout.

Existing verified chemistry assets and model weights were reused to avoid
re-downloading multiple GB. This tests fresh dependency/source installation,
not a completely empty asset-cache download. `macfoldkit fetch boltz` completed
against this cache and resolved the pinned model snapshot. Unit tests cover
checksum rejection, partial-download cleanup and verified-cache reuse.

## GPU integration results

The frontend launched each backend in its fresh isolated environment. These are
individual runs, not distributions or direct comparisons between model speeds.

| Workflow | Settings | Observed result |
|---|---|---|
| Mosaic / ProteinMPNN | 1UBQ, 76 residues, 25 steps, seed 7, float32 | Loss 3.97534 → 1.80127; directional finite-difference relative error 2.01e-6; 5.80 s measured workflow; 0.536 GiB peak RSS |
| Boltz2 / FastPLMs | 76-residue redesign, empty MSA, balanced profile, seed 7, BF16 pair updates | pLDDT 94.443/100; pTM 0.92922; 3.377 s model call, 13.546 s load plus model |
| ColabFold 1.6.3 | Ubiquitin, local A3M, AF2-pTM model 1, one recycle, seed 7, MSA 128:256 | pLDDT 94.4/100; pTM 0.812; 11.6 s reported prediction, 39.52 s worker process; 2.845 GiB peak RSS |

Boltz's preprocessed-feature SHA256 was
`2dd9803684005f3c70a06af595ea81b2d579b21362778923eb0a9250a1a3c466`,
matching the earlier candidate run. Its confidence scores also reproduced.
ColabFold audited actual model outputs on `MPS:0` for both recycle passes and
reported no failed queries. CPU feature preprocessing remains part of the pipeline.

Timing boundaries differ: Boltz's model timer excludes preprocessing, Python
startup and downloads; Mosaic's workflow timer excludes initial module imports;
ColabFold's worker timer includes startup and feature processing. Peak RSS is
not a measurement of all system-wide unified-memory use. No new Boltz peak-memory
measurement was taken for this release smoke test.

## Earlier research evidence

Before packaging, the same underlying Boltz runner was tested on monomers and
complexes up to 427 residues. The final 427-residue balanced run recorded 22.27 s
for the synchronized model call, 32.40 s including model loading (excluding
preprocessing), and 6.94 GiB peak process footprint. Complex C-alpha RMSD was
0.723 Å to the experimental reference. These historical timings are not a fresh
installer benchmark and do not establish general accuracy or NVIDIA parity.

The 76-residue Mosaic redesign changed 36 positions and refolded with Boltz2 to
1.095 Å C-alpha RMSD relative to the input backbone, with pLDDT 94.44. Raw CPU/MPS
sequence gradients agreed to relative L2 error 1.28e-6. This is one in-silico
example; neither function nor experimental stability was established.

## Reproduction

From a source checkout, after setup and fetching the relevant assets:

```bash
macfoldkit design examples/1ubq.cif design-results --steps 25 --seed 7
macfoldkit fold examples/redesign.yaml boltz-results --backend boltz -- --offline
macfoldkit fold ubiquitin.a3m af2-results --backend colabfold -- --msa-mode mmseqs2_uniref_env --num-recycle 1
```

The local A3M used for ColabFold is not bundled; its timing and prediction are
not reproducible from the FASTA alone without generating an alignment. No MSA
search was performed during this integration test.

CPU package checks use `python -m unittest discover -s tests -v`, followed by
`python -m build` and `python -m twine check dist/*`. GitHub Actions runs those
checks on Linux; it does not validate Metal execution. GPU results above were
measured locally on the Mac.

All 29 CPU tests passed. The wheel and source distribution passed metadata
validation, and the wheel contains all four dependency snapshots, backend
runners and required license notices. Installed into a separate minimal Python
environment, the wheel's CLI reported the correct version and runtimes and
completed the 25-step CPU design example. That run also succeeded with
`PYTHONOPTIMIZE=1` in the caller: the launcher removes that setting so upstream
assertions remain active, and the adapter's numerical checks use explicit errors.
