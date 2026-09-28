# MacFoldKit

**Protein folding and design, optimized for Apple Silicon.**

MacFoldKit brings tested Mac settings for Boltz2, ColabFold and an experimental
Mosaic design workflow behind one command. Each model gets its own pinned Python
environment, keeping incompatible Torch, JAX and NumPy requirements separate.

**Alpha release:** validated on an M5 Pro with 24 GB unified memory, macOS 26.6.2.
Other Apple Silicon configurations are unvalidated. This is an integration and
optimization toolkit; the models are developed by the upstream projects credited
below. The first release uses ordinary MPS/JAX-MPS paths. Experimental custom
native Metal kernels are not part of the installer.

| Backend | Included workflow | Status |
|---|---|---|
| Boltz2 via FastPLMs | Protein monomers and complexes; compiled pair updates, bounded attention, BF16 pair work | Tested; upstream FastPLMs implementation remains provisional |
| ColabFold 1.6.3 | AlphaFold2 inference on JAX-MPS, bounded Evoformer batching, GPU output audit | Tested; community GPU backend is experimental |
| Mosaic / ProteinMPNN | Redesign every position of a single fixed protein backbone | Experimental; not a binder-design pipeline |

## Install

Requires Apple Silicon, native arm64 Python 3.12, [uv](https://docs.astral.sh/uv/)
and Git. Boltz compilation also requires Xcode's Metal compiler (`xcrun --find
metal`). This release is available from GitHub; it is **not published on PyPI**.

```bash
uv tool install --python 3.12 'git+https://github.com/Vedasheersh/macfoldkit.git@v0.1.0a1'
macfoldkit doctor
macfoldkit setup boltz
macfoldkit fetch boltz
```

`setup` downloads pinned source and Python dependencies. `fetch` downloads model
weights and chemistry data—several GB for Boltz, with additional disk space for
extraction. Downloads retain their upstream licenses. Runtimes and caches live
in `~/Library/Caches/macfoldkit`. Set `MACFOLDKIT_HOME` or put `--home PATH`
before the subcommand to choose another location. Paths containing spaces work.

For a source checkout, use `uv tool install --python 3.12 .` from its root.

## Fold a protein

Create `input.yaml`:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
      msa: empty
```

```bash
macfoldkit fold input.yaml results --backend boltz -- --offline
```

`msa: empty` runs without an alignment search. For complexes, good paired MSAs
can be crucial. Supply local A3M paths, or omit `msa` and explicitly pass
`--allow-msa-server` to upload sequences to the public ColabFold MSA service.
`--offline` prevents model downloads and MSA requests for the Boltz adapter.

The balanced profile uses one recycle setting (two trunk passes), 100 diffusion
steps and up to 1,024 ordered MSA rows. `--profile quality` uses more computation;
`--precision float32` overrides pair precision. These settings change the
speed/quality tradeoff. See [Boltz details](docs/boltz.md).

## ColabFold on the GPU

```bash
macfoldkit setup colabfold
macfoldkit fetch colabfold
macfoldkit fold sequence.fasta af2-results --backend colabfold
```

The default is single-sequence inference with one AF2 model and three recycles.
Explicit `--msa-mode mmseqs2_uniref_env` permits MSA search for raw sequences.
For an existing A3M, that mode also preserves its alignment rows. Multimer
weights are fetched separately with `fetch colabfold --model-type
alphafold2_multimer_v3`; forward the matching `--model-type` when folding.
See [ColabFold details](docs/colabfold.md). The Boltz `--offline` flag is not a
ColabFold option; cached weights plus local A3M/single-sequence inputs avoid MSA
requests, but this wrapper does not implement a blanket network prohibition.

## Experimental fixed-backbone design

```bash
macfoldkit setup mosaic
macfoldkit design backbone.pdb design-results --steps 25 --seed 7
```

Accepts a single protein chain with complete N/CA/C/O atoms. All positions are
redesigned; there are no fixed-position or binder constraints in this interface.
Outputs include a candidate FASTA, numerical checks and an optimization report.
The deterministic fixed-order ProteinMPNN objective is a starting point, not a
complete design-quality filter. Refold candidates and assess the intended task.
`--device cpu` provides a reference path. See [design scope](docs/design.md).

A separate [full AF2-guided design experiment](experiments/mosaic_af2/README.md)
completes de novo monomer optimization, ProteinMPNN redesign and independent
Boltz2 refolding on the Mac. It remains outside the released CLI because full
CPU/MPS gradients differ by roughly 3% and design-quality validation is limited.
See also the [BindCraft2 Mac feasibility audit](docs/bindcraft2-feasibility.md).

## Evidence

The underlying optimized Boltz runner was tested on monomers and complexes up to
427 residues on the M5 Pro. The 427-residue balanced run took 22.27 seconds for
the model call at 6.94 GiB peak process footprint, with 0.723 Å complex C-alpha
RMSD to its experimental reference. These are specific research runs, not a
general performance or accuracy guarantee and not a same-settings comparison
against every upstream implementation.

A 76-residue Mosaic redesign with 36 substitutions refolded with Boltz2 to
1.095 Å C-alpha RMSD, with predicted pLDDT 94.44. Raw CPU/GPU sequence gradients
agreed to relative L2 error 1.28e-6. This is an in-silico example, not evidence
of preserved biological function. [Release validation](docs/validation.md)
separates those earlier results from tests of the packaged installer.

## Contributing and upstream patches

The frontend owns installation, adapters and reproducible Mac settings. General
fixes should be contributed to their upstream projects. A version-pinned Mosaic
import patch is included under [patches](patches/README.md); it has not yet been
submitted upstream. We avoid maintaining complete copies of every model.

See [CONTRIBUTING.md](CONTRIBUTING.md), [third-party notices](THIRD_PARTY_NOTICES.md)
and [limitations](docs/limitations.md). MacFoldKit code is Apache-2.0 except where
individual third-party notices specify otherwise. Model code, weights and data
retain their own terms. MacFoldKit is independent of Apple and the upstream
model developers.
