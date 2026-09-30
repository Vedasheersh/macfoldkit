# Boltz2 on Apple Silicon

The Boltz backend runs the pinned Synthyra/FastPLMs Boltz2 checkpoint on PyTorch
MPS. Official Boltz preprocessing runs in a separate CPU environment. The runner
accepts protein-only Boltz version 1 YAML, including multiple protein chains.
Ligands, nucleic acids, modified or cyclic proteins, templates, restraints, and
affinity prediction are not supported by this adapter.

## Input and network use

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
      msa: empty
```

Use `msa: empty` for a single-sequence prediction, or a path to a local alignment
(relative to the YAML file). Omitting `msa` requests a public MSA search, which
requires the explicit `--allow-msa-server` flag because sequences are uploaded to
https://api.colabfold.com. Cached features can be reused without upload permission.
`--offline` prevents MSA searches and model downloads; assets must already exist.
An MSA is particularly important for many complexes: a successful execution or
high model confidence does not establish a correct interface.

## Profiles

| Profile | Recycling setting | Diffusion steps | Pair precision |
|---|---:|---:|---|
| `balanced` (default) | 1 | 100 | BF16 |
| `fast` | 0 | 50 | BF16 |
| `quality` | 3 | 200 | FP32 |

A recycling setting of N executes N + 1 trunk passes. Every profile retains FP32
weights and residual storage, compiles shared pair updates with PyTorch Inductor,
and uses 32-row triangle attention chunks. Diffusion remains FP32. `--precision
float32` overrides pair precision. First-use compilation may increase latency.
The quality profile spends more computation but does not guarantee a better result.

The default `--msa-rows 1024` takes an ordered prefix of the official prepared MSA,
retaining paired and highly ranked unpaired rows first. Full prepared features are
cached. `--seed` defaults to 7; official preprocessing uses seed 42. Neither these
settings nor the frozen software ensure identical results across hardware or
changing public MSA databases.

## Runtime provenance

- FastPLMs source: `Synthyra/FastPLMs` at
  `d642641b1dcbd58b21863cb76118b2996c617d1f` (Apache-2.0 code).
- Model: `Synthyra/Boltz2` at
  `bce98f7ce914d468182726b5a0fbd23737167875`. FastPLMs still labels this model
  provisional; native end-to-end equivalence with the official implementation
  has not been established.
- Preprocessor: `colbyford/boltz` at
  `1a56ef48739835d8863ec4666644bf7a7d85bc14`, package version 2.2.1 (MIT).
- Tested inference runtime: Python 3.12, PyTorch 2.14.0, Transformers 5.13.0.
  The package locks deliberately record this tested runtime instead of
  FastPLMs' different upstream Torch requirement. Inference and preprocessing
  have separate locks because their NumPy and dependency requirements differ.

FastPLMs is source code, not an installable Python distribution. Its pinned
`src` directory is supplied through `PYTHONPATH`. Model loading executes the
pinned Hugging Face runtime via `trust_remote_code=True`. Model weights and
upstream source are fetched from their respective providers rather than shipped
inside this package. Consult their licenses separately from MacFoldKit's license.

The preprocessor molecule archive is `boltz-community/boltz-2`'s `mols.tar` at
revision `6fdef46d763fee7fbb83ca5501ccceff43b85607`, SHA-256
`39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7`.
Boltz2 preprocessing uses this directory rather than Boltz1's CCD pickle.
FastPLMs additionally needs `biohub/ESMFold2`'s `ccd.pkl` at revision
`1ebf0e3481a5184eb6171d40615c79e384b48796`, SHA-256
`9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5`.
The CCD is verified before deserialization. A regular file supplied through
`ESMCFOLD_CCD_PATH` avoids incompatibility with shared Hugging Face blob symlinks.
For an authenticated Hugging Face mirror, set `HF_ENDPOINT` (HTTPS) and `HF_TOKEN`
when fetching; the chemistry assets follow that endpoint and retain their
checksum checks. The mirror must also contain the pinned `Synthyra/Boltz2` model
for the folding runner. Credentials are not stored by MacFoldKit.

## Validation boundaries

The research implementation was tested on an M5 Pro with 24 GiB unified memory
on proteins and complexes up to 427 residues. This is a tested scope, not a hard
maximum or a guarantee for every target. Other Apple Silicon generations and
memory sizes need independent validation. The initial package uses ordinary
PyTorch MPS compilation; experimental custom Metal kernels are not enabled.

Results include a validated protein mmCIF and a JSON report with model revision,
input and feature digests, chain information, settings, confidence, timing, and
runtime version. CIF chain numbering and entity metadata are repaired without
changing coordinates so downstream structural tools can read them correctly.
Existing prediction files are never overwritten; choose another result directory
or seed. `--validate-only` performs lightweight input validation. `--prepare-only`
prepares features without loading the folding model.
