# ColabFold on Apple Silicon

MacFoldKit's ColabFold runner uses ColabFold 1.6.3, AlphaFold-ColabFold 2.3.20,
JAX 0.11.2, and JAX-MPS 0.11.0. CPU code prepares features; the runner checks
that actual model outputs and atom coordinates live on the Apple GPU. It fails
if predictions are missing, ColabFold reports failed queries, or that device
audit fails. This is an experimental Mac backend, not upstream ColabFold's
supported CUDA environment.

The bounded batching optimization retains the original model equations and
precision. It widens selected column-attention and transition batches for the
measured BF16 shapes up to 427 residues, with per-tensor memory budgets. Other
shapes retain the upstream batching behavior. Different batch sizes may select
different GPU kernels and change rounding; bitwise equivalence is not promised.
`--no-optimized-batching` disables it. Experimental custom Metal attention and
the separately rebuilt M5 backend are not enabled by this runner.

## Runtime and weights

The frontend launches `runtimes/colabfold/bin/python` inside `MACFOLDKIT_HOME`.
If run directly without that variable, the runner uses
`~/Library/Caches/macfoldkit` on macOS for its weight and cache locations.
`--data DIR` overrides the weight root; AlphaFold parameters belong in
`DIR/params/`, not directly in `DIR`.

`src/macfoldkit/data/colabfold.lock` records the complete installed distribution
snapshot from the tested Python 3.12 environment. It has exact versions and no
machine-local paths. It must be installed into an isolated environment with
`--no-deps`: ColabFold's declared dependency constraints select an older JAX,
whereas the tested Apple GPU plugin uses the versions in this snapshot. This
snapshot is not a hash-verified dependency lock, and a resolved install is not
proof of GPU compatibility. Do not apply it to an existing ColabFold environment.

The package does not redistribute model weights. ColabFold's own downloader
fetches absent weights from their original provider when prediction starts.
The optional [local job queue](jobs.md) instead requires cached weights before
submission and prevents any model download or MMseqs2 search during execution.
For an explicit download stage, its Python API is:

```python
from pathlib import Path
from colabfold.download import download_alphafold_params

# Run this in the isolated ColabFold runtime. Use the actual MACFOLDKIT_HOME.
data = Path("/your/macfoldkit/home/weights/colabfold")
download_alphafold_params("alphafold2_ptm", data)
# Download separately if complex prediction is needed:
download_alphafold_params("alphafold2_multimer_v3", data)
```

These downloads are large and require network access. AlphaFold weights have
separate terms from the software; see the upstream
[AlphaFold license notice](https://github.com/google-deepmind/alphafold#license)
and [ColabFold repository](https://github.com/sokrypton/ColabFold).

## Input and external services

Both an input path and output directory are required. The default is
`--msa-mode single_sequence`, so ordinary sequence input does **not** initiate
an external MSA search. This is a convenience and privacy default, not a claim
that single-sequence predictions have the same quality as MSA-backed results.
The full process may still download missing weights.

To request the public MSA service, explicitly pass
`--msa-mode mmseqs2_uniref_env` (or `mmseqs2_uniref`). This sends sequences to the
configured service when an input alignment is absent; upstream service terms
apply. Template options can also invoke external services.

**Existing A3M alignments also need `--msa-mode mmseqs2_uniref_env`:** ColabFold's
`single_sequence` mode intentionally discards their alignment rows. A complete
local A3M input bypasses the MSA search despite the mode name. For a directory
with mixed FASTA and A3M files, FASTA entries can still invoke the service.

Extra ColabFold arguments pass through unchanged. For a complex, supply a
ColabFold-compatible multi-chain FASTA (chains separated by `:`) or multimer
A3M and pass `--model-type alphafold2_multimer_v3`. Defaults are model 1 only,
three recycles, seed 7, no relaxation, and maximum MSA `1:1` for single-sequence
mode or `128:256` otherwise. Override those settings explicitly as needed.
`--zip` is unsupported because output verification requires the exported PDBs.

## Local templates and initial guesses

For two domains joined by a linker in **one polypeptide**, submit one
continuous FASTA sequence (`domain 1 + linker + domain 2`), not two chains
separated by `:`. ColabFold can align separate local domain structures to
different parts of that sequence. Put **copies** of their PDBs in a writable
directory, with four-character alphanumeric, PDB-ID-like basenames such as
`1abc.pdb` and `2def.pdb`:

```bash
mkdir -p templates
cp domain1.pdb templates/1abc.pdb
cp domain2.pdb templates/2def.pdb
macfoldkit fold construct.fasta results --backend colabfold -- \
  --local-only --msa-mode single_sequence --templates \
  --custom-template-path templates --max-msa 5:1
```

This uses local HHsearch and cached weights without an MSA search or remote
template-service request. `--templates` **without** `--custom-template-path`
can query an external template service; `--local-only` makes an attempted
MMseqs2 search or weight download fail. ColabFold converts PDBs to CIF and
builds `pdb70*` index files in the supplied template directory; do not point
it at irreplaceable originals. Check `results/*_template_domain_names.json`
to see which structures actually matched. In this pinned build, the normal `1:1`
single-sequence MSA default fails during feature preparation when templates
are enabled; pass `--max-msa 5:1` as above. For a complete local A3M, use
`--msa-mode mmseqs2_uniref_env` instead to preserve its rows, together with
`--local-only` and a local template directory.

An **initial guess** is different: it seeds the model's starting coordinates
from **one full-length** `.pdb` or `.cif` containing the entire construct in
the same residue order as the query, including the linker. It does not
automatically join two domain PDBs or fill in missing linker coordinates:

```bash
macfoldkit fold construct.fasta guess-results --backend colabfold -- \
  --local-only --initial-guess assembled.pdb
```

You can add `--initial-guess assembled.pdb` to the local-template command to
use **both** in one run. An initial guess is not a coordinate restraint;
neither it nor two separate domain templates determines the domains' relative
orientation across a flexible linker. Examine interdomain PAE and alternative
predictions rather than treating one pose as established. These flags are
available through the standalone `fold` command, **not** the [local jobs
queue](jobs.md).

## Outputs

Each output directory includes `process.log`, `benchmark.json`, and
`model-device-audit.jsonl`. Reusing an output directory containing predictions
requires the explicit upstream `--overwrite-existing-results` option.
Queued runs place these under a per-job, per-attempt output directory; the
[queue input/output contract](jobs.md#input-contract) defines its accepted
formats, job JSON, and artifact paths.

## Validation scope

This release transplants the previously measured runner and batching policy;
packaging changes concern paths, required inputs, and the MSA default. CPU
mock tests cover those routing changes without loading models. They do not
replace a clean installation and actual GPU prediction test on the destination
Mac. Existing hardware benchmarks were obtained on an M5 Pro with 24 GiB unified
memory; other Macs and larger inputs need separate validation.
