# Third-party notices

MacFoldKit integrates models developed by other projects. Our wrappers and CLI
are Apache-2.0. This does not relicense their code, checkpoints or datasets.

| Project | Role | Code license / source |
|---|---|---|
| Synthyra FastPLMs | Boltz inference implementation | [Apache-2.0](https://github.com/Synthyra/FastPLMs/blob/d642641b1dcbd58b21863cb76118b2996c617d1f/LICENSE) |
| Boltz / colbyford fork | Official-style features and molecule preprocessing | [MIT](https://github.com/colbyford/boltz/blob/1a56ef48739835d8863ec4666644bf7a7d85bc14/LICENSE) |
| ColabFold | AlphaFold2 inference pipeline | [Upstream licenses](https://github.com/sokrypton/ColabFold) |
| AlphaFold | Structure predictor, distributed through alphafold-colabfold | [Apache-2.0 code](https://github.com/google-deepmind/alphafold/blob/main/LICENSE); weight terms separate |
| JAX-MPS | Community Apple GPU backend | [Apache-2.0](https://github.com/tillahoffmann/jax-mps) |
| Mosaic | Modular design and optimization | [MIT](https://github.com/escalante-bio/mosaic/blob/b94b9d4eb9907a700a6d78ed2d29d3704c5df46c/LICENSE) |
| ProteinMPNN | Fixed-backbone sequence model | [MIT, Justas Dauparas](https://github.com/dauparas/ProteinMPNN) |
| Joltz | Torch-to-JAX conversion utility | [MIT](https://github.com/nboyd/joltz/tree/ed0f04257dac85bd4b7bf45521cc280f86d38ded) |

The only vendored upstream implementation in the wheel is Joltz's conversion
utility at `src/macfoldkit/runners/mosaic/_conversion.py`, copied unchanged from
commit `ed0f04257dac85bd4b7bf45521cc280f86d38ded`. Its SHA256 is
`f73d1a8ce0ed3200234f049b6e5f2d8109ad7a56e193d3ec6afabff19e11cc3c`.
The adjacent `JOLTZ_LICENSE` and `JOLTZ_NOTICE` retain Nick Boyd's MIT notice and
the upstream Boltz attribution. `MOSAIC_LICENSE` also accompanies this adapter.
Downloaded Mosaic source retains its ProteinMPNN, AlphaFold and data notices.

The example `1ubq.cif` is copied from Mosaic's tests and represents the public
ubiquitin structure [PDB 1UBQ](https://www.rcsb.org/structure/1UBQ). It is a
numerical/structural example, not a newly solved structure.

Model checkpoints and chemistry databases are fetched from their upstream
providers; their licenses may differ from implementation licenses. Preserve the
license files accompanying downloaded artifacts. See FastPLMs' source and weight
license inventory rather than assuming that every checkpoint is Apache-2.0.
