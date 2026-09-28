# Tested scope and limitations

- M5 Pro, 24 GB unified memory and macOS 26.6.2 are the tested hardware/software
  combination. M1–M4, other M5 variants and older macOS versions need testing.
- PyTorch MPS compilation and community JAX-MPS can have unsupported operators
  or numerical differences. The setup snapshots intentionally override some
  upstream dependency bounds; backend environments must stay separate.
- Only protein inputs are supported by the Boltz adapter. Small molecules,
  nucleic acids, modifications, affinity prediction, templates and restraints
  are not implemented here.
- MSA caps, precision and recycle/sampling settings affect output quality. Model
  confidence is not experimental accuracy. Large complexes may exceed memory.
- Mosaic design is a deterministic single-chain fixed-backbone prototype.
  Full binder-design workflows and folding-model gradients are not exposed.
- Custom Metal attention kernels and rebuilt NAX JAX backends are not included.
  Their inference-only kernels cannot simply be reused for design gradients.
- `doctor` reports platform and installation state, not proof that a GPU model
  will execute successfully. ColabFold additionally audits device placement of
  actual neural outputs during prediction.
- No automatic PyPI publication, remote telemetry, model hosting or cloud GPU
  service is provided. Explicit model/asset fetching contacts upstream providers;
  MSA searches send sequences to the chosen public service.
