# Changelog

## Unreleased — source-checkout experiments

- Full Mosaic AF2/ProteinMPNN monomer and ubiquitin binder workflows on MPS.
- Contact-loss reverse-mode workaround with analytical and full-model CPU/MPS checks.
- Frozen binder screening with target-aligned pose metrics, six recorded refolds
  (no passing candidate), and 13 evaluator tests in CI.
- BindCraft2 feasibility audit; no BindCraft2 GPU port or CLI expansion.

## 0.1.0a1

- Lightweight CLI with isolated pinned backend environments and portable caches.
- Mac Boltz2 and ColabFold adapters using previously measured optimizations.
- Experimental single-chain Mosaic/ProteinMPNN fixed-backbone design.
- Explicit Boltz MSA upload permission and single-sequence ColabFold default.
- Compatible protein CIF metadata for folding-to-design handoff.
- CPU tests, reproducible source patch, upstream attribution and scope documentation.
