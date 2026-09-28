# Contributing

Use Python 3.12. The CLI has no model dependencies. Lightweight tests require
PyYAML and Gemmi; build tests additionally use `build` and `twine`.

```bash
python -m pip install -e . pyyaml gemmi build twine
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

Keep model dependencies in separate backend snapshots. A dependency change must
include a fresh installation check and actual inference/gradient validation on
the hardware claimed. Do not promote a faster microbenchmark to an end-to-end
speedup without measuring it. Report quality, precision, sequence length, MSA
settings, compilation and memory definitions alongside timing.

Prefer upstream fixes for reusable model/backend changes. Until merged, keep
patches small and pinned to exact revisions. Retain licenses and modified-file
records. Never commit credentials, user sequences, local absolute paths, model
weights, virtual environments or large caches.

CPU CI checks packaging and routing; it does not establish GPU correctness.
GPU regressions require a local Apple Silicon run. Use a new output directory
and retain model/input revisions in the result report.
