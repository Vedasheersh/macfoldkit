# Version-pinned upstream patch

`mosaic-standalone-proteinmpnn.patch` applies to Mosaic commit
`b94b9d4eb9907a700a6d78ed2d29d3704c5df46c`.

It removes ProteinMPNN's dependency on Joltz's eager Boltz imports by copying the
unchanged conversion utility into the ProteinMPNN package, retaining its notices,
and changing one import. `macfoldkit setup mosaic` applies the equivalent checked
transformation automatically. Do not apply this patch a second time to that
prepared checkout.

For an unmodified checkout at the exact revision:

```bash
git apply --check /path/to/mosaic-standalone-proteinmpnn.patch
git apply /path/to/mosaic-standalone-proteinmpnn.patch
```

This patch has been validated locally but not submitted upstream. A future
upstream refactor may prefer a small shared conversion package over vendoring.
The remaining Mac changes currently live in our integration adapters; custom
Metal/backend experiments are not shipped by this release.
