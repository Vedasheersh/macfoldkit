# Experimental Mosaic design

The first design adapter supports only fixed-backbone redesign of one complete
protein chain. It uses Mosaic at `b94b9d4eb9907a700a6d78ed2d29d3704c5df46c`, its
bundled ProteinMPNN `v_48_020.pt`, and ordinary JAX autodiff. It pre-encodes the
backbone and runs projected sequence optimization with a fixed decoding order.

MacFoldKit checks raw finite/nonzero gradients and a directional finite difference
before calling Mosaic's optimizer. It also checks gradients and simplex validity
at every step, and reevaluates the final sequence. This avoids hiding backend
failures behind upstream optimizer `nan_to_num` behavior. Reports include startup,
warm gradient timings, loss history and peak process RSS; RSS is not total GPU
or whole-machine memory consumption.

The source patch changes one import and vendors Joltz's exact conversion utility
under ProteinMPNN, keeping its private registration table independent of the full
Boltz import stack. It changes neither model mathematics nor checkpoint values.

The broader Mosaic framework supports composite objectives and gradients through
folding models. Those are not exposed by this release's `design` command.
Small AF2 distogram gradient probes worked in the research prototype, but full
folding-guided optimization, large binder complexes, fixed residues and Boltz2
backpropagation remain future integration and validation work.

Do not infer design quality from an optimizer's decreasing loss. At minimum,
independently refold candidate sequences using an appropriate predictor and
evaluate agreement with the supplied backbone. Full experimental validation
depends on the biological objective.
