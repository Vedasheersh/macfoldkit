"""Run ColabFold on Apple GPU and verify actual model-output device placement."""
from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def worker(optimized_batching: bool = False) -> None:
    """Instrument the neural model before ColabFold copies outputs to NumPy."""
    import jax
    from alphafold.model.model import RunModel

    if jax.default_backend() != "mps":
        raise RuntimeError(f"Expected MPS as default backend, got {jax.default_backend()!r}")
    if optimized_batching:
        from mac_batching import install
        install()
    audit_path = Path(os.environ["COLABFOLD_DEVICE_AUDIT_PATH"])
    original_init = RunModel.__init__
    model_counter = 0

    def emit(record: dict) -> None:
        print("MODEL_DEVICE_AUDIT " + json.dumps(record), flush=True)
        with audit_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    @functools.wraps(original_init)
    def init_with_audit(self, *args, **kwargs):
        nonlocal model_counter
        original_init(self, *args, **kwargs)
        original_apply = self.apply
        model_index = model_counter
        model_counter += 1
        call_index = 0

        @functools.wraps(original_apply)
        def audited_apply(*apply_args, **apply_kwargs):
            nonlocal call_index
            record = {"model_instance": model_index, "apply_call": call_index}
            try:
                result = original_apply(*apply_args, **apply_kwargs)
                leaves = [leaf for leaf in jax.tree_util.tree_leaves(result) if isinstance(leaf, jax.Array)]
                if not leaves:
                    raise RuntimeError("RunModel.apply returned no JAX arrays to verify")
                devices = {device for leaf in leaves for device in leaf.devices()}
                unexpected = [str(device) for device in devices if device.platform != "mps"]
                if unexpected:
                    raise RuntimeError(f"Neural model outputs are on unexpected devices: {unexpected}")
                coords = result.get("structure_module", {}).get("final_atom_positions")
                if not isinstance(coords, jax.Array):
                    raise RuntimeError("Neural model output lacks JAX final_atom_positions")
                coords.block_until_ready()
                record.update({"ok": True, "jax_array_leaves": len(leaves),
                               "devices": sorted(str(d) for d in devices),
                               "platforms": sorted({d.platform for d in devices}),
                               "coordinate_shape": list(coords.shape), "coordinate_dtype": str(coords.dtype),
                               "coordinate_devices": sorted(str(d) for d in coords.devices())})
            except Exception as exc:
                emit({**record, "ok": False, "error": str(exc)})
                raise
            emit(record)
            call_index += 1
            return result

        self.apply = audited_apply
        for attr in ("lower", "trace", "eval_shape", "clear_cache"):
            if hasattr(original_apply, attr):
                setattr(self.apply, attr, getattr(original_apply, attr))

    RunModel.__init__ = init_with_audit
    from colabfold.batch import main
    main()


def main() -> None:
    default_home = Path.home() / ("Library/Caches/macfoldkit" if sys.platform == "darwin" else ".cache/macfoldkit")
    home = Path(os.environ.get("MACFOLDKIT_HOME", str(default_home))).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("input", type=Path, help="Local FASTA, A3M, CSV, or supported input directory.")
    parser.add_argument("output", type=Path, help="Directory for predictions and device audit.")
    parser.add_argument("--msa-mode", default="single_sequence", choices=["single_sequence", "mmseqs2_uniref_env", "mmseqs2_uniref"])
    parser.add_argument("--num-recycle", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--max-msa", default=None)
    parser.add_argument("--async-dispatch", action="store_true", help="Enable the jax-mps asynchronous-dispatch experiment; off by default.")
    parser.add_argument("--no-optimized-batching", action="store_true", help="Use the original four-row batching policy instead of the validated Mac policy.")
    parser.add_argument("--data", type=Path, default=home / "weights/colabfold")
    args, extra = parser.parse_known_args()
    if any(option == "--zip" or option.startswith("--zip=") for option in extra):
        parser.error("--zip is unsupported: GPU result verification requires the exported PDB files.")
    if not args.input.expanduser().exists():
        parser.error(f"Input does not exist: {args.input}")
    args.output = args.output.expanduser().resolve()
    previous_pdbs = {p: p.stat().st_mtime_ns for p in args.output.glob("*_unrelaxed_rank_*.pdb")}
    if previous_pdbs and "--overwrite-existing-results" not in extra:
        parser.error("Output already contains predictions; choose another directory or pass --overwrite-existing-results")
    args.output.mkdir(parents=True, exist_ok=True)
    audit_path = args.output / "model-device-audit.jsonl"
    audit_path.write_text("")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               str(args.input.expanduser().resolve()), str(args.output),
               "--data", str(args.data.expanduser().resolve()), "--msa-mode", args.msa_mode,
               "--model-type", "alphafold2_ptm", "--num-models", "1", "--model-order", "1",
               "--num-recycle", str(args.num_recycle), "--num-seeds", "1", "--random-seed", str(args.random_seed),
               "--max-msa", args.max_msa or ("1:1" if args.msa_mode == "single_sequence" else "128:256"),
               "--num-relax", "0", "--no-use-fast-kernels", "--recompile-padding", "0", *extra]
    env = dict(os.environ, JAX_PLATFORMS="mps,cpu", PYTHONUNBUFFERED="1",
               COLABFOLD_DEVICE_AUDIT_PATH=str(audit_path),
               COLABFOLD_OPTIMIZED_BATCHING="0" if args.no_optimized_batching else "1",
               JAX_MPS_ASYNC_DISPATCH="1" if args.async_dispatch else "0")
    env.setdefault("MPLCONFIGDIR", str(home / "cache/matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(home / "cache"))
    started = time.perf_counter()
    failed_queries = []
    with (args.output / "process.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if any(message in line for message in ("Could not predict ",
                    "Could not get MSA/templates for", "Could not generate input features")):
                failed_queries.append(line.strip())
        code = proc.wait()
    peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    audit = [json.loads(line) for line in audit_path.read_text().splitlines() if line]
    fresh_pdbs = [str(p) for p in args.output.glob("*_unrelaxed_rank_*.pdb")
                  if p not in previous_pdbs or p.stat().st_mtime_ns != previous_pdbs[p]]
    verified = bool(audit) and all(record.get("ok") and record.get("platforms") == ["mps"] for record in audit)
    metrics = {"command": command, "backend": "mps", "cpu_feature_preprocessing": True,
               "async_dispatch": args.async_dispatch,
               "optimized_batching": not args.no_optimized_batching,
               "process_seconds": time.perf_counter() - started,
               "maximum_resident_set_bytes": peak_rss if sys.platform == "darwin" else peak_rss * 1024,
               "returncode": code, "model_output_devices_verified": verified,
               "failed_queries": failed_queries,
               "model_apply_calls": len(audit), "fresh_predictions": fresh_pdbs,
               "versions": {name: importlib.metadata.version(name) for name in
                            ["colabfold", "alphafold-colabfold", "jax", "jaxlib", "jax-mps", "dm-haiku", "biopython"]}}
    (args.output / "benchmark.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if code:
        raise SystemExit(code)
    if failed_queries:
        raise SystemExit("ColabFold reported failed queries; inspect benchmark.json and process.log")
    if not fresh_pdbs or not verified:
        raise SystemExit("ColabFold did not produce a fresh PDB with verified MPS model outputs; inspect process.log and model-device-audit.jsonl")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        del sys.argv[1]
        worker(optimized_batching=os.environ.get("COLABFOLD_OPTIMIZED_BATCHING") == "1")
    else:
        main()
