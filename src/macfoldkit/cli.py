"""Unified CLI; heavy frameworks run only in isolated subprocesses."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

from . import __version__
from .config import BACKENDS, PACKAGE, cache_home, runtime_env, runtime_python, supported_platform
from .install import setup


def require_runtime(home, backend):
    python = runtime_python(home, backend)
    if not python.is_file() or not (home / f"{backend}.json").is_file():
        raise RuntimeError(f"Backend {backend!r} is not set up. Run: macfoldkit --home '{home}' setup {backend}")
    return python


def doctor(home):
    result = {"version": __version__, "home": str(home), "system": platform.system(),
              "machine": platform.machine(), "supported_platform": supported_platform(),
              "tested_hardware": "M5 Pro, 24 GB, macOS 26.6.2; other Apple Silicon configurations unvalidated",
              "tools": {t: shutil.which(t) for t in ("uv", "git", "xcrun")},
              "backends": {b: {"python": str(runtime_python(home, b)),
                                "installed": runtime_python(home, b).is_file() and (home / f"{b}.json").is_file()}
                           for b in BACKENDS}}
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Protein folding and design, optimized for Apple Silicon")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--home", help="Runtime/cache root (or MACFOLDKIT_HOME)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Show platform, tools and installed environments as JSON")
    install = sub.add_parser("setup", help="Create an isolated model runtime; downloads code and packages")
    install.add_argument("backend", choices=BACKENDS)
    fetch = sub.add_parser("fetch", help="Download model weights and required chemistry assets")
    fetch.add_argument("backend", choices=("boltz", "colabfold"))
    fetch.add_argument("--model-type", choices=("alphafold2_ptm", "alphafold2_multimer_v3"), default="alphafold2_ptm")
    fold = sub.add_parser("fold", help="Predict proteins with the tested Mac settings")
    fold.add_argument("input")
    fold.add_argument("output")
    fold.add_argument("--backend", choices=("boltz", "colabfold"), default="boltz")
    design = sub.add_parser("design", help="EXPERIMENTAL: redesign every position of one fixed protein backbone")
    design.add_argument("structure")
    design.add_argument("output")
    design.add_argument("--steps", type=int, default=25)
    design.add_argument("--seed", type=int, default=7)
    design.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args, extra = parser.parse_known_args(argv)
    if args.command != "fold" and extra:
        parser.error("unrecognized arguments: " + " ".join(extra))
    home = cache_home(args.home)
    try:
        if args.command == "doctor":
            return doctor(home)
        if args.command == "setup":
            setup(home, args.backend)
            return 0
        if args.command == "fetch":
            python = require_runtime(home, args.backend)
            command = [str(python), str(PACKAGE / "runners/fetch_assets.py"), args.backend,
                       "--model-type", args.model_type]
            return subprocess.run(command, env=runtime_env(home, args.backend)).returncode
        if args.command == "fold":
            python = require_runtime(home, args.backend)
            extra = extra[1:] if extra[:1] == ["--"] else extra
            script = {"boltz": "boltz/predict_mac.py", "colabfold": "colabfold/run_mps.py"}[args.backend]
            env = runtime_env(home, args.backend)
            if "--offline" in extra:
                env["HF_HUB_OFFLINE"] = "1"
            command = [str(python), str(PACKAGE / "runners" / script), args.input, args.output, *extra]
        else:
            python = require_runtime(home, "mosaic")
            if args.steps < 1:
                parser.error("--steps must be positive")
            env = runtime_env(home, "mosaic")
            env["JAX_PLATFORMS"] = args.device
            command = [str(python), str(PACKAGE / "runners/mosaic/design.py"), "--structure", args.structure,
                       "--output", args.output, "--platform", args.device, "--steps", str(args.steps), "--seed", str(args.seed)]
        return subprocess.run(command, env=env).returncode
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"macfoldkit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
