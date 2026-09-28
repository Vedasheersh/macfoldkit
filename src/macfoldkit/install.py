"""Create isolated, version-pinned model environments using uv and Git."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .config import PACKAGE, runtime_python, supported_platform

SOURCES = {
    "fastplms": ("https://github.com/Synthyra/FastPLMs.git", "d642641b1dcbd58b21863cb76118b2996c617d1f"),
    "boltz": ("https://github.com/colbyford/boltz.git", "1a56ef48739835d8863ec4666644bf7a7d85bc14"),
    "mosaic": ("https://github.com/escalante-bio/mosaic.git", "b94b9d4eb9907a700a6d78ed2d29d3704c5df46c"),
}
RUNTIMES = {"boltz": ("boltz", "boltz-preprocess"), "colabfold": ("colabfold",), "mosaic": ("mosaic",)}


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def clone_source(home, name):
    url, revision = SOURCES[name]
    destination = home / "sources" / name
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(name + ".installing")
        if temporary.exists():
            raise RuntimeError(f"Incomplete previous checkout at {temporary}; inspect and move it before retrying.")
        run(["git", "clone", "--filter=blob:none", "--no-checkout", url, temporary])
        run(["git", "-C", temporary, "checkout", "--detach", revision])
        temporary.rename(destination)
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError(f"{destination} is at {actual}, expected {revision}; refusing to overwrite it.")
    return destination


def patch_mosaic(source):
    relative = "src/mosaic/proteinmpnn/mpnn.py"
    original = subprocess.check_output(["git", "-C", str(source), "show", f"HEAD:{relative}"], text=True)
    updated = original.replace("from joltz.backend import (", "from ._conversion import (")
    target = source / relative
    if updated == original or target.read_text() not in (original, updated):
        raise RuntimeError("Mosaic source has unexpected edits; refusing to overwrite them.")
    files = ("_conversion.py", "JOLTZ_LICENSE", "JOLTZ_NOTICE")
    for name in files:
        src, dst = PACKAGE / "runners/mosaic" / name, target.parent / name
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise RuntimeError(f"Unexpected existing file: {dst}")
    for name in files:
        shutil.copyfile(PACKAGE / "runners/mosaic" / name, target.parent / name)
    target.write_text(updated)


def setup(home, backend):
    if not supported_platform():
        raise RuntimeError("Model setup requires native arm64 Python on Apple Silicon macOS. The lightweight CLI is portable.")
    for tool in ("uv", "git"):
        if not shutil.which(tool):
            raise RuntimeError(f"Required tool is missing: {tool}. See installation instructions.")
    home.mkdir(parents=True, exist_ok=True)
    with (home / ".setup.lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        marker = home / f"{backend}.json"
        # A failed upgrade must not leave the old runtime marked ready.
        marker.unlink(missing_ok=True)
        for source_name in {"boltz": ("fastplms", "boltz"), "mosaic": ("mosaic",), "colabfold": ()}[backend]:
            clone_source(home, source_name)
        if backend == "mosaic":
            patch_mosaic(home / "sources/mosaic")
        env = dict(os.environ, UV_CACHE_DIR=str(home / "cache/uv"))
        hashes = {}
        for name in RUNTIMES[backend]:
            python = runtime_python(home, name)
            if not python.exists():
                python.parent.parent.parent.mkdir(parents=True, exist_ok=True)
                run(["uv", "venv", "--python", "3.12", python.parent.parent], env=env)
            lock = PACKAGE / "data" / f"{name}.lock"
            # The tested JAX versions intentionally override upstream ColabFold
            # bounds. All transitive dependencies are explicitly listed.
            run(["uv", "pip", "install", "--python", python, "--no-deps", "-r", lock], env=env)
            hashes[name] = hashlib.sha256(lock.read_bytes()).hexdigest()
        marker.write_text(json.dumps({"backend": backend, "requirements_sha256": hashes,
                                      "sources": {n: SOURCES[n][1] for n in
                                                  {"boltz": ("fastplms", "boltz"), "mosaic": ("mosaic",), "colabfold": ()}[backend]}}, indent=2) + "\n")
    print(f"{backend} environment installed at {home}. " +
          ("Mosaic's ProteinMPNN weights are included in its source checkout." if backend == "mosaic"
           else f"Next: macfoldkit --home '{home}' fetch {backend}"))
