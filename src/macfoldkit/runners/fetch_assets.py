"""Fetch assets inside the appropriate backend runtime, retaining upstream terms."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
from urllib.parse import urlsplit

MOLS_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/6fdef46d763fee7fbb83ca5501ccceff43b85607/mols.tar"
MOLS_SHA256 = "39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7"
CCD_URL = "https://huggingface.co/biohub/ESMFold2/resolve/1ebf0e3481a5184eb6171d40615c79e384b48796/ccd.pkl"
CCD_SHA256 = "9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_verified(url, destination, expected):
    if destination.is_file() and sha256(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, prefix=destination.name + ".")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            endpoint = os.environ.get("HF_ENDPOINT")
            if endpoint and urlsplit(url).hostname == "huggingface.co":
                if urlsplit(endpoint).scheme != "https":
                    raise ValueError("HF_ENDPOINT must use HTTPS for authenticated asset downloads")
                from huggingface_hub import hf_hub_download
                owner, repo, resolve, revision, filename = urlsplit(url).path.strip("/").split("/", 4)
                if resolve != "resolve":
                    raise ValueError(f"Unexpected Hugging Face asset URL: {url}")
                cached = hf_hub_download(repo_id=f"{owner}/{repo}", revision=revision,
                                         filename=filename, endpoint=endpoint)
                with open(cached, "rb") as response:
                    shutil.copyfileobj(response, output)
            else:
                with urllib.request.urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, output)
        if sha256(temporary) != expected:
            raise RuntimeError(f"Asset checksum mismatch: {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_boltz(home):
    from huggingface_hub import snapshot_download
    cache = home / "cache/boltz"
    cache.mkdir(parents=True, exist_ok=True)
    print("Downloading pinned Boltz model and chemistry assets (several GB).", flush=True)
    download_verified(CCD_URL, cache / "fastplms-ccd.pkl", CCD_SHA256)
    marker = cache / "mols.complete.json"
    if marker.exists():
        try:
            recorded = json.loads(marker.read_text())
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"Invalid molecule cache marker: {marker}") from exc
        if recorded != {"source": MOLS_URL, "sha256": MOLS_SHA256}:
            raise RuntimeError(f"Molecule cache version mismatch: {marker}; use a separate --home for this release.")
    if not (cache / "mols").is_dir() or not marker.exists():
        if (cache / "mols").exists():
            raise RuntimeError(f"Unverified existing {cache / 'mols'}; move it aside before fetching.")
        archive = cache / "mols.tar"
        download_verified(MOLS_URL, archive, MOLS_SHA256)
        with tempfile.TemporaryDirectory(dir=cache, prefix="extract-") as temporary:
            with tarfile.open(archive) as stream:
                stream.extractall(temporary, filter="data")
            extracted = Path(temporary) / "mols"
            if not extracted.is_dir() or not any(extracted.iterdir()):
                raise RuntimeError("Molecule archive lacks the expected mols directory")
            extracted.rename(cache / "mols")
        marker.write_text(json.dumps({"source": MOLS_URL, "sha256": MOLS_SHA256}) + "\n")
    snapshot_download(repo_id="Synthyra/Boltz2", revision="bce98f7ce914d468182726b5a0fbd23737167875")
    print("Boltz assets ready; --offline folding can now use local or empty MSAs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("boltz", "colabfold"))
    parser.add_argument("--model-type", default="alphafold2_ptm", choices=("alphafold2_ptm", "alphafold2_multimer_v3"))
    args = parser.parse_args()
    home = Path(os.environ["MACFOLDKIT_HOME"])
    with (home / ".fetch.lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if args.backend == "boltz":
            fetch_boltz(home)
        else:
            from colabfold.download import download_alphafold_params
            target = home / "weights/colabfold"
            target.mkdir(parents=True, exist_ok=True)
            download_alphafold_params(args.model_type, target)


if __name__ == "__main__":
    main()
