"""Validate protein-only Boltz YAML and cache official preprocessing outside Git."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import re
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HOME = Path.home() / ("Library/Caches/macfoldkit" if sys.platform == "darwin" else ".cache/macfoldkit")
HOME = Path(os.environ.get("MACFOLDKIT_HOME", DEFAULT_HOME)).expanduser().resolve()
CACHE_VERSION = 2


def input_description(input_path: Path) -> dict:
    input_path = input_path.expanduser().resolve(strict=True)
    schema = yaml.safe_load(input_path.read_text())
    if not isinstance(schema, dict) or schema.get("version", 1) != 1:
        raise ValueError("Expected a Boltz version: 1 YAML mapping.")
    unsupported = set(schema) - {"version", "sequences"}
    if unsupported:
        raise ValueError(f"This protein-only FastPLMs runner does not support: {sorted(unsupported)} (including templates, constraints and affinity properties).")
    entries = schema.get("sequences")
    if not isinstance(entries, list) or not entries:
        raise ValueError("sequences must be a nonempty list of protein entries.")
    chains, local_msas, normalized, entity_msas = {}, {}, [], {}
    needs_server = False
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"protein"}:
            raise ValueError("Only protein entries are supported; ligands, DNA and RNA are unsupported.")
        protein = entry["protein"]
        if not isinstance(protein, dict) or set(protein) - {"id", "sequence", "msa"}:
            raise ValueError("Protein entries support only id, sequence and msa; modifications and cyclic proteins are unsupported.")
        ids = protein.get("id")
        ids = [ids] if isinstance(ids, str) else ids
        if not isinstance(ids, list) or not ids:
            raise ValueError("Each protein needs an id string or nonempty list of id strings.")
        sequence = protein.get("sequence")
        if not isinstance(sequence, str) or not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError("Sequences must contain only uppercase standard amino-acid letters (20 canonical residues), without gaps or whitespace.")
        for chain_id in ids:
            if not isinstance(chain_id, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,4}", chain_id) is None:
                raise ValueError("Chain IDs must be 1–5 alphanumeric characters beginning with a letter.")
            if chain_id in chains:
                raise ValueError(f"Duplicate chain ID: {chain_id}")
            chains[chain_id] = sequence
        item = {"id": ids, "sequence": sequence}
        msa = protein.get("msa")
        if msa is None or msa == "":
            needs_server = True
        elif msa == "empty":
            item["msa"] = "empty"
        elif isinstance(msa, str):
            msa_path = (input_path.parent / Path(msa).expanduser()).resolve(strict=True)
            if not msa_path.is_file():
                raise ValueError(f"MSA path is not a file: {msa_path}")
            item["msa"] = str(msa_path)
            local_msas[str(msa_path)] = hashlib.sha256(msa_path.read_bytes()).hexdigest()
        else:
            raise ValueError("msa must be a local file path, 'empty', or omitted for the public MSA service.")
        msa_setting = item.get("msa", "auto")
        if sequence in entity_msas and entity_msas[sequence] != msa_setting:
            raise ValueError("Copies of an identical protein must use the same MSA setting.")
        entity_msas[sequence] = msa_setting
        normalized.append({"protein": item})
    if local_msas and needs_server:
        raise ValueError("Official Boltz preprocessing cannot mix local and automatic MSAs; supply an MSA or msa: empty for every protein.")
    schema = {"version": 1, "sequences": normalized}
    cache_identity = {"schema": schema, "local_msa_sha256": local_msas, "cache_version": CACHE_VERSION}
    digest = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
    return {
        "input": str(input_path), "input_sha256": digest,
        "schema": schema, "chains": chains, "needs_msa_server": needs_server,
        "local_msa_sha256": local_msas, "cache_version": CACHE_VERSION,
        "job_dir": str(HOME / "jobs" / "boltz" / digest),
    }


def prepare(description: dict, refresh: bool = False, offline: bool = False, allow_msa_server: bool = False) -> dict:
    import fcntl

    job_dir = Path(description["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    feature_path = job_dir / "features.pt"
    summary_path = job_dir / "features.json"
    with (job_dir / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if feature_path.exists() and summary_path.exists() and not refresh:
            report = json.loads(summary_path.read_text())
            print(f"Using cached official features: {feature_path}", flush=True)
            return report
        if offline and description["needs_msa_server"]:
            raise ValueError("This input needs an MSA search but --offline was set. Provide local MSAs, msa: empty, or previously cached features.")
        if description["needs_msa_server"] and not allow_msa_server:
            raise ValueError("MSA search uploads sequences to https://api.colabfold.com. Pass --allow-msa-server to enable it, or provide local MSAs / msa: empty.")
        import torch
        import random
        import numpy as np
        from rdkit import Chem
        from boltz.data.module.inferencev2 import PredictionDataset
        from boltz.data.types import Manifest
        from boltz.main import process_inputs
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        # A fresh processing directory makes --refresh bypass Boltz's own cache.
        import tempfile
        processed_dir = Path(tempfile.mkdtemp(prefix="preprocess-", dir=job_dir))
        normalized_path = processed_dir / "input.yaml"
        normalized_path.write_text(yaml.safe_dump(description["schema"], sort_keys=False))
        cache_dir = HOME / "cache" / "boltz"
        if not (cache_dir / "mols").exists():
            raise FileNotFoundError(f"Official Boltz molecule cache is missing: {cache_dir}")
        if description["needs_msa_server"]:
            print("Submitting the input sequences to https://api.colabfold.com for MSA search.", flush=True)
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        process_inputs(
            data=[normalized_path], out_dir=processed_dir,
            ccd_path=cache_dir / "ccd.pkl", mol_dir=cache_dir / "mols",
            msa_server_url="https://api.colabfold.com", msa_pairing_strategy="greedy",
            use_msa_server=description["needs_msa_server"], boltz2=True,
            preprocessing_threads=1,
        )
        manifest = Manifest.load(processed_dir / "processed" / "manifest.json")
        if len(manifest.records) != 1:
            raise RuntimeError("Official preprocessing did not produce exactly one target.")
        dataset = PredictionDataset(
            manifest=manifest, target_dir=processed_dir / "processed" / "structures",
            msa_dir=processed_dir / "processed" / "msa", mol_dir=cache_dir / "mols",
            constraints_dir=processed_dir / "processed" / "constraints",
            template_dir=processed_dir / "processed" / "templates",
            extra_mols_dir=processed_dir / "processed" / "mols",
        )
        unbatched = dataset[0]
        features = {key: value.unsqueeze(0) for key, value in unbatched.items() if torch.is_tensor(value)}
        if "token_index" not in features:
            raise RuntimeError("Official preprocessing returned no protein features.")
        ordered_chains = []
        # Boltz groups identical entities, which may change YAML chain order.
        for chain in sorted(manifest.records[0].chains, key=lambda c: c.chain_id):
            sequence = description["chains"][chain.chain_name]
            if chain.num_residues != len(sequence):
                raise RuntimeError(f"Unexpected residue count for chain {chain.chain_name}.")
            ordered_chains.append({"id": chain.chain_name, "asym_id": chain.chain_id, "sequence": sequence})
        expected_tokens = sum(len(c["sequence"]) for c in ordered_chains)
        if int(features["token_pad_mask"].sum()) != expected_tokens:
            raise RuntimeError("Protein token count differs from input residue count.")
        report = {**description, "chains": ordered_chains, "feature_path": str(feature_path),
                  "preprocessing_seed": 42,
                  "processed_dir": str(processed_dir), "total_residues": expected_tokens,
                  "msa_rows": int(features["msa"].shape[1])}
        torch.save(features, job_dir / "features.tmp.pt")
        (job_dir / "features.tmp.pt").replace(feature_path)
        (job_dir / "features.tmp.json").write_text(json.dumps(report, indent=2) + "\n")
        (job_dir / "features.tmp.json").replace(summary_path)
        print(f"Prepared {expected_tokens} residues; {report['msa_rows']} MSA rows: {feature_path}", flush=True)
        return report


def snapshot_features(job_dir: Path) -> dict:
    """Pin metadata and feature bytes together while excluding cache refreshes."""
    import fcntl
    import tempfile

    snapshot_dir = job_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    with (job_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # prepare() publishes both files under this same lock. Reading metadata
        # and opening the source here therefore identifies one preprocessing run.
        report = json.loads((job_dir / "features.json").read_text())
        digest = hashlib.sha256()
        try:
            # Keep one source descriptor open throughout the copy/hash. Even an
            # external atomic replacement cannot change the bytes we snapshot.
            with (job_dir / "features.pt").open("rb") as source, tempfile.NamedTemporaryFile(
                dir=snapshot_dir, prefix=".features-", suffix=".tmp", delete=False
            ) as target:
                temporary = Path(target.name)
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
            checksum = digest.hexdigest()
            snapshot = snapshot_dir / f"{checksum}.pt"
            if snapshot.exists():
                temporary.unlink()
            else:
                temporary.replace(snapshot)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {**report, "feature_path": str(snapshot), "feature_sha256": checksum}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-msa-server", action="store_true", help="Allow sequence submission to the public ColabFold MSA service.")
    parser.add_argument("--validate-only", action="store_true", help="Validate YAML and local MSA paths without preprocessing or network access.")
    args = parser.parse_args()
    try:
        description = input_description(args.input)
        report = description if args.validate_only else prepare(description, args.refresh, args.offline, args.allow_msa_server)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
