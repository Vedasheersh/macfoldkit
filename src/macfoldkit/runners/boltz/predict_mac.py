"""Run a protein-only Boltz2 monomer or complex with MacFoldKit on Apple Silicon."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from prepare_mac_features import SCRIPT_DIR, HOME, input_description, snapshot_features

PROFILES = {
    "balanced": {"recycling_steps": 1, "sampling_steps": 100, "precision": "bfloat16"},
    "fast": {"recycling_steps": 0, "sampling_steps": 50, "precision": "bfloat16"},
    "quality": {"recycling_steps": 3, "sampling_steps": 200, "precision": "float32"},
}

CIF_ATOM_TAGS = (
    "group_PDB", "id", "type_symbol", "label_atom_id", "label_comp_id",
    "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
    "occupancy", "B_iso_or_equiv", "pdbx_PDB_model_num",
)


def repair_and_check_cif(path: Path, chains: list[dict], atom_count: int) -> None:
    source_lines = path.read_text().splitlines()
    if [line for line in source_lines if line.startswith("_atom_site.")] != [
        f"_atom_site.{tag}" for tag in CIF_ATOM_TAGS
    ]:
        raise RuntimeError("Unexpected CIF atom table layout or chain ID.")
    offsets, expected_residues, offset = {}, {}, 0
    entities, chain_entities = {}, {}
    for chain in chains:
        offsets[chain["id"]] = offset
        expected_residues[chain["id"]] = set(range(1, len(chain["sequence"]) + 1))
        offset += len(chain["sequence"])
        # Identical protein chains are copies of the same molecular entity.
        entities.setdefault(chain["sequence"], len(entities) + 1)
        chain_entities[chain["id"]] = entities[chain["sequence"]]
    seen, ca_seen, atom_keys, lines = {}, {}, set(), []
    for line in source_lines:
        if line == "_atom_site.pdbx_PDB_model_num":
            lines.extend([line, "_atom_site.label_alt_id", "_atom_site.label_entity_id"])
            continue
        fields = line.split()
        if fields and fields[0] == "ATOM":
            if len(fields) != 13 or fields[5] not in offsets:
                raise RuntimeError("Unexpected CIF atom table layout or chain ID.")
            chain_id = fields[5]
            residue_id = int(fields[6]) - offsets[chain_id]
            if residue_id not in expected_residues[chain_id]:
                raise RuntimeError("CIF residue numbering does not match input chains.")
            if not all(math.isfinite(float(v)) for v in fields[7:12]):
                raise RuntimeError("CIF contains non-finite coordinates or confidence values.")
            key = (chain_id, residue_id, fields[3])
            if key in atom_keys:
                raise RuntimeError(f"Duplicate atom in CIF: {key}")
            atom_keys.add(key)
            seen.setdefault(chain_id, set()).add(residue_id)
            if fields[3] == "CA":
                ca_seen.setdefault(chain_id, set()).add(residue_id)
            fields[6] = str(residue_id)
            # Preserve the original 13 columns; add only missing metadata.
            line = " ".join(fields + [".", str(chain_entities[chain_id])])
        lines.append(line)
    if len(atom_keys) != atom_count or seen != expected_residues or ca_seen != expected_residues:
        raise RuntimeError("CIF atom, chain or C-alpha residue count differs from input.")
    # Gemmi requires label_alt_id to read atoms and entity_type to remove
    # ligands/waters. These are protein-only predictions with no alternates.
    lines.extend(["loop_", "_entity.id", "_entity.type"])
    lines.extend(f"{entity} polymer" for entity in entities.values())
    lines.extend(["#", "loop_", "_entity_poly.entity_id", "_entity_poly.type"])
    lines.extend(f"{entity} 'polypeptide(L)'" for entity in entities.values())
    lines.extend(["#", "loop_", "_struct_asym.id", "_struct_asym.entity_id"])
    lines.extend(f"{chain_id} {entity}" for chain_id, entity in chain_entities.items())
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")


def predict(description: dict, args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModel
    from mac_optimizations import optimize_model
    from model_info import MODEL_ID, REVISION, first_float, to_jsonable

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable; this runner requires an Apple Silicon Mac.")
    settings = dict(PROFILES[args.profile])
    settings["precision"] = args.precision or settings["precision"]
    input_name = re.sub(r"[^A-Za-z0-9_-]", "_", Path(description["input"]).stem)
    run_name = f"{args.profile}-seed{args.seed}-msa{args.msa_rows}-{settings['precision']}"
    result_dir = args.result_dir.expanduser().resolve() / f"{input_name}-{description['input_sha256'][:12]}" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / ".prediction.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (result_dir / "prediction.cif").exists() or (result_dir / "prediction.json").exists():
            raise FileExistsError(f"A prediction already exists at {result_dir}; choose another result directory or seed.")
        started = time.perf_counter()
        model = AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True, dtype=torch.float32,
            local_files_only=args.offline,
        ).eval().to("mps")
        optimization = optimize_model(model, precision=settings["precision"], chunk_size=32)
        loaded = time.perf_counter()
        helpers = model.predict_structure.__func__.__globals__
        sequence = "".join(c["sequence"] for c in description["chains"])
        minimal, template = helpers["build_boltz2_features"](sequence)
        prepared = torch.load(description["feature_path"], map_location="cpu", weights_only=True)
        missing = set(minimal) - set(prepared)
        if missing:
            raise RuntimeError(f"Official features lack required FastPLMs keys: {sorted(missing)}")
        features = {key: prepared[key] for key in minimal}
        expected_asym, residue_chains = [], []
        for chain in description["chains"]:
            expected_asym.extend([chain["asym_id"]] * len(chain["sequence"]))
            residue_chains.extend([chain["id"]] * len(chain["sequence"]))
        token_mask = features["token_pad_mask"][0].bool()
        if features["asym_id"][0, token_mask].tolist() != expected_asym:
            raise RuntimeError("Prepared chain order differs from the official manifest.")
        atom_mask = features["atom_pad_mask"][0].bool()
        atom_count = int(atom_mask.sum())
        if atom_count != template.num_atoms or not atom_mask[:atom_count].all() or atom_mask[atom_count:].any():
            raise RuntimeError("Prepared atom layout does not match the protein export template.")
        atom_tokens = features["atom_to_token"][0, :atom_count].argmax(dim=-1)
        if atom_tokens.tolist() != template.atom_residue_index:
            raise RuntimeError("Official and FastPLMs residue-to-atom mappings differ.")
        official_names = features["ref_atom_name_chars"][0, :atom_count].argmax(dim=-1)
        minimal_names = minimal["ref_atom_name_chars"][0, :atom_count].argmax(dim=-1)
        if not torch.equal(official_names, minimal_names):
            raise RuntimeError("Official and FastPLMs atom names/order differ; refusing unsafe coordinate export.")
        template.atom_chain_id = [residue_chains[i] for i in template.atom_residue_index]
        msa_input = int(features["msa"].shape[1])
        msa_used = min(msa_input, args.msa_rows)
        for key in ("msa", "msa_paired", "deletion_value", "has_deletion", "msa_mask"):
            features[key] = features[key][:, :msa_used]
        features = model._to_model_device(features, float_dtype=torch.float32)
        torch.mps.synchronize()
        inference_started = time.perf_counter()
        with helpers["_seed_context"](args.seed), torch.no_grad():
            raw = model(
                feats=features, recycling_steps=settings["recycling_steps"],
                num_sampling_steps=settings["sampling_steps"], diffusion_samples=1,
                run_confidence_sequentially=True, return_dict=True, verbose=True,
            )
        torch.mps.synchronize()
        inference_finished = time.perf_counter()
        raw = {key: helpers["_to_cpu_detached"](value) for key, value in raw.items()}
        coords = raw["sample_atom_coords"]
        if not torch.isfinite(coords).all():
            raise RuntimeError("Prediction contains non-finite coordinates.")
        plddt, complex_plddt, ptm, iptm = (raw.get(k) for k in ("plddt", "complex_plddt", "ptm", "iptm"))
        confidence = None
        if all(value is not None for value in (complex_plddt, ptm, iptm)):
            confidence = (4 * complex_plddt + (ptm if torch.allclose(iptm, torch.zeros_like(iptm)) else iptm)) / 5
        for value in (plddt, complex_plddt, ptm, iptm, confidence):
            if value is not None and not torch.isfinite(value).all():
                raise RuntimeError("Prediction contains non-finite confidence values.")
        prediction = helpers["Boltz2StructureOutput"](
            sample_atom_coords=coords, atom_pad_mask=atom_mask, plddt=plddt,
            confidence_score=confidence, complex_plddt=complex_plddt, iptm=iptm, ptm=ptm,
            sequence=sequence, structure_template=template, raw_output=raw, seed=args.seed,
        )
        temporary_cif = result_dir / "prediction.tmp.cif"
        model.save_as_cif(prediction, str(temporary_cif))
        repair_and_check_cif(temporary_cif, description["chains"], atom_count)
        report = {
            "model": MODEL_ID, "revision": REVISION, "device": "mps", "weight_dtype": "float32",
            "input": description["input"], "input_sha256": description["input_sha256"],
            "chains": description["chains"], "total_residues": len(sequence), "atoms": atom_count,
            "profile": args.profile, **settings, "seed": args.seed, "diffusion_samples": 1,
            "msa_rows_input": msa_input, "msa_rows_used": msa_used,
            "msa_selection": "ordered prefix of official features; paired and best-matching rows first",
            "feature_path": description["feature_path"], "optimization": optimization,
            "feature_sha256": description["feature_sha256"],
            "processed_dir": description["processed_dir"],
            "feature_cache_version": description["cache_version"],
            "preprocessing_seed": description["preprocessing_seed"],
            "plddt_mean": None if plddt is None else float(plddt.mean()),
            "complex_plddt": first_float(complex_plddt), "ptm": first_float(ptm), "iptm": first_float(iptm),
            "confidence_score": first_float(confidence),
            "pair_chains_iptm": to_jsonable(raw.get("pair_chains_iptm")),
            "load_seconds": loaded - started,
            "inference_seconds": inference_finished - inference_started,
            "total_model_seconds": inference_finished - started,
            "torch_version": torch.__version__,
        }
        temporary_json = result_dir / "prediction.tmp.json"
        temporary_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary_cif.replace(result_dir / "prediction.cif")
        temporary_json.replace(result_dir / "prediction.json")
        print(json.dumps(report, indent=2))
        print(f"Saved validated CIF and confidence report: {result_dir}")
    return result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Protein-only Boltz YAML.")
    parser.add_argument("result_dir", type=Path, help="Results root; input/settings subdirectories are created automatically.")
    parser.add_argument("--profile", choices=PROFILES, default="balanced")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--msa-rows", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), help="Pair-update precision override; model weights remain FP32.")
    parser.add_argument("--refresh-features", action="store_true", help="Repeat official preprocessing and MSA search.")
    parser.add_argument("--offline", action="store_true", help="Do not submit sequences or download model files; reuse cache or local/empty MSAs.")
    parser.add_argument("--allow-msa-server", action="store_true", help="Allow sequence submission to https://api.colabfold.com for MSA search.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare/cache features without loading the prediction model.")
    parser.add_argument("--validate-only", action="store_true", help="Check YAML and local MSA files without network or GPU work.")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.msa_rows < 1:
        parser.error("--msa-rows must be positive.")
    try:
        description = input_description(args.input)
        if args.validate_only:
            print(json.dumps(description, indent=2))
            return
        job_dir = Path(description["job_dir"])
        if args.refresh_features or not (job_dir / "features.pt").exists() or not (job_dir / "features.json").exists():
            command = [str(HOME / "runtimes" / "boltz-preprocess" / "bin" / "python"), str(SCRIPT_DIR / "prepare_mac_features.py"), str(args.input)]
            if args.refresh_features:
                command.append("--refresh")
            if args.offline:
                command.append("--offline")
            if args.allow_msa_server:
                command.append("--allow-msa-server")
            environment = dict(os.environ, PYTHONPATH=str(HOME / "sources" / "boltz" / "src"))
            subprocess.run(command, env=environment, check=True)
        else:
            print(f"Using cached official features: {job_dir / 'features.pt'}", flush=True)
        if args.prepare_only:
            return
        current_input = description["input"]
        description = snapshot_features(job_dir)
        description["input"] = current_input
        predict(description, args)
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
