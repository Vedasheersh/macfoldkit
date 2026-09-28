"""CPU-only guards for YAML/cache identity and protein coordinate export."""
from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

import yaml
import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/macfoldkit/runners/boltz"))

from predict_mac import CIF_ATOM_TAGS, repair_and_check_cif
from prepare_mac_features import input_description, prepare, snapshot_features


class MacRunnerGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.input = self.root / "input.yaml"

    def describe(self, entries):
        self.input.write_text(yaml.safe_dump({"version": 1, "sequences": entries}))
        return input_description(self.input)

    def test_local_msa_bytes_invalidate_cache(self):
        msa = self.root / "local.a3m"
        msa.write_text(">query\nACD\n")
        entries = [{"protein": {"id": "A", "sequence": "ACD", "msa": "local.a3m"}}]
        first = self.describe(entries)
        self.assertEqual(first["input_sha256"], self.describe(entries)["input_sha256"])
        self.assertIn(str(msa.resolve()), first["local_msa_sha256"])
        msa.write_text(">query\nACD\n>hit\nACE\n")
        self.assertNotEqual(first["input_sha256"], self.describe(entries)["input_sha256"])

    def test_chain_copy_ids(self):
        result = self.describe([{"protein": {"id": ["X", "Z9"], "sequence": "ACD", "msa": "empty"}}])
        self.assertEqual(result["chains"], {"X": "ACD", "Z9": "ACD"})
        self.assertFalse(result["needs_msa_server"])

    def test_duplicate_ids_within_list_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate chain"):
            self.describe([{"protein": {"id": ["A", "A"], "sequence": "ACD"}}])

    def test_duplicate_ids_across_lists_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate chain"):
            self.describe([
                {"protein": {"id": ["A", "B"], "sequence": "ACD"}},
                {"protein": {"id": ["C", "B"], "sequence": "DEF"}},
            ])

    def test_unsupported_entity_rejected(self):
        with self.assertRaisesRegex(ValueError, "Only protein"):
            self.describe([{"ligand": {"id": "L", "smiles": "CC"}}])

    def test_offline_missing_msa_fails_before_heavy_imports(self):
        description = self.describe([{"protein": {"id": "A", "sequence": "ACD"}}])
        description["job_dir"] = str(self.root / "cache")
        with self.assertRaisesRegex(ValueError, "offline"):
            prepare(description, offline=True)

    def test_public_msa_requires_explicit_permission(self):
        description = self.describe([{"protein": {"id": "A", "sequence": "ACD"}}])
        description["job_dir"] = str(self.root / "cache")
        with self.assertRaisesRegex(ValueError, "allow-msa-server"):
            prepare(description)

    def test_offline_cached_features_reused(self):
        import json
        description = self.describe([{"protein": {"id": "A", "sequence": "ACD"}}])
        cache = self.root / "cache"
        cache.mkdir()
        description["job_dir"] = str(cache)
        (cache / "features.pt").write_bytes(b"sentinel; not loaded by cache lookup")
        report = {"feature_path": str(cache / "features.pt")}
        (cache / "features.json").write_text(json.dumps(report))
        self.assertEqual(prepare(description, offline=True), report)

    def test_refresh_keeps_prior_feature_snapshot_and_provenance(self):
        import hashlib
        import json
        cache = self.root / "cache"
        cache.mkdir()
        source = cache / "features.pt"
        metadata = cache / "features.json"
        first_bytes = b"first official features"
        source.write_bytes(first_bytes)
        metadata.write_text(json.dumps({"feature_path": str(source), "processed_dir": "preprocess-first"}))
        first = snapshot_features(cache)
        first_path = Path(first["feature_path"])
        self.assertEqual(first["feature_sha256"], hashlib.sha256(first_bytes).hexdigest())
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertNotEqual(first_path, source)
        # Simulate the atomic replacements made by an official cache refresh.
        replacement = cache / "features.tmp.pt"
        replacement.write_bytes(b"refreshed official features")
        replacement.replace(source)
        metadata.write_text(json.dumps({"feature_path": str(source), "processed_dir": "preprocess-second"}))
        second = snapshot_features(cache)
        self.assertNotEqual(first["feature_path"], second["feature_path"])
        self.assertNotEqual(first["feature_sha256"], second["feature_sha256"])
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertEqual(first["processed_dir"], "preprocess-first")
        self.assertEqual(second["processed_dir"], "preprocess-second")
        self.assertEqual(Path(second["feature_path"]).read_bytes(), source.read_bytes())
        # Reusing the same bytes must preserve the existing snapshot file.
        inode = Path(second["feature_path"]).stat().st_ino
        self.assertEqual(snapshot_features(cache), second)
        self.assertEqual(Path(second["feature_path"]).stat().st_ino, inode)

    def cif(self, rows):
        path = self.root / "prediction.cif"
        header = ["data_test", "#", "loop_"] + [f"_atom_site.{tag}" for tag in CIF_ATOM_TAGS]
        path.write_text("\n".join(header + rows + ["#"]) + "\n")
        return path

    def test_cif_residue_numbering_restarts_per_chain(self):
        path = self.cif([
            "ATOM 1 C CA ALA X 1 1 2 3 1.00 90 1",
            "ATOM 2 C CA CYS X 2 1 2 3 1.00 90 1",
            "ATOM 3 C CA ASP Z9 3 1 2 3 1.00 90 1",
        ])
        repair_and_check_cif(path, [{"id": "X", "sequence": "AC"}, {"id": "Z9", "sequence": "D"}], 3)
        self.assertIn("CA ASP Z9 1 ", path.read_text())

    def test_cif_loads_as_protein_and_preserves_coordinates_and_chain_ids(self):
        chains = [{"id": "X", "sequence": "AC"}, {"id": "Z9", "sequence": "D"}]
        rows, expected = [], []
        for chain, global_id, local_id, residue in [
            ("X", 1, 1, "ALA"), ("X", 2, 2, "CYS"), ("Z9", 3, 1, "ASP"),
        ]:
            for atom, element in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]:
                serial = len(rows) + 1
                xyz = (serial + 0.125, -serial - 0.25, serial + 0.5)
                rows.append(
                    f"ATOM {serial} {element} {atom} {residue} {chain} {global_id} "
                    f"{xyz[0]} {xyz[1]} {xyz[2]} 1.00 90 1"
                )
                expected.append((chain, local_id, atom, xyz))
        path = self.cif(rows)
        repair_and_check_cif(path, chains, len(rows))
        structure = gemmi.read_structure(str(path))
        self.assertEqual(len(structure), 1)
        self.assertEqual([(c.name, len(c)) for c in structure[0]], [("X", 2), ("Z9", 1)])
        actual = [
            (c.name, r.seqid.num, a.name, (a.pos.x, a.pos.y, a.pos.z))
            for c in structure[0] for r in c for a in r
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(len(structure.entities), 2)
        for chain in structure[0]:
            for residue in chain:
                self.assertEqual(residue.entity_type, gemmi.EntityType.Polymer)
                self.assertEqual(
                    structure.get_entity(residue.entity_id).polymer_type, gemmi.PolymerType.PeptideL
                )
                self.assertEqual([a.altloc for a in residue], ["\x00"] * 4)
        # Mosaic applies these exact cleanup calls before extracting backbones.
        structure.remove_ligands_and_waters()
        structure.remove_alternative_conformations()
        structure.remove_empty_chains()
        self.assertEqual(structure[0].count_atom_sites(), len(rows))
        self.assertEqual([(c.name, len(c)) for c in structure[0]], [("X", 2), ("Z9", 1)])
        new_rows = [line.split() for line in path.read_text().splitlines() if line.startswith("ATOM ")]
        for old, new, (_, local_id, _, _) in zip(rows, new_rows, expected):
            old_fields = old.split()
            old_fields[6] = str(local_id)
            self.assertEqual(new[:13], old_fields)

    def test_cif_same_sequence_chains_share_entity(self):
        path = self.cif([
            "ATOM 1 C CA ALA X 1 1 2 3 1.00 90 1",
            "ATOM 2 C CA ALA Z9 2 4 5 6 1.00 90 1",
        ])
        repair_and_check_cif(path, [{"id": "X", "sequence": "A"}, {"id": "Z9", "sequence": "A"}], 2)
        structure = gemmi.read_structure(str(path))
        self.assertEqual(len(structure.entities), 1)
        self.assertEqual(structure[0][0][0].entity_id, structure[0][1][0].entity_id)
        self.assertEqual([(c.name, c[0].seqid.num) for c in structure[0]], [("X", 1), ("Z9", 1)])
        self.assertEqual(list(structure.entities[0].subchains), ["X", "Z9"])

    def test_cif_missing_loop_metadata_rejected(self):
        path = self.root / "prediction.cif"
        original = "data_test\nATOM 1 C CA ALA X 1 1 2 3 1.00 90 1\n"
        path.write_text(original)
        with self.assertRaisesRegex(RuntimeError, "Unexpected CIF atom table"):
            repair_and_check_cif(path, [{"id": "X", "sequence": "A"}], 1)
        self.assertEqual(path.read_text(), original)

    def test_cif_incorrect_atom_count_rejected(self):
        path = self.cif(["ATOM 1 C CA ALA X 1 1 2 3 1.00 90 1"])
        with self.assertRaisesRegex(RuntimeError, "atom, chain or C-alpha"):
            repair_and_check_cif(path, [{"id": "X", "sequence": "A"}], 5)

    def test_cif_duplicate_atom_rejected(self):
        path = self.cif([
            "ATOM 1 C CA ALA X 1 1 2 3 1.00 90 1",
            "ATOM 2 C CA ALA X 1 1 2 3 1.00 90 1",
        ])
        with self.assertRaisesRegex(RuntimeError, "Duplicate atom"):
            repair_and_check_cif(path, [{"id": "X", "sequence": "A"}], 2)

    def test_cif_nonfinite_values_rejected(self):
        path = self.cif(["ATOM 1 C CA ALA X 1 nan 2 3 1.00 90 1"])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            repair_and_check_cif(path, [{"id": "X", "sequence": "A"}], 1)


if __name__ == "__main__":
    unittest.main()
