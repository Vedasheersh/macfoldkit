"""Test CLI routing and local cache paths without loading JAX or model weights."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


RUNNER = Path(__file__).resolve().parents[1] / "src/macfoldkit/runners/colabfold/run_mps.py"
spec = importlib.util.spec_from_file_location("macfoldkit_colabfold_runner_test", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ColabFoldRunnerTests(unittest.TestCase):
    def run_fake_prediction(self, extra=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            home, output = root / "user cache", root / "predictions"
            source = root / "input.fasta"
            source.write_text(">example\nACDEFGHIKLMNPQRSTVWY\n")
            observed = {}

            def popen(command, **kwargs):
                observed.update(command=command, env=kwargs["env"])
                Path(kwargs["env"]["COLABFOLD_DEVICE_AUDIT_PATH"]).write_text(
                    json.dumps({"ok": True, "platforms": ["mps"]}) + "\n"
                )
                (output / "example_unrelaxed_rank_001_model_1.pdb").write_text("END\n")
                process = unittest.mock.Mock()
                process.stdout = io.StringIO("mocked prediction\n")
                process.wait.return_value = 0
                return process

            with patch.dict(os.environ, {"MACFOLDKIT_HOME": str(home)}, clear=True), \
                 patch.object(runner.sys, "argv", [str(RUNNER), str(source), str(output), *extra]), \
                 patch.object(runner.subprocess, "Popen", side_effect=popen), \
                 patch.object(runner.importlib.metadata, "version", return_value="test-version"), \
                 patch("sys.stdout", new_callable=io.StringIO):
                runner.main()
            observed["metrics"] = json.loads((output / "benchmark.json").read_text())
            observed["home"] = str(home)
            return observed

    def test_default_routes_single_sequence_and_portable_weight_directory(self):
        observed = self.run_fake_prediction()
        command, env = observed["command"], observed["env"]
        self.assertEqual(command[command.index("--msa-mode") + 1], "single_sequence")
        self.assertEqual(command[command.index("--max-msa") + 1], "1:1")
        self.assertEqual(command[command.index("--data") + 1], observed["home"] + "/weights/colabfold")
        self.assertEqual(env["XDG_CACHE_HOME"], observed["home"] + "/cache")
        self.assertEqual(env["JAX_PLATFORMS"], "mps,cpu")
        self.assertEqual(env["COLABFOLD_OPTIMIZED_BATCHING"], "1")
        self.assertTrue(observed["metrics"]["model_output_devices_verified"])

    def test_explicit_msa_mode_and_multimer_options_reach_colabfold(self):
        observed = self.run_fake_prediction(("--msa-mode", "mmseqs2_uniref_env", "--model-type", "alphafold2_multimer_v3", "--no-optimized-batching"))
        command = observed["command"]
        self.assertEqual(command[command.index("--msa-mode") + 1], "mmseqs2_uniref_env")
        self.assertEqual(command[command.index("--max-msa") + 1], "128:256")
        self.assertEqual(command[-2:], ["--model-type", "alphafold2_multimer_v3"])
        self.assertEqual(observed["env"]["COLABFOLD_OPTIMIZED_BATCHING"], "0")

    def test_input_and_output_are_required(self):
        with patch.object(runner.sys, "argv", [str(RUNNER)]), patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                runner.main()
        self.assertEqual(error.exception.code, 2)

    def test_missing_input_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.object(runner.sys, "argv", [str(RUNNER), str(Path(directory) / "missing.fasta"), str(output)]), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    runner.main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
