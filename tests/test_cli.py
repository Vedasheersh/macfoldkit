import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from macfoldkit import cli
from macfoldkit.config import runtime_env
from macfoldkit.install import patch_mosaic


class CliTests(unittest.TestCase):
    def test_frontend_routes_backend_flags_after_positionals_and_preserves_spaces(self):
        with patch.object(cli, "require_runtime", return_value=Path("/tmp/model env/bin/python")), \
             patch.object(cli.subprocess, "run", return_value=Mock(returncode=0)) as run:
            code = cli.main(["--home", "/tmp/cache root", "fold", "input sequence.a3m", "output dir",
                             "--backend", "colabfold", "--", "--num-recycle", "1"])
        self.assertEqual(code, 0)
        command = run.call_args.args[0]
        self.assertTrue(command[1].endswith("colabfold/run_mps.py"))
        self.assertEqual(command[2:], ["input sequence.a3m", "output dir", "--num-recycle", "1"])
        self.assertEqual(run.call_args.kwargs["env"]["MACFOLDKIT_HOME"], str(Path("/tmp/cache root").resolve()))

    def test_design_sets_requested_cpu_without_loading_gpu_in_frontend(self):
        with patch.object(cli, "require_runtime", return_value=Path("/tmp/python")), \
             patch.object(cli.subprocess, "run", return_value=Mock(returncode=3)) as run:
            code = cli.main(["design", "backbone.cif", "out", "--device", "cpu", "--seed", "19"])
        self.assertEqual(code, 3)
        self.assertEqual(run.call_args.kwargs["env"]["JAX_PLATFORMS"], "cpu")
        self.assertEqual(run.call_args.args[0][-2:], ["--seed", "19"])

    def test_offline_reaches_model_and_hub(self):
        with patch.object(cli, "require_runtime", return_value=Path("/tmp/python")), \
             patch.object(cli.subprocess, "run", return_value=Mock(returncode=0)) as run:
            cli.main(["fold", "in.yaml", "out", "--offline"])
        self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertIn("--offline", run.call_args.args[0])

    def test_unknown_setup_argument_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["setup", "mosaic", "--bogus"])

    def test_missing_runtime_returns_actionable_error(self):
        with tempfile.TemporaryDirectory() as home, contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(cli.main(["--home", home, "design", "in.pdb", "out"]), 1)
        self.assertIn("setup mosaic", error.getvalue())

    def test_old_experimental_backend_does_not_leak_into_runtime(self):
        with patch.dict(os.environ, {"JAX_MPS_LIBRARY_PATH": "bad", "MLX_METAL_GPU_ARCH": "bad", "PYTHONPATH": "bad", "PYTHONOPTIMIZE": "1"}):
            env = runtime_env(Path("/tmp/new"), "mosaic")
        self.assertNotIn("JAX_MPS_LIBRARY_PATH", env)
        self.assertNotIn("MLX_METAL_GPU_ARCH", env)
        self.assertNotIn("PYTHONOPTIMIZE", env)
        self.assertEqual(env["PYTHONPATH"], "/tmp/new/sources/mosaic/src")

    def test_source_patch_refuses_unrelated_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            target = source / "src/mosaic/proteinmpnn/mpnn.py"
            target.parent.mkdir(parents=True)
            target.write_text("user changes")
            with patch("macfoldkit.install.subprocess.check_output", return_value="from joltz.backend import (\n"):
                with self.assertRaisesRegex(RuntimeError, "unexpected edits"):
                    patch_mosaic(source)
            self.assertEqual(target.read_text(), "user changes")


if __name__ == "__main__":
    unittest.main()
