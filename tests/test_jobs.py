"""Local queue tests that do not load ColabFold or require launchd."""
import contextlib
import io
import json
from pathlib import Path
import plistlib
import tempfile
import unittest
from unittest.mock import Mock, patch

from macfoldkit import cli, jobs


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root / "runtime"
        (self.home / "runtimes/colabfold/bin").mkdir(parents=True)
        (self.home / "runtimes/colabfold/bin/python").touch()
        (self.home / "colabfold.json").write_text("{}")
        self.weights = jobs.weight_path(self.home, "alphafold2_ptm")
        self.weights.parent.mkdir(parents=True)
        self.weights.write_bytes(b"model")
        self.source = self.root / "input.fasta"
        self.source.write_text(">example\nACDEFGHIKLMNPQRSTVWY\n")
        self.store = jobs.JobStore(self.root / "Application Support/macfoldkit")

    def submit(self, **kwargs):
        return self.store.submit(self.source, self.home, "alphafold2_ptm", 3, 7, **kwargs)

    def test_submit_snapshots_input_and_checks_weights_before_accepting(self):
        job = self.submit()
        self.source.write_text(">modified\nAAAA\n")
        self.assertEqual(Path(self.store.details(job["id"])["input"]).read_text(),
                         ">example\nACDEFGHIKLMNPQRSTVWY\n")
        self.assertGreater(self.store.storage_bytes(job["id"]), 0)
        self.weights.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing local weights"):
            self.submit()
        self.assertEqual(len(self.store.all()), 1)

    def test_rejects_ambiguous_and_incomplete_inputs(self):
        cases = [
            ("input.fasta", ">one\nAAA\n>two\nBBB\n"),
            ("input.fasta", ">one\nAAA:BBB\n"),
            ("input.a3m", ">query\nACDE\n>truncated\nACD\n"),
            ("input.a3m", ">query\nACDE\n>truncated\n"),
            ("input.a3m", ">query\nACDE\n>broken\nACD?\n"),
            ("input.a3m", "#4,4\t1\n>query\nACDEFGHI\n"),
            ("input.a3m", "#4,4\t1,0\n>query\nACDEFGHI\n"),
        ]
        for name, content in cases:
            with self.subTest(content=content):
                source = self.root / name
                source.write_text(content)
                with self.assertRaises(ValueError):
                    self.store.submit(source, self.home, "alphafold2_ptm", 3, 7)
        self.assertEqual(self.store.all(), [])

    def test_accepts_single_local_a3m_and_multimer_inputs(self):
        a3m = self.root / "local.a3m"
        a3m.write_text(">query\nACDE\n>aligned\nACeDE\n")
        self.assertEqual(self.store.submit(a3m, self.home, "alphafold2_ptm", 1, 7)["status"],
                         "pending")
        self.source.write_text(">complex\nACDE:FGHI\n")
        weights = jobs.weight_path(self.home, "alphafold2_multimer_v3")
        weights.write_bytes(b"model")
        self.assertEqual(self.store.submit(self.source, self.home, "alphafold2_multimer_v3",
                                           1, 7)["status"], "pending")
        a3m.write_text("#4,4\t1,1\n>query\nACDEFGHI\n>aligned\nACDEFGHI\n")
        self.assertEqual(self.store.submit(a3m, self.home, "alphafold2_multimer_v3",
                                           1, 7)["status"], "pending")

    def test_cancel_retry_and_delete_retains_attempt_history_until_explicit_removal(self):
        job = self.submit()
        job_id = job["id"]
        self.assertEqual(self.store.change(job_id, "cancel")["status"], "cancelled")
        with self.assertRaisesRegex(ValueError, "Cannot cancel"):
            self.store.change(job_id, "cancel")
        self.store.change(job_id, "retry")
        claimed = self.store.claim()
        self.assertEqual((claimed["id"], claimed["attempt"]), (job_id, 1))
        self.store.update(job_id, status="failed", ended_at=jobs.now(), error="mock error")
        self.store.change(job_id, "retry")
        self.assertEqual(self.store.claim()["attempt"], 2)
        self.store.recover()
        self.assertEqual(self.store.get(job_id)["status"], "interrupted")
        self.store.change(job_id, "retry")
        self.assertEqual(self.store.claim()["attempt"], 3)
        with self.assertRaisesRegex(ValueError, "running job"):
            self.store.delete(job_id)
        self.store.update(job_id, status="succeeded")
        self.store.delete(job_id)
        self.assertFalse(self.store.folder(job_id).exists())
        with self.assertRaisesRegex(ValueError, "Unknown job"):
            self.store.get(job_id)

    def test_worker_uses_offline_caffeinate_and_records_results(self):
        job = self.submit()
        captured = {}

        class FinishedProcess:
            pid = 345678
            returncode = 0

            def poll(self):
                return 0

        def start(command, **kwargs):
            captured.update(command=command, env=kwargs["env"],
                            start_new_session=kwargs["start_new_session"])
            return FinishedProcess()

        with patch.object(jobs, "supported_platform", return_value=True), \
             patch.object(jobs.subprocess, "Popen", side_effect=start):
            jobs.worker(self.store, once=True)
        result = self.store.details(job["id"])
        self.assertEqual(result["status"], "succeeded")
        self.assertIsNone(result["pid"])
        self.assertEqual(result["attempt"], 1)
        self.assertTrue(captured["start_new_session"])
        self.assertEqual(captured["command"][:2], ["/usr/bin/caffeinate", "-i"])
        self.assertIn("--local-only", captured["command"])
        self.assertEqual(captured["command"][captured["command"].index("--msa-mode") + 1],
                         "single_sequence")
        self.assertEqual(captured["env"]["HF_HUB_OFFLINE"], "1")
        self.assertTrue(Path(result["log"]).is_file())
        self.assertEqual(self.store.storage_bytes(job["id"]), result["storage_bytes"])

    def test_worker_checks_again_before_run_and_surfaces_failure(self):
        job = self.submit()
        self.weights.unlink()
        with patch.object(jobs, "supported_platform", return_value=True), \
             patch.object(jobs.subprocess, "Popen") as popen:
            jobs.worker(self.store, once=True)
            popen.assert_not_called()
        result = self.store.details(job["id"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("Missing local weights", result["error"])
        self.assertIn("Missing local weights", Path(result["log"]).read_text())

    def test_running_cancel_requests_worker_termination(self):
        job = self.submit()
        class RunningProcess:
            pid = 345678
            returncode = None

            def poll(self):
                return self.returncode

        process = RunningProcess()

        def tick(_):
            self.store.change(job["id"], "cancel")

        def terminate(proc):
            proc.returncode = -15

        with patch.object(jobs, "supported_platform", return_value=True), \
             patch.object(jobs.subprocess, "Popen", return_value=process), \
             patch.object(jobs.time, "sleep", side_effect=tick), \
             patch.object(jobs, "_terminate_group", side_effect=terminate):
            jobs.worker(self.store, once=True)
        self.assertEqual(self.store.get(job["id"])["status"], "cancelled")
        self.assertIsNone(self.store.get(job["id"])["pid"])

    def test_restart_waits_for_possible_orphaned_gpu_process(self):
        job = self.submit()
        self.store.claim()
        self.store.update(job["id"], pid=345678)
        with patch.object(jobs.os, "kill", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "may still be running"):
                self.store.recover()
        self.assertEqual(self.store.get(job["id"])["status"], "running")
        with patch.object(jobs.os, "kill", side_effect=ProcessLookupError):
            self.store.recover()
        self.assertEqual(self.store.get(job["id"])["status"], "interrupted")
        self.assertIsNone(self.store.get(job["id"])["pid"])

    def test_cli_json_and_explicit_delete(self):
        with patch.object(jobs, "jobs_home", return_value=self.store.root):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(["--home", str(self.home), "jobs", "submit",
                                           str(self.source)]), 0)
            job_id = json.loads(output.getvalue())["id"]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                cli.main(["jobs", "status", job_id])
            self.assertEqual(json.loads(output.getvalue())["status"], "pending")
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cli.main(["jobs", "delete", job_id]), 1)
            self.assertIn("--yes", error.getvalue())
            with contextlib.redirect_stdout(io.StringIO()) as output:
                cli.main(["jobs", "delete", job_id, "--yes"])
            self.assertEqual(json.loads(output.getvalue()), {"deleted": job_id})

    def test_launch_agent_uses_installed_python_and_does_not_delete_jobs(self):
        agent_home = self.root / "user"
        with patch.object(jobs.Path, "home", return_value=agent_home), \
             patch.object(jobs, "supported_platform", return_value=True), \
             patch.object(jobs, "_loaded", side_effect=(False, True, True, False)), \
             patch.object(jobs.subprocess, "run", return_value=Mock(returncode=0)) as run:
            config = jobs.install_worker(self.store)
            plist = plistlib.loads(Path(config["launch_agent"]).read_bytes())
            self.assertEqual(plist["ProgramArguments"],
                             [jobs.sys.executable, "-m", "macfoldkit.cli", "jobs", "worker"])
            self.assertEqual(plist["KeepAlive"], True)
            self.assertEqual(plist["StandardOutPath"], str(self.store.root / "worker.log"))
            self.assertEqual(run.call_args.args[0][:2], ["/bin/launchctl", "bootstrap"])
            self.assertFalse(jobs.uninstall_worker()["installed"])
        self.assertFalse(Path(config["launch_agent"]).exists())


if __name__ == "__main__":
    unittest.main()
