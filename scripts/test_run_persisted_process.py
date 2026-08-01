import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name("run_persisted_process.py")


class PersistedProcessRunnerTests(unittest.TestCase):
    def execute(self, command: list[str]) -> tuple[subprocess.CompletedProcess[str], Path, dict]:
        temporary = tempfile.TemporaryDirectory(prefix="colibri-persisted-runner-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        evidence = root / "evidence"
        request = root / "request.json"
        request.write_text(
            json.dumps(
                {
                    "cwd": str(root),
                    "command": command,
                    "environment": {"COLIBRI_RUNNER_TEST": "present"},
                    "evidence_directory": str(evidence),
                }
            ),
            encoding="utf-8",
        )
        process = subprocess.run(
            [sys.executable, str(RUNNER), str(request)],
            check=False,
            capture_output=True,
            text=True,
        )
        record = json.loads((evidence / "exit.json").read_text(encoding="utf-8"))
        for stream in ("stdout", "stderr"):
            payload = (evidence / f"{stream}.log").read_bytes()
            self.assertEqual(record[stream]["bytes"], len(payload))
            self.assertEqual(record[stream]["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertFalse((evidence / "exit.json.incomplete").exists())
        return process, evidence, record

    def test_success_preserves_argv_environment_and_evidence(self) -> None:
        process, evidence, record = self.execute(
            [
                sys.executable,
                "-c",
                "import os,sys; print(os.environ['COLIBRI_RUNNER_TEST']); print('err', file=sys.stderr)",
            ]
        )
        self.assertEqual(process.returncode, 0)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual((evidence / "stdout.log").read_text().strip(), "present")
        self.assertEqual((evidence / "stderr.log").read_text().strip(), "err")

    def test_nonzero_child_exit_is_preserved(self) -> None:
        process, _, record = self.execute([sys.executable, "-c", "raise SystemExit(7)"])
        self.assertEqual(process.returncode, 7)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["exit_code"], 7)

    def test_launch_failure_has_nonnull_exit_and_logs(self) -> None:
        process, evidence, record = self.execute(["colibri-executable-that-does-not-exist"])
        self.assertEqual(process.returncode, 127)
        self.assertEqual(record["status"], "launch_failed")
        self.assertEqual(record["exit_code"], 127)
        self.assertTrue((evidence / "stdout.log").exists())
        self.assertTrue((evidence / "stderr.log").exists())


if __name__ == "__main__":
    unittest.main()
