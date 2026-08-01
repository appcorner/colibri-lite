import copy
import json
import unittest
from pathlib import Path

from validate_m6_3_r1_1a_candidate_telemetry import validate_contract, validate_run


class CandidateTelemetryValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repository = Path(__file__).resolve().parents[1]
        cls.contract = json.loads(
            (repository / "models/qwen3-30b-a3b/m6.3-r1-1a-candidate-telemetry-contract-v1.json").read_text()
        )
        cls.valid_record = {
            "schema": "m6.3-r1.1a-candidate-run-telemetry-v1",
            "candidate_id": "cpu-safe-rust-int8-group64-layer0-r1-1a",
            "fixture": {"id": "code_newline", "token_ids": [87, 28, 16, 198]},
            "run_directory": "D:\\tmp\\colibri-lite-runs\\run-1",
            "artifact": {
                "path": "D:\\tmp\\colibri-lite-runs\\run-1\\candidate.bin",
                "group_size": 64,
                "bytes": 641728512,
                "sha256": "35a3ef6aba723d302fb1a7fded6ede4543a0c3dfcd35651596b78f1c5158cad2",
            },
            "persisted_process": {
                "status": "completed", "exit_code": 0, "child_pid": 42,
                "stdout_sha256": "1" * 64, "stderr_sha256": "2" * 64,
            },
            "direct_consumption": {
                "verification_bytes_read": 641728512,
                "payload_bytes_read": 5013504 * 32,
                "peak_packed_expert_bytes": 5013504,
                "complete_f32_weight_materializations": 0,
            },
            "process_memory": {
                "samples": 10, "sample_interval_ms": 100,
                "working_set_peak_bytes": 1, "private_bytes_peak": 1,
            },
            "physical_io": {
                "status": "correlated", "pid": 42,
                "artifact_path": "D:\\tmp\\colibri-lite-runs\\run-1\\candidate.bin",
                "candidate_disk_read_bytes": 4096, "total_events_lost": 0,
            },
            "layer0_checkpoint_sha256": "3" * 64,
            "final_logits_sha256": "4" * 64,
        }

    def test_contract_and_valid_run(self) -> None:
        validate_contract(self.contract)
        validate_run(self.valid_record, self.contract)

    def test_rejects_reference_or_old_candidate(self) -> None:
        record = copy.deepcopy(self.valid_record)
        record["candidate_id"] = "cpu-safe-rust-int8-group128-layer0-v1"
        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            validate_run(record, self.contract)

    def test_rejects_artifact_outside_run(self) -> None:
        record = copy.deepcopy(self.valid_record)
        record["artifact"]["path"] = "D:\\models\\candidate.bin"
        with self.assertRaisesRegex(ValueError, "flat run"):
            validate_run(record, self.contract)

    def test_rejects_f32_materialization(self) -> None:
        record = copy.deepcopy(self.valid_record)
        record["direct_consumption"]["complete_f32_weight_materializations"] = 1
        with self.assertRaisesRegex(ValueError, "F32"):
            validate_run(record, self.contract)

    def test_rejects_pid_or_physical_io_substitution(self) -> None:
        record = copy.deepcopy(self.valid_record)
        record["physical_io"]["pid"] = 43
        with self.assertRaisesRegex(ValueError, "PID"):
            validate_run(record, self.contract)
        record = copy.deepcopy(self.valid_record)
        record["physical_io"]["status"] = "not_measured"
        with self.assertRaisesRegex(ValueError, "PID"):
            validate_run(record, self.contract)


if __name__ == "__main__":
    unittest.main()
