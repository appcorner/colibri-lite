import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from m6_3_r1_1a_cold_cache import ColdCacheError, arm, prepare, validate_contract, verify_launch


class ColdCacheProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "run-1"
        self.run.mkdir()
        self.artifact = self.run / "candidate.bin"
        self.artifact.write_bytes(b"candidate")
        os.chmod(self.artifact, 0o444)
        digest = __import__("hashlib").sha256(b"candidate").hexdigest()
        self.candidate_id = "cpu-safe-rust-int8-group64-layer0-r1-1a"
        self.contract = {
            "schema": "m6.3-r1.1a-cold-cache-contract-v1",
            "schema_version": 1,
            "status": "pre_registered_after_invalid_warm_cache_run",
            "temp_root": str(self.root),
            "candidates": [
                {"candidate_id": self.candidate_id, "artifact_bytes": 9, "artifact_sha256": digest},
                {"candidate_id": "cpu-safe-rust-int8-group32-layer0-r1-1a", "artifact_bytes": 679477248,
                 "artifact_sha256": "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2"},
            ],
            "prepare_phase": {"full_sha256_required": True, "read_only_required": True, "stable_file_identity_required": True},
            "reboot_boundary": {"minimum_boot_time_separation_seconds": 5, "maximum_arm_uptime_seconds": 1800},
            "arm_phase": {"payload_reads_permitted": False, "metadata_identity_only": True, "one_shot_authorization": True},
            "physical_io_gate": {"status": "correlated", "minimum_candidate_disk_read_bytes": 1, "total_events_lost": 0},
            "numerical_contract_changed": False,
            "authorizes_group32_before_group64_closure": False,
            "authorizes_r1_2": False,
        }
        self.prepare_boot = {"observed_unix_ns": 100_000_000_000, "uptime_ms": 60_000, "boot_time_unix_ns": 40_000_000_000}
        self.arm_boot = {"observed_unix_ns": 200_000_000_000, "uptime_ms": 10_000, "boot_time_unix_ns": 190_000_000_000}

        # The production validator freezes candidate identities. Keep the tiny
        # test payload while exercising the same state machine.
        self.contract["candidates"][0]["artifact_bytes"] = 641728512
        self.contract["candidates"][0]["artifact_sha256"] = "35a3ef6aba723d302fb1a7fded6ede4543a0c3dfcd35651596b78f1c5158cad2"

    def tearDown(self) -> None:
        os.chmod(self.artifact, 0o666)
        self.temporary.cleanup()

    def test_two_phase_authorization_is_one_shot(self) -> None:
        validate_contract(self.contract)
        prepare_path = self.run / "prepare.json"
        authorization_path = self.run / "authorization.json"
        with patch("m6_3_r1_1a_cold_cache.validate_contract"):
            self.contract["candidates"][0]["artifact_bytes"] = 9
            self.contract["candidates"][0]["artifact_sha256"] = __import__("hashlib").sha256(b"candidate").hexdigest()
            prepared = prepare(self.contract, self.candidate_id, self.artifact, prepare_path, self.prepare_boot)
            authorization = arm(self.contract, prepared, authorization_path, self.arm_boot)
            result = verify_launch(
                self.contract, authorization, self.candidate_id, self.artifact, authorization_path,
                consume=True, boot={**self.arm_boot, "observed_unix_ns": 201_000_000_000, "uptime_ms": 11_000},
            )
            self.assertTrue(result["consumed"])
            with self.assertRaisesRegex(ColdCacheError, "already consumed"):
                verify_launch(self.contract, authorization, self.candidate_id, self.artifact, authorization_path, boot=self.arm_boot)

    def test_arm_rejects_same_boot(self) -> None:
        with patch("m6_3_r1_1a_cold_cache.validate_contract"):
            self.contract["candidates"][0].update(artifact_bytes=9, artifact_sha256=__import__("hashlib").sha256(b"candidate").hexdigest())
            prepared = prepare(self.contract, self.candidate_id, self.artifact, self.run / "prepare.json", self.prepare_boot)
            same_boot = {"observed_unix_ns": 110_000_000_000, "uptime_ms": 70_000, "boot_time_unix_ns": 40_000_000_000}
            with self.assertRaisesRegex(ColdCacheError, "reboot boundary"):
                arm(self.contract, prepared, self.run / "authorization.json", same_boot)

    def test_arm_rejects_artifact_identity_change_without_hashing(self) -> None:
        with patch("m6_3_r1_1a_cold_cache.validate_contract"):
            self.contract["candidates"][0].update(artifact_bytes=9, artifact_sha256=__import__("hashlib").sha256(b"candidate").hexdigest())
            prepared = prepare(self.contract, self.candidate_id, self.artifact, self.run / "prepare.json", self.prepare_boot)
            os.chmod(self.artifact, 0o666)
            self.artifact.write_bytes(b"changed!!")
            os.chmod(self.artifact, 0o444)
            with patch("m6_3_r1_1a_cold_cache.sha256_file", side_effect=AssertionError("arm must not hash")):
                with self.assertRaisesRegex(ColdCacheError, "identity changed"):
                    arm(self.contract, prepared, self.run / "authorization.json", self.arm_boot)

    def test_rejects_gate_or_scope_weakening(self) -> None:
        changed = copy.deepcopy(self.contract)
        changed["physical_io_gate"]["minimum_candidate_disk_read_bytes"] = 0
        with self.assertRaisesRegex(ColdCacheError, "physical I/O gate"):
            validate_contract(changed)
        changed = copy.deepcopy(self.contract)
        changed["authorizes_r1_2"] = True
        with self.assertRaisesRegex(ColdCacheError, "R1.2"):
            validate_contract(changed)


if __name__ == "__main__":
    unittest.main()
