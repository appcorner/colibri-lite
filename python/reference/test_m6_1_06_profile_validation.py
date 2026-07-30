"""M6.1-06 validation of the tracked doctor and model-profile artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs" / "benchmarks"
SCHEMAS = ROOT / "docs" / "schemas"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class M6106ProfileValidationTests(unittest.TestCase):
    def test_doctor_profile_validates_and_preserves_unavailable_gpu_state(self) -> None:
        document = load(BENCHMARKS / "m6.1-05-doctor-v1.json")
        schema = load(SCHEMAS / "hardware-profile-v1.schema.json")
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(document)), [])
        self.assertEqual(document["recommendations"]["confidence"], "partial")
        self.assertEqual(document["recommendations"]["vram_budget_bytes"], 0)
        for backend in document["measurements"]["backends"]:
            self.assertEqual(backend["availability"], "unavailable")
            self.assertEqual(backend["vram"]["safe_budget_bytes"], 0)
            self.assertEqual(backend["host_to_device"]["status"], "not_run")
            self.assertEqual(backend["device_to_host"]["status"], "not_run")

    def test_model_profile_validates_and_matches_frozen_quality_reference(self) -> None:
        document = load(BENCHMARKS / "m6.1-05-model-profile-v1.json")
        schema = load(SCHEMAS / "model-profile-v1.schema.json")
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(document)), [])
        quality_path = ROOT / "models" / "qwen3-30b-a3b" / "m6.0-04-quality-fixtures-v1.json"
        self.assertEqual(
            document["quality_reference"]["bilingual_fixture_manifest_sha256"],
            hashlib.sha256(quality_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(document["execution"]["precision_candidates"][0]["status"], "accepted_reference")
        self.assertEqual(document["execution"]["precision_candidates"][0]["weight_dtype"], "F32")

    def test_validation_evidence_records_repeatability_and_handle_release(self) -> None:
        evidence = load(BENCHMARKS / "m6.1-06-profile-validation-v1.json")
        self.assertEqual(evidence["schema_validation"]["hardware_profile_v1"], "passed")
        self.assertEqual(evidence["schema_validation"]["model_profile_v1"], "passed")
        self.assertTrue(evidence["repeatability"]["doctor"]["byte_identical"])
        self.assertTrue(evidence["repeatability"]["model"]["byte_identical"])
        self.assertEqual(evidence["windows_resource_release"]["result"], "passed")


if __name__ == "__main__":
    unittest.main()
