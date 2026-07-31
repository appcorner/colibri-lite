"""Contract checks for the tracked M6.2 planner-schema definition."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs/schemas/planner-contract-v1.schema.json"
SHA = "a" * 64


def request() -> dict[str, object]:
    return {
        "schema": "colibri-lite-planner-request-v1",
        "schema_version": 1,
        "request_id": "planner-request-fixture-v1",
        "hardware_profile": {"profile_id": "doctor-v1", "document_sha256": SHA},
        "model_profile": {"profile_id": "model-v1", "document_sha256": SHA},
        "workload": {"workload_id": "interactive-v1", "prefill_tokens": 16, "decode_tokens": 32, "context_tokens": 128},
        "budgets": {"ram_budget_bytes": 1024, "vram_budget_bytes": 0},
    }


class M62PlannerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.validator = Draft202012Validator(self.schema)

    def errors(self, document: dict[str, object]) -> list[object]:
        return list(self.validator.iter_errors(document))

    def test_request_requires_hashed_profile_provenance_and_explicit_budgets(self) -> None:
        self.assertEqual(self.schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(self.errors(request()))
        missing_budget = request()
        del missing_budget["budgets"]
        self.assertTrue(self.errors(missing_budget))

    def test_result_accepts_measured_input_estimate_and_budget_rejection(self) -> None:
        result = {
            "schema": "colibri-lite-planner-result-v1",
            "schema_version": 1,
            "result_id": "planner-result-fixture-v1",
            "request": request(),
            "candidate_plans": [{
                "plan_id": "cpu-ram-reference-f32",
                "placement": {"backend_id": "cpu", "dense_location": "ram", "expert_location": "ram"},
                "precision_candidate_id": "reference-f32-v1",
                "resources": {"ram_bytes": 1000, "vram_bytes": 0, "expert_cache_bytes": 0, "max_context_tokens": 128, "disk_bytes_per_token": 0, "startup_seconds": 0},
                "estimate": {"status": "available", "method": "analytical-v1", "confidence": "measured_inputs", "prefill_tokens_per_second": 1.0, "decode_tokens_per_second": 1.0, "measurement_references": [{"profile_kind": "hardware", "profile_id": "doctor-v1", "document_sha256": SHA, "measurement_id": "cpu-kernels.scalar"}]},
                "quality_risk": {"level": "none", "reference_id": "reference-f32-v1"},
            }],
            "rejections": [{"plan_id": "gpu-vram", "code": "vram_budget_exceeded", "reason": "required device memory exceeds the requested budget", "required": 1, "limit": 0, "unit": "bytes"}],
            "ranking": ["cpu-ram-reference-f32"],
        }
        self.assertFalse(self.errors(result))

    def test_available_estimate_and_resource_rejection_cannot_omit_evidence(self) -> None:
        result = {
            "schema": "colibri-lite-planner-result-v1", "schema_version": 1, "result_id": "invalid-result", "request": request(),
            "candidate_plans": [{
                "plan_id": "invalid", "placement": {"backend_id": "cpu", "dense_location": "ram", "expert_location": "ram"}, "precision_candidate_id": "reference-f32-v1",
                "resources": {"ram_bytes": 0, "vram_bytes": 0, "expert_cache_bytes": 0, "max_context_tokens": 0, "disk_bytes_per_token": 0, "startup_seconds": 0},
                "estimate": {"status": "available", "method": "analytical-v1", "confidence": "measured_inputs"},
                "quality_risk": {"level": "unvalidated", "reference_id": "reference-f32-v1"},
            }],
            "rejections": [{"plan_id": "invalid", "code": "ram_budget_exceeded", "reason": "no requirement"}],
            "ranking": [],
        }
        self.assertTrue(self.errors(result))


if __name__ == "__main__":
    unittest.main()
