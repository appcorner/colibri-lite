"""Validation and reproducibility tests for the M4.4 baseline index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from python.reference.build_m4_4_baseline import BASELINE_ID, build, canonical_bytes


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models" / "qwen3-30b-a3b"
BASELINE_PATH = MODEL_ROOT / "m4.4-performance-baseline-v1.json"


class M44BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    def test_required_identity_and_sections(self) -> None:
        self.assertEqual(self.document["baseline_id"], BASELINE_ID)
        self.assertEqual(
            self.document["schema"], "colibri-qwen3-moe-m4.4-performance-baseline-v1"
        )
        for section in (
            "model_identity",
            "runtime_identity",
            "correctness_baseline",
            "performance_baseline",
            "external_optimized_reference",
            "quantization_decision_registry",
            "frozen_optimization_invariants",
            "memory_hierarchy_study_inputs",
            "future_comparison_record_schema",
            "success_gates",
            "references",
        ):
            self.assertIn(section, self.document)

    def test_references_exist_and_hashes_match(self) -> None:
        roles = set()
        for reference in self.document["references"]:
            roles.add(reference["role"])
            path = ROOT / reference["path"]
            payload = path.read_bytes()
            self.assertEqual(len(payload), reference["bytes"], reference["path"])
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(), reference["sha256"], reference["path"]
            )
        self.assertEqual(len(roles), 31)
        self.assertEqual(len(roles), len(self.document["references"]))

    def test_frozen_values_and_statuses(self) -> None:
        model = self.document["model_identity"]
        self.assertEqual(model["revision"], "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
        self.assertEqual(model["canonical_root_manifest_sha256"], "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2")
        correctness = self.document["correctness_baseline"]
        self.assertEqual(correctness["tier_a"]["generated_token_ids"], [1096, 374])
        performance = self.document["performance_baseline"]
        self.assertEqual(performance["total_logical_bytes"], 73004834816)
        self.assertEqual(performance["expert_cache"]["misses"], performance["expert_cache"]["loads"])
        statuses = {
            item["item"]: item["status"]
            for item in self.document["quantization_decision_registry"]["statuses"]
        }
        self.assertEqual(statuses["f32_baseline"], "authoritative_and_accepted")
        self.assertEqual(statuses["int8_per_output_channel"], "rejected_full_model_candidate")
        self.assertNotIn("accepted_for_runtime_prototype", statuses.values())

    def test_generation_is_byte_identical(self) -> None:
        expected = BASELINE_PATH.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "baseline.json"
            first = canonical_bytes(build(ROOT))
            output.write_bytes(first)
            second = canonical_bytes(build(ROOT))
            self.assertEqual(first, second)
            self.assertEqual(first, expected)
            self.assertEqual(hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest())


if __name__ == "__main__":
    unittest.main()
