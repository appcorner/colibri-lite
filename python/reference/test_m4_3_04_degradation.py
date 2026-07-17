"""Regression checks for the M4.3-04 INT8 degradation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/qwen3-30b-a3b"


def read(name: str) -> dict:
    return json.loads((MODEL / name).read_text(encoding="utf-8"))


class DegradationEvidenceTests(unittest.TestCase):
    def test_evidence_hash_and_staged_counts(self) -> None:
        path = MODEL / "m4.3-04-degradation-evidence-v1.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "0a2f5c85087de32a23b975bc206ed98b007e353dbc897fb71317fcef6568e140",
        )
        evidence = read(path.name)
        self.assertEqual(evidence["status"], "tier_a_complete")
        self.assertEqual(evidence["candidate_classification"], "quality_risk")
        self.assertEqual(len(evidence["tier_c"]["cases"]), 8)
        self.assertEqual(len(evidence["tier_b"]), 6)
        self.assertEqual(evidence["tier_a"]["baseline_generated_ids"], [1096, 374])
        self.assertEqual(evidence["tier_a"]["int8_generated_ids"], [1096, 374])

    def test_local_gates_pass_but_tier_b_quality_risk_is_frozen(self) -> None:
        evidence = read("m4.3-04-degradation-evidence-v1.json")
        self.assertTrue(evidence["tier_c"]["all_gates_pass"])
        thai = next(item for item in evidence["tier_b"] if item["fixture"]["name"] == "short_thai")
        self.assertEqual(thai["final"]["classification"], "numerically_ambiguous")
        self.assertNotEqual(thai["final"]["baseline"]["argmax_token_id"], thai["final"]["candidate"]["argmax_token_id"])
        self.assertEqual(evidence["first_failure"], None)
        self.assertEqual(evidence["first_quality_risk"]["fixture"], "single_low_token")

    def test_provisional_gates_do_not_modify_f32_contract(self) -> None:
        gates = read("m4.3-04-provisional-degradation-gates-v1.json")
        self.assertEqual(gates["decision"], "quality_risk")
        self.assertTrue(gates["f32_registry_unchanged"])
        self.assertEqual(gates["semantic_observations"]["tier_a_candidate_generated_ids"], [1096, 374])


if __name__ == "__main__":
    unittest.main()
