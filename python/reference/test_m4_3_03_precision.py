"""Contract checks for the M4.3-03 precision-sensitivity diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/qwen3-30b-a3b"


def read(name: str) -> dict:
    return json.loads((MODEL / name).read_text(encoding="utf-8"))


class PrecisionSensitivityTests(unittest.TestCase):
    def test_evidence_is_deterministic_and_complete(self) -> None:
        path = MODEL / "m4.3-03-precision-sensitivity-evidence-v1.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "1387addd232a80e970af00d7c86dc1a747085589fff14663b2f909ab3b38db81",
        )
        evidence = read(path.name)
        self.assertEqual(evidence["status"], "precision_sensitivity_diagnostic_complete")
        self.assertEqual(set(evidence["groups"]), {
            "embedding_weights", "attention_q_projection", "attention_k_projection",
            "attention_v_projection", "attention_o_projection", "input_rmsnorm_weights",
            "post_attention_rmsnorm_weights", "q_norm_weights", "k_norm_weights",
            "router_weights", "final_rmsnorm_weights", "lm_head_weights",
        })
        self.assertEqual(len(evidence["records"]), 117)
        self.assertEqual(len(evidence["router_records"]), 12)
        self.assertTrue(all(item["canonical_f32_unchanged"] for item in [evidence["model"]]))

    def test_router_int8_safe_margin_failure_is_preserved(self) -> None:
        evidence = read("m4.3-03-precision-sensitivity-evidence-v1.json")
        layer0 = [item for item in evidence["router_records"] if item["layer"] == 0 and item["variant"] == "int8_per_output_channel"][0]
        self.assertEqual(layer0["classification"], "true_mismatch")
        self.assertNotEqual(layer0["f32_ids"], layer0["candidate_ids"])
        self.assertGreater(layer0["boundary_margin"], 0.0)

    def test_policy_keeps_sensitive_groups_f32(self) -> None:
        registry = read("m4.3-03-tensor-precision-registry-v1.json")
        groups = {item["group"]: item for item in registry["groups"]}
        for name in ("router_weights", "input_rmsnorm_weights", "post_attention_rmsnorm_weights", "q_norm_weights", "k_norm_weights", "final_rmsnorm_weights"):
            self.assertEqual(groups[name]["classification"], "must_remain_f32")
        self.assertEqual(groups["router_weights"]["int8_status"], "rejected_lower_precision")

    def test_policy_and_evidence_are_diagnostic_only(self) -> None:
        policy = read("m4.3-03-mixed-precision-policy-v1.json")
        self.assertEqual(policy["status"], "policy_draft_no_runtime_change")
        self.assertEqual(policy["baseline"]["activation_dtype"], "F32")
        self.assertEqual(policy["baseline"]["accumulation_dtype"], "F32")
        self.assertIn("Tier A", policy["acceptance_rule"])


if __name__ == "__main__":
    unittest.main()
