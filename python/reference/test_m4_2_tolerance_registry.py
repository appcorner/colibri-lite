"""Consistency and evidence-reference tests for the M4.2 contract registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
REGISTRY_PATH = (
    REPOSITORY
    / "models"
    / "qwen3-30b-a3b"
    / "m4.2-tolerance-contract-registry-v1.json"
)
REQUIRED_CHECKPOINTS = {
    "tensor_conversion",
    "embedding",
    "rmsnorm_internal",
    "rmsnorm_cross_runtime",
    "attention_output",
    "residual_output",
    "post_attention_rmsnorm",
    "router_logits",
    "routing_weights",
    "expert_input",
    "expert_gate_projection",
    "expert_up_projection",
    "expert_activation",
    "expert_activated_product",
    "expert_down_projection",
    "weighted_expert_output",
    "aggregated_moe_output",
    "moe_residual_addition",
    "final_block_output",
    "final_rmsnorm",
    "lm_head_logits",
    "cached_vs_recomputed_logits",
    "router_top_k_margin",
    "vocabulary_top_1_margin",
}
STATUSES = {
    "frozen",
    "provisional",
    "layer_specific",
    "fixture_specific",
    "diagnostic_only",
    "semantic_margin",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class M42ToleranceRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    def test_required_contracts_are_unique_and_scoped(self) -> None:
        self.assertEqual(self.registry["schema_version"], 1)
        checkpoints = self.registry["checkpoints"]
        identifiers = [checkpoint["id"] for checkpoint in checkpoints]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(identifiers), REQUIRED_CHECKPOINTS)

        for checkpoint in checkpoints:
            with self.subTest(checkpoint=checkpoint["id"]):
                self.assertIn(checkpoint["contract_status"], STATUSES)
                self.assertTrue(checkpoint["comparison_paths"])
                self.assertTrue(checkpoint["budget_formula"])
                self.assertTrue(checkpoint["stop_condition"])
                self.assertTrue(checkpoint["supporting_documents"])
                for document in checkpoint["supporting_documents"]:
                    self.assertTrue((REPOSITORY / document).is_file(), document)

                mode = checkpoint["comparison_mode"]
                self.assertIn(mode, {"exact", "tolerant", "semantic_margin"})
                if mode == "exact":
                    self.assertEqual(checkpoint["absolute_tolerance"], 0.0)
                    self.assertEqual(checkpoint["relative_tolerance"], 0.0)
                    self.assertEqual(checkpoint["ulp_guard"], 0)
                elif mode == "tolerant":
                    self.assertNotEqual(checkpoint["contract_status"], "frozen")

    def test_provisional_guards_are_not_promoted(self) -> None:
        checkpoints = {
            checkpoint["id"]: checkpoint for checkpoint in self.registry["checkpoints"]
        }
        rmsnorm = checkpoints["rmsnorm_cross_runtime"]
        self.assertEqual(rmsnorm["contract_status"], "provisional")
        self.assertEqual(rmsnorm["absolute_tolerance"], 5e-7)
        self.assertEqual(rmsnorm["ulp_guard"], 8)

        cached = checkpoints["cached_vs_recomputed_logits"]
        self.assertEqual(cached["contract_status"], "diagnostic_only")
        self.assertIsNone(cached["absolute_tolerance"])

        for identifier in (
            "attention_output",
            "residual_output",
            "post_attention_rmsnorm",
            "router_logits",
            "routing_weights",
        ):
            self.assertEqual(checkpoints[identifier]["contract_status"], "layer_specific")

    def test_frozen_fixture_and_margin_invariants_are_present(self) -> None:
        self.assertEqual(self.registry["fixture"]["generated_token_ids"], [1096, 374])
        self.assertEqual(self.registry["global_rules"]["nan_and_infinity"]["allowed_count"], 0)
        checkpoints = {
            checkpoint["id"]: checkpoint for checkpoint in self.registry["checkpoints"]
        }
        self.assertIn(
            "2 * measured per-token router-logit error",
            checkpoints["router_top_k_margin"]["budget_formula"],
        )
        self.assertIn(
            "2 * measured all-logit error",
            checkpoints["vocabulary_top_1_margin"]["budget_formula"],
        )
        self.assertEqual(checkpoints["expert_down_projection"]["internal_trace_ulp_guard"], 0)

    def test_every_evidence_reference_exists_and_matches_hash(self) -> None:
        evidence = self.registry["evidence"]
        self.assertGreaterEqual(len(evidence), 12)
        paths = [record["path"] for record in evidence]
        self.assertEqual(len(paths), len(set(paths)))
        for record in evidence:
            with self.subTest(path=record["path"]):
                path = REPOSITORY / record["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(sha256_file(path), record["sha256"])


if __name__ == "__main__":
    unittest.main()
