import json
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models" / "qwen3-30b-a3b"


class M43DecisionTests(unittest.TestCase):
    def setUp(self):
        self.registry_path = MODEL_ROOT / "m4.3-06-candidate-status-registry-v1.json"
        self.summary_path = MODEL_ROOT / "m4.3-06-decision-summary-v1.json"
        self.registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.summary = json.loads(self.summary_path.read_text(encoding="utf-8"))

    def test_selected_candidate_is_not_runtime_accepted(self):
        statuses = set(self.registry["candidate"]["statuses"])
        self.assertIn("rejected_full_model_candidate", statuses)
        self.assertIn("retained_for_diagnostics", statuses)
        self.assertNotIn("accepted_for_runtime_prototype", statuses)
        self.assertEqual(self.summary["investment_decision"]["immediate_production_int8"], False)

    def test_phase_verdict_covers_required_items(self):
        items = {entry["item"]: entry["status"] for entry in self.registry["phase_verdict"]}
        self.assertEqual(items["f32_baseline"], "authoritative_and_accepted")
        self.assertEqual(items["int8_per_tensor"], "rejected")
        self.assertEqual(items["int8_per_output_channel"], "rejected_full_model_candidate")
        self.assertEqual(items["int8_group_128"], "promising_but_insufficient_evidence")
        self.assertEqual(items["router_int8"], "rejected")
        self.assertEqual(items["ik_llama_q4_k_m"], "external_performance_reference_only")

    def test_evidence_references_exist(self):
        for evidence in self.registry["evidence"]:
            path = ROOT / evidence["path"]
            self.assertTrue(path.is_file(), evidence["path"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), evidence["sha256"], evidence["path"]
            )

    def test_roadmap_has_all_budget_points_and_frozen_ids(self):
        roadmap = (ROOT / "docs" / "m4.3-next-phase-memory-hierarchy-roadmap.md").read_text(encoding="utf-8")
        for budget in ("1", "2", "4", "8", "16", "24", "32"):
            self.assertIn(budget, roadmap)
        self.assertIn("[1096, 374]", json.dumps(self.summary))
        self.assertTrue(self.summary["next_phase"]["simulation_only_first"])


if __name__ == "__main__":
    unittest.main()
