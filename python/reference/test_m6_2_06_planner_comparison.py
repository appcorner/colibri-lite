"""Validate the tracked M6.2 planner prediction-error evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "docs/benchmarks/m6.2-06-planner-comparison-v1.json"


class M62PlannerComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(COMPARISON.read_text(encoding="utf-8"))

    def test_error_is_recomputed_from_tracked_prediction_and_observation(self) -> None:
        planner = self.document["planner_input"]
        benchmark = self.document["recorded_benchmark"]
        expected = (planner["predicted_decode_tokens_per_second"] - benchmark["observed_decode_tokens_per_second"]) / benchmark["observed_decode_tokens_per_second"]
        self.assertAlmostEqual(self.document["error"]["relative_to_observed"], expected)
        self.assertAlmostEqual(self.document["error"]["percent_relative_to_observed"], expected * 100.0)
        self.assertFalse(self.document["decision"]["cost_model_retuned"])
        self.assertFalse(self.document["decision"]["prediction_accepted_for_promotion"])

    def test_referenced_inputs_and_historical_evidence_hashes_are_pinned(self) -> None:
        planner = self.document["planner_input"]
        for profile_name in ("hardware_profile", "model_profile"):
            profile = planner[profile_name]
            payload = (ROOT / profile["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), profile["sha256"])
        benchmark = self.document["recorded_benchmark"]
        for path_key, hash_key in (("runtime_path", "runtime_sha256"), ("profile_path", "profile_sha256")):
            self.assertEqual(hashlib.sha256((ROOT / benchmark[path_key]).read_bytes()).hexdigest(), benchmark[hash_key])

    def test_compute_input_is_the_recorded_per_decode_flop_sum(self) -> None:
        planner = self.document["planner_input"]
        benchmark = self.document["recorded_benchmark"]
        profile = json.loads((ROOT / benchmark["profile_path"]).read_text(encoding="utf-8"))
        totals: dict[str, int] = {}
        for event in profile["events"]:
            phase = event["phase"]
            if phase.startswith("decode_"):
                totals[phase] = totals.get(phase, 0) + event["estimated_flops"]
        self.assertEqual(set(totals), {"decode_1", "decode_2", "decode_3"})
        self.assertEqual(set(totals.values()), {5_460_983_808})
        self.assertEqual(planner["compute_gflop_per_token"]["value"], 5.460983808)

    def test_record_is_explicitly_non_promotable_when_contracts_do_not_match(self) -> None:
        self.assertEqual(self.document["schema"], "colibri-lite-m6.2-06-planner-comparison-v1")
        self.assertEqual(self.document["comparability"]["status"], "partial_not_promotable")
        self.assertGreaterEqual(len(self.document["comparability"]["mismatches"]), 3)


if __name__ == "__main__":
    unittest.main()
