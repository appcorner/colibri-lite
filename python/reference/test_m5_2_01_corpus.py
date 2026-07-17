"""Regression and compatibility checks for the M5.2-01 trace corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts import analyze_m5_2_01_corpus as analyzer
from scripts import simulate_m5_1_memory_hierarchy as simulator


ROOT = Path("models/qwen3-30b-a3b")
FIXTURE_MANIFEST = ROOT / "m5.2-01-representative-fixture-manifest-v1.json"
CORPUS_MANIFEST = ROOT / "m5.2-01-trace-corpus-manifest-v1.json"
REPEATABILITY = ROOT / "m5.2-01-repeatability-v1.json"
AGGREGATE = ROOT / "m5.2-01-trace-corpus-aggregate-v1.json"
SCHEMA = ROOT / "m5.2-01-ordered-expert-trace-schema-v2.json"
DISCREPANCY = ROOT / "m5.2-01-m5.1-03-counter-discrepancy-v1.json"
CONTROL = ROOT / "m5.1-00-ordered-expert-trace-v1.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M52RepresentativeCorpusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
        self.corpus_manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
        self.repeatability = json.loads(REPEATABILITY.read_text(encoding="utf-8"))
        self.aggregate = json.loads(AGGREGATE.read_text(encoding="utf-8"))

    def test_manifest_has_required_diverse_fixtures(self) -> None:
        fixtures = self.fixture_manifest["fixtures"]
        self.assertEqual(len(fixtures), 8)
        classes = {fixture["workload_class"] for fixture in fixtures}
        self.assertTrue({
            "frozen_tier_a_control",
            "english_natural_language_prompt",
            "thai_natural_language_prompt",
            "source_code_prompt",
            "repeated_or_redundant_text",
            "special_token_or_formatting_heavy_input",
            "longer_context_english",
            "longer_decode_sequence",
        } <= classes)
        self.assertTrue(any(f["requested_generation_length"] > 2 for f in fixtures))
        for fixture in fixtures:
            self.assertEqual(fixture["input_length"], len(fixture["token_ids"]))
            self.assertEqual(fixture["decoding_mode"], "greedy")
            self.assertEqual(fixture["seed"], 0)

    def test_hashes_records_and_expected_outputs(self) -> None:
        fixtures = {fixture["fixture_id"]: fixture for fixture in self.fixture_manifest["fixtures"]}
        for entry in self.corpus_manifest["fixtures"]:
            fixture = fixtures[entry["fixture_id"]]
            trace_path = Path(entry["trace_path"])
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            is_control = entry["fixture_id"] == "tier_a_control"
            analyzer.validate_records(trace, fixture, is_control)
            self.assertEqual(file_sha256(trace_path), entry["trace_sha256"])
            self.assertEqual(entry["repeat_sha256"], [entry["trace_sha256"]] * 2)
            self.assertEqual(trace["expected_generated_token_ids"], fixture["expected_generated_token_ids"])
            if not is_control:
                self.assertEqual(trace["trace_instrumentation_commit"], "a650acc")
            self.assertNotIn("timestamp", trace)
            self.assertNotIn("process_id", trace)
            self.assertNotIn("D:\\", trace_path.read_text(encoding="utf-8"))

    def test_repeats_and_aggregate_are_closed(self) -> None:
        baseline = self.corpus_manifest["baseline"]
        self.assertEqual(
            baseline["m4_performance_baseline_sha256"],
            file_sha256(ROOT / "m4.4-performance-baseline-v1.json"),
        )
        self.assertEqual(
            baseline["m5_1_00_control_trace_sha256"],
            file_sha256(CONTROL),
        )
        self.assertEqual(
            baseline["m5_1_03_results_sha256"],
            file_sha256(ROOT / "m5.1-03-full-model-cache-results-v1.json"),
        )
        self.assertEqual(len(self.repeatability["fixtures"]), 8)
        self.assertTrue(all(entry["repeat_count"] == 2 for entry in self.repeatability["fixtures"]))
        self.assertTrue(all(entry["byte_identical_repeat"] for entry in self.repeatability["fixtures"]))
        self.assertEqual(self.aggregate["simulation_executed"], False)
        self.assertEqual(self.aggregate["eight_gib_recommendation_classification"]["classification"], "inconclusive")
        self.assertFalse(self.aggregate["eight_gib_recommendation_classification"]["cache_policy_recommendation_made"])
        self.assertEqual(self.aggregate["corpus_wide"]["total_expert_occurrences"], 11520)
        self.assertEqual(self.aggregate["corpus_wide"]["unique_layer_expert_keys"], 3148)

    def test_existing_simulator_record_adapter_accepts_every_new_trace(self) -> None:
        """Check the simulator's record-key contract without invoking simulation."""
        for entry in self.corpus_manifest["fixtures"]:
            if entry["fixture_id"] == "tier_a_control":
                continue
            trace = json.loads(Path(entry["trace_path"]).read_text(encoding="utf-8"))
            keys = [simulator.key_of(record) for record in trace["records"]]
            self.assertEqual(len(keys), entry["record_count"])
            self.assertTrue(all(key.startswith("layer.") for key in keys))

    def test_control_and_discrepancy_evidence_are_pinned(self) -> None:
        control_entry = self.corpus_manifest["fixtures"][0]
        self.assertEqual(control_entry["trace_path"], CONTROL.as_posix())
        self.assertEqual(file_sha256(CONTROL), control_entry["trace_sha256"])
        discrepancy = json.loads(DISCREPANCY.read_text(encoding="utf-8"))
        self.assertEqual(discrepancy["status"], "resolved_without_runtime_change")
        self.assertEqual(discrepancy["diagnosis"]["confirmed_cause"], "payload_budget_accounting")
        self.assertFalse(discrepancy["diagnosis"]["runtime_behavior_defect"])


if __name__ == "__main__":
    unittest.main()
