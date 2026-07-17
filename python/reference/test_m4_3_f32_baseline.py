"""Consistency tests for the frozen M4.3-01 F32 baseline bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from python.reference.build_f32_baseline_bundle import (
    MODEL_DIR,
    OUTPUT_NAMES,
    build_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / MODEL_DIR


class F32BaselineBundleTests(unittest.TestCase):
    def test_bundle_rebuild_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = build_bundle(ROOT, output)
            first_bytes = {name: (output / name).read_bytes() for name in first}
            second = build_bundle(ROOT, output)
            self.assertEqual(first, second)
            for name, payload in first_bytes.items():
                self.assertEqual((output / name).read_bytes(), payload)
                self.assertEqual((MODEL_ROOT / name).read_bytes(), payload)

    def test_manifest_references_exist_and_match_hashes(self) -> None:
        manifest = json.loads(
            (MODEL_ROOT / OUTPUT_NAMES["manifest"]).read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["status"], "authoritative_unquantized_f32_baseline_frozen"
        )
        self.assertEqual(
            manifest["model"]["canonical_root_manifest_sha256"],
            "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2",
        )
        for reference in manifest["supporting_evidence"]:
            path = ROOT / reference["path"]
            payload = path.read_bytes()
            self.assertEqual(len(payload), reference["bytes"], reference["path"])
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(),
                reference["sha256"],
                reference["path"],
            )
        for documentation in manifest["documentation"]:
            self.assertTrue((ROOT / documentation).is_file(), documentation)

    def test_fixture_hierarchy_has_required_coverage(self) -> None:
        fixtures = json.loads(
            (MODEL_ROOT / OUTPUT_NAMES["fixtures"]).read_text(encoding="utf-8")
        )
        tier_a = fixtures["tiers"]["A"]
        self.assertEqual(tier_a["input_token_ids"], [9707, 11, 1879, 0])
        self.assertEqual(tier_a["generated_token_ids"], [1096, 374])
        tier_b = fixtures["tiers"]["B"]
        self.assertEqual(tier_b["fixture_count"], 6)
        self.assertEqual(tier_b["processed_positions"], 11)
        names = {fixture["name"] for fixture in tier_b["fixtures"]}
        self.assertEqual(
            names,
            {
                "single_low_token",
                "short_english",
                "short_thai",
                "code_newline",
                "repeated_pattern",
                "special_token",
            },
        )
        all_tokens = [
            token
            for fixture in tier_b["fixtures"]
            for token in fixture["token_ids"]
        ]
        self.assertIn(0, all_tokens)
        self.assertIn(151643, all_tokens)
        self.assertTrue(all(len(fixture["guard_router_ids"]) == 3 for fixture in tier_b["fixtures"]))
        operations = {
            fixture["operation"] for fixture in fixtures["tiers"]["C"]["fixtures"]
        }
        self.assertEqual(
            operations,
            {
                "embedding",
                "rmsnorm",
                "attention",
                "router",
                "selected_expert_mlp",
                "final_norm_and_lm_head",
                "kv_cache_update",
            },
        )

    def test_tier_b_fixture_budgets_and_semantics_pass(self) -> None:
        with (MODEL_ROOT / OUTPUT_NAMES["rust_evidence"]).open(
            "r", encoding="ascii", newline=""
        ) as source:
            records = list(csv.DictReader(source, delimiter="\t"))
        self.assertEqual(len(records), 6)
        for record in records:
            norm_error = float(record["maximum_fixed_final_norm_error"])
            norm_budget = float(record["final_norm_fixture_budget"])
            logit_error = max(
                float(record["maximum_fixed_logit_error"]),
                float(record["maximum_top20_logit_error"]),
            )
            logit_budget = float(record["logit_fixture_budget"])
            margin = float(record["top1_margin"])
            required = float(record["required_safe_margin"])
            self.assertLessEqual(norm_error, norm_budget, record["fixture"])
            self.assertLessEqual(logit_error, logit_budget, record["fixture"])
            self.assertEqual(required, 2.0 * logit_error, record["fixture"])
            self.assertGreater(margin, required, record["fixture"])
            self.assertEqual(record["classification"], "exact_match_safe_compact")
            loads = int(record["expert_loads"])
            self.assertEqual(int(record["expert_evictions"]), loads - 1)

    def test_f64_and_future_comparison_scopes_are_explicit(self) -> None:
        diagnostics = json.loads(
            (MODEL_ROOT / OUTPUT_NAMES["f64"]).read_text(encoding="utf-8")
        )
        self.assertFalse(diagnostics["complete_model_f64_executed"])
        self.assertEqual(len(diagnostics["records"]), 6)
        self.assertTrue(
            all(record["status"] == "diagnostic_only" for record in diagnostics["records"])
        )
        self.assertTrue(
            all(record["contract_impact"] == "none" for record in diagnostics["records"])
        )
        comparison = json.loads(
            (MODEL_ROOT / OUTPUT_NAMES["comparison"]).read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(comparison["classifications"]),
            {
                "exact-equivalent",
                "numerically equivalent within contract",
                "semantically equivalent",
                "quality-risk",
                "correctness failure",
            },
        )
        self.assertEqual(len(comparison["mandatory_deltas"]), 7)
        self.assertEqual(
            comparison["acceptance_rule"],
            "performance_gain_alone_is_never_sufficient",
        )


if __name__ == "__main__":
    unittest.main()
