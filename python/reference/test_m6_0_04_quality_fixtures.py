"""Integrity and source-consistency tests for frozen M6 bilingual fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models/qwen3-30b-a3b"
MANIFEST = MODEL_ROOT / "m6.0-04-quality-fixtures-v1.json"
SOURCE = MODEL_ROOT / "m4.3-01-tier-b-transformers-f32-v1.json"


class M60QualityFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        self.source_by_name = {fixture["name"]: fixture for fixture in source["fixtures"]}

    def test_integrity_records_match_frozen_inputs(self) -> None:
        for record in self.manifest["integrity_records"]:
            path = ROOT / record["path"]
            self.assertTrue(path.is_file(), record["path"])
            self.assertEqual(path.stat().st_size, record["bytes"], record["path"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"], record["path"]
            )

    def test_english_and_thai_expected_outputs_match_reference(self) -> None:
        fixtures = self.manifest["fixtures"]
        self.assertEqual([fixture["fixture_id"] for fixture in fixtures], ["short_english", "short_thai"])
        self.assertEqual([fixture["input"]["text"] for fixture in fixtures], ["Hello world", "ไทย"])
        self.assertEqual([fixture["input"]["token_ids"] for fixture in fixtures], [[9707, 1879], [125451]])

        for fixture in fixtures:
            source = self.source_by_name[fixture["fixture_id"]]
            expected = fixture["expected"]
            source_logits = source["logits"]
            expected_logits = expected["logits"]
            self.assertEqual(fixture["input"], {"text": source["text"], "token_ids": source["token_ids"]})
            self.assertEqual(fixture["final_position"], source["final_position"])
            self.assertEqual(expected["final_norm_sha256_f32_le"], source["final_norm"]["sha256_f32_le"])
            self.assertEqual(expected["guard_router_ids"], source["guard_router_ids"])
            for field in ("argmax_logit", "argmax_token_id", "fixed_indices", "fixed_logits", "top1_margin", "top20_token_ids", "vocabulary_size"):
                self.assertEqual(expected_logits[field], source_logits[field], field)
            self.assertEqual(expected_logits["finite_counts"]["nan"], source_logits["nan_count"])
            self.assertEqual(expected_logits["finite_counts"]["positive_infinity"], source_logits["positive_infinity_count"])
            self.assertEqual(expected_logits["finite_counts"]["negative_infinity"], source_logits["negative_infinity_count"])

    def test_fixture_policy_preserves_existing_scope(self) -> None:
        self.assertEqual(self.manifest["reference_id"], "reference-f32-v1")
        self.assertEqual(self.manifest["status"], "frozen")
        self.assertIn("M6.3", self.manifest["comparison_policy"]["top_k_policy"])
        self.assertIn("semantic-margin", self.manifest["comparison_policy"]["router_policy"])


if __name__ == "__main__":
    unittest.main()
