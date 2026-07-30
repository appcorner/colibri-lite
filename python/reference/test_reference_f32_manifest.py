"""Integrity tests for the frozen M6 `reference-f32-v1` manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "models/qwen3-30b-a3b/reference-f32-v1-manifest.json"


class ReferenceF32ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_reference_identity_and_f32_scope_are_frozen(self) -> None:
        self.assertEqual(self.document["reference_id"], "reference-f32-v1")
        self.assertEqual(
            self.document["schema"], "colibri-qwen3-moe-reference-f32-manifest-v1"
        )
        self.assertEqual(self.document["artifact"]["storage_dtype"], "F32")
        self.assertEqual(
            self.document["freeze_policy"]["reference_execution"],
            "safe_scalar_ordered_F32",
        )
        self.assertEqual(self.document["freeze_policy"]["status"], "frozen")
        self.assertTrue(self.document["freeze_policy"]["no_model_payload_is_copied"])

    def test_all_integrity_records_match_existing_inputs(self) -> None:
        roles = set()
        for record in self.document["integrity_records"]:
            self.assertNotIn(record["role"], roles)
            roles.add(record["role"])
            path = ROOT / record["path"]
            self.assertTrue(path.is_file(), record["path"])
            self.assertEqual(path.stat().st_size, record["bytes"], record["path"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["sha256"],
                record["path"],
            )

    def test_fixtures_include_authoritative_and_bilingual_coverage(self) -> None:
        fixture_contract = self.document["fixture_contract"]
        self.assertEqual(
            fixture_contract["authoritative_generation"]["input_token_ids"],
            [9707, 11, 1879, 0],
        )
        self.assertEqual(
            fixture_contract["authoritative_generation"]["generated_token_ids"],
            [1096, 374],
        )
        self.assertEqual(fixture_contract["tier_b_fixture_count"], 6)
        self.assertEqual(fixture_contract["coverage"], ["English", "Thai", "prefill", "cached_decode"])


if __name__ == "__main__":
    unittest.main()
