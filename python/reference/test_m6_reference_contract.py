"""Failure-mode and repeatability tests for the M6 F32 reference contract."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

from python.reference.m6_reference_contract import (
    QUALITY_PATH,
    REFERENCE_PATH,
    ReferenceContractError,
    load_json,
    validate_reference_contract,
)


ROOT = Path(__file__).resolve().parents[2]


class M6ReferenceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = load_json(ROOT / REFERENCE_PATH)
        self.quality = load_json(ROOT / QUALITY_PATH)

    def test_contract_is_repeatable(self) -> None:
        first = validate_reference_contract(ROOT, self.reference, self.quality)
        second = validate_reference_contract(ROOT, self.reference, self.quality)
        self.assertEqual(first, second)
        self.assertEqual(first["fixture_ids"], ["short_english", "short_thai"])

    def test_rejects_wrong_reference_identity(self) -> None:
        quality = copy.deepcopy(self.quality)
        quality["reference_id"] = "other-reference"
        with self.assertRaisesRegex(ReferenceContractError, "reference ID"):
            validate_reference_contract(ROOT, self.reference, quality)

    def test_rejects_corrupted_integrity_record(self) -> None:
        quality = copy.deepcopy(self.quality)
        quality["integrity_records"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ReferenceContractError, "SHA-256 mismatch"):
            validate_reference_contract(ROOT, self.reference, quality)

    def test_rejects_changed_expected_output(self) -> None:
        quality = copy.deepcopy(self.quality)
        quality["fixtures"][1]["expected"]["logits"]["argmax_token_id"] = 0
        with self.assertRaisesRegex(ReferenceContractError, "argmax_token_id mismatch"):
            validate_reference_contract(ROOT, self.reference, quality)

    def test_rejects_changed_finite_counts(self) -> None:
        quality = copy.deepcopy(self.quality)
        quality["fixtures"][0]["expected"]["logits"]["finite_counts"]["nan"] = 1
        with self.assertRaisesRegex(ReferenceContractError, "finite counts mismatch"):
            validate_reference_contract(ROOT, self.reference, quality)


if __name__ == "__main__":
    unittest.main()
