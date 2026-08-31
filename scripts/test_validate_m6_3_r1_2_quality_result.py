from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from validate_m6_3_r1_2_quality_result import validate

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "models/qwen3-30b-a3b/m6.3-r1-2-quality-result-v1.json"


class QualityResultValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = json.loads(RECORD.read_text(encoding="utf-8"))

    def test_committed_result_passes(self) -> None:
        self.assertEqual(validate(self.record, ROOT), [])

    def test_wrong_candidate_fails(self) -> None:
        record = copy.deepcopy(self.record)
        record["candidate"]["candidate_id"] = "wrong"
        self.assertTrue(validate(record))

    def test_changed_sequence_fails(self) -> None:
        record = copy.deepcopy(self.record)
        record["fixtures"][0]["generated_token_ids"] = [0, 0]
        self.assertTrue(validate(record))

    def test_top20_failure_fails(self) -> None:
        record = copy.deepcopy(self.record)
        record["fixtures"][0]["prompt_top20_exact"] = False
        self.assertTrue(validate(record))

    def test_logit_envelope_failure_fails(self) -> None:
        record = copy.deepcopy(self.record)
        record["fixtures"][1]["prompt_top20_logit_max_abs"] = 0.06
        self.assertTrue(validate(record))

    def test_repeatability_failure_fails(self) -> None:
        record = copy.deepcopy(self.record)
        record["fixtures"][1]["repeatability"] = "mismatch"
        self.assertTrue(validate(record))

    def test_m6_4_authorization_fails(self) -> None:
        record = copy.deepcopy(self.record)
        record["decision"]["authorizes_m6_4"] = True
        self.assertTrue(validate(record))


if __name__ == "__main__":
    unittest.main()
