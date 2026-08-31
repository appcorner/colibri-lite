import copy
import json
import unittest
from pathlib import Path

from validate_m6_3_r1_1d_admission_amendment import GROUP32_ID, validate

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "models/qwen3-30b-a3b/m6.3-r1-1d-admission-amendment-v1.json"


class AdmissionAmendmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = json.loads(RECORD.read_text(encoding="utf-8"))

    def test_canonical_record_passes(self) -> None:
        self.assertEqual(validate(self.record, ROOT), [])

    def test_historical_outcome_cannot_be_rewritten(self) -> None:
        candidate = copy.deepcopy(self.record)
        candidate["historical_admission"]["outcome"] = "exactly_one_candidate"
        self.assertIn("historical outcome changed", validate(candidate, ROOT))

    def test_candidate_identity_is_frozen(self) -> None:
        candidate = copy.deepcopy(self.record)
        candidate["candidate"]["candidate_id"] = "other"
        self.assertIn("wrong admitted candidate", validate(candidate, ROOT))

    def test_envelope_cannot_be_loosened_above_preregistered_cap(self) -> None:
        candidate = copy.deepcopy(self.record)
        candidate["candidate"]["admission_logit_max_abs_envelope"] = 0.051
        self.assertIn("invalid admission logit envelope", validate(candidate, ROOT))

    def test_r1_3_and_m6_4_remain_blocked(self) -> None:
        candidate = copy.deepcopy(self.record)
        candidate["decision"]["authorizes_r1_3"] = True
        candidate["decision"]["authorizes_m6_4"] = True
        errors = validate(candidate, ROOT)
        self.assertIn("R1.3 must remain blocked", errors)
        self.assertIn("M6.4 must remain blocked", errors)

    def test_r1_2_cannot_reselect_candidate(self) -> None:
        candidate = copy.deepcopy(self.record)
        candidate["r1_2_gate"]["must_not_reselect_candidate"] = False
        self.assertIn("R1.2 must not reselect the candidate", validate(candidate, ROOT))

    def test_selected_candidate_is_group32(self) -> None:
        self.assertEqual(self.record["decision"]["selected_candidate_id"], GROUP32_ID)


if __name__ == "__main__":
    unittest.main()
