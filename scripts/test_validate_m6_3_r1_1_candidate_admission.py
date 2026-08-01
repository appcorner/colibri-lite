#!/usr/bin/env python3
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_m6_3_r1_1_candidate_admission import validate

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "models/qwen3-30b-a3b/m6.3-r1-1-candidate-admission-v1.json"


class CandidateAdmissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(RECORD.read_text(encoding="utf-8"))

    def test_no_candidate_admitted_is_valid(self):
        self.assertEqual(validate(self.record), [])

    def test_missing_provenance_is_rejected(self):
        record = copy.deepcopy(self.record)
        record["reference"]["license"] = ""
        self.assertTrue(validate(record))

    def test_post_hoc_threshold_is_rejected(self):
        record = copy.deepcopy(self.record)
        record["numerical_gates"]["registered_before_final_fixture_execution"] = False
        self.assertTrue(validate(record))

    def test_whole_expert_expansion_is_rejected(self):
        record = copy.deepcopy(self.record)
        record["candidates"][0]["direct_consumption"]["complete_f32_expert_materializations"] = 1
        self.assertTrue(validate(record))

    def test_malformed_hash_is_rejected(self):
        record = copy.deepcopy(self.record)
        record["candidates"][0]["artifact"]["source_sha256"] = "bad"
        self.assertTrue(validate(record))

    def test_malformed_offsets_and_shapes_are_rejected_for_admission(self):
        record = copy.deepcopy(self.record)
        candidate = record["candidates"][0]
        candidate["status"] = "admitted"
        candidate["conversion"].update(status="passed", byte_identical_reconversion=True)
        candidate["artifact"].update(output_sha256="a" * 64, offsets_valid=False)
        candidate["repeated_execution"]["byte_identical"] = True
        candidate["characterization"]["proposed_logit_max_abs_envelope"] = 0.01
        candidate["layout"]["shapes"] = ["bad"]
        record["selection"].update(outcome="exactly_one_candidate", selected_candidate_id=candidate["candidate_id"])
        self.assertTrue(validate(record))

    def test_more_than_one_selection_is_rejected(self):
        record = copy.deepcopy(self.record)
        record["selection"]["outcome"] = "exactly_one_candidate"
        record["candidates"][0]["status"] = "admitted"
        record["candidates"][1]["status"] = "admitted"
        for candidate in record["candidates"]:
            candidate["conversion"].update(status="passed", byte_identical_reconversion=True)
            candidate["artifact"].update(output_sha256="a" * 64, offsets_valid=True)
            candidate["repeated_execution"]["byte_identical"] = True
            candidate["characterization"]["proposed_logit_max_abs_envelope"] = 0.01
        record["selection"]["selected_candidate_id"] = record["candidates"][0]["candidate_id"]
        self.assertTrue(validate(record))

    def test_nondeterministic_conversion_is_rejected_for_admission(self):
        record = copy.deepcopy(self.record)
        candidate = record["candidates"][0]
        candidate["status"] = "admitted"
        candidate["conversion"].update(status="passed", byte_identical_reconversion=False)
        candidate["artifact"].update(output_sha256="a" * 64, offsets_valid=True)
        candidate["repeated_execution"]["byte_identical"] = False
        candidate["characterization"]["proposed_logit_max_abs_envelope"] = 0.01
        record["selection"].update(outcome="exactly_one_candidate", selected_candidate_id=candidate["candidate_id"])
        self.assertTrue(validate(record))


if __name__ == "__main__":
    unittest.main()
