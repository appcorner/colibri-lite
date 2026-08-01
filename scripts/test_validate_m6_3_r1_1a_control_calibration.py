import copy
import json
import unittest
from pathlib import Path

from validate_m6_3_r1_1a_control_calibration import validate


class ControlCalibrationValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = Path(__file__).resolve().parents[1]
        cls.record = json.loads(
            (cls.repository / "models/qwen3-30b-a3b/m6.3-r1-1a-code-newline-control-calibration-result-v1.json").read_text()
        )
        cls.plan = json.loads(
            (cls.repository / "models/qwen3-30b-a3b/m6.3-r1-1a-code-newline-control-calibration-plan-v1.json").read_text()
        )

    def test_valid_record(self) -> None:
        validate(self.record, self.plan, self.repository, False)

    def test_rejects_candidate_scope(self) -> None:
        record = copy.deepcopy(self.record)
        record["candidate_results_seen"] = True
        with self.assertRaisesRegex(ValueError, "candidate scope"):
            validate(record, self.plan, self.repository, False)

    def test_rejects_nonzero_run(self) -> None:
        record = copy.deepcopy(self.record)
        record["runs"][0]["exit_code"] = 101
        with self.assertRaisesRegex(ValueError, "run failure"):
            validate(record, self.plan, self.repository, False)

    def test_rejects_checkpoint_nondeterminism(self) -> None:
        record = copy.deepcopy(self.record)
        record["runs"][1]["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "determinism"):
            validate(record, self.plan, self.repository, False)

    def test_rejects_budget_change(self) -> None:
        record = copy.deepcopy(self.record)
        record["derived_f32_budgets"]["layer0.selected_expert_output"]["value"] *= 2
        with self.assertRaisesRegex(ValueError, "derived budget"):
            validate(record, self.plan, self.repository, False)

    def test_rejects_failed_final_control(self) -> None:
        record = copy.deepcopy(self.record)
        record["final_control_verification"]["exit_code"] = 101
        with self.assertRaisesRegex(ValueError, "final control"):
            validate(record, self.plan, self.repository, False)


if __name__ == "__main__":
    unittest.main()
