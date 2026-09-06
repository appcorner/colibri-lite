import csv
import tempfile
import unittest
from pathlib import Path

from run_m6_3_r2_3b import completed_prefix, decision, parse_tsv
from test_validate_m6_3_r2_3b import candidate
from validate_m6_3_r2_3b import GROUPS, LAYERS, SUBSETS


class R23BRunnerTests(unittest.TestCase):
    def test_parse_tsv_requires_frozen_subset_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.tsv"
            rows = [candidate(1, 32, subset) for subset in SUBSETS]
            fieldnames = [name for name in rows[0] if name not in ("canonical_f32_control_exact",)]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    serialized = {name: row[name] for name in fieldnames}
                    for name, value in serialized.items():
                        if isinstance(value, bool):
                            serialized[name] = str(value).lower()
                    writer.writerow(serialized)
            parsed = parse_tsv(path)
            self.assertEqual(len(parsed), 7)
            self.assertTrue(all(row["canonical_f32_control_exact"] for row in parsed))

    def test_decision_uses_larger_group_then_subset_order(self) -> None:
        rows = [candidate(1, group, subset) for group in GROUPS for subset in SUBSETS]
        result = decision(1, rows)
        self.assertEqual(result["group_size"], 32)
        self.assertEqual(result["subset"], "gate_up_down")

    def test_completed_runs_must_be_exact_prefix(self) -> None:
        state = {"completed_runs": [{"layer": LAYERS[0], "group_size": GROUPS[1]}]}
        with self.assertRaisesRegex(RuntimeError, "exact frozen prefix"):
            completed_prefix(state)


if __name__ == "__main__":
    unittest.main()
