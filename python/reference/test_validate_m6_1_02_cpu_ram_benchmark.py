from __future__ import annotations

import unittest

from python.reference.validate_m6_1_02_cpu_ram_benchmark import SCHEMA, validate


def valid_document() -> dict[str, object]:
    distribution = {
        "samples": [1.0, 2.0, 3.0],
        "minimum": 1.0,
        "p10": 1.0,
        "median": 2.0,
        "p90": 3.0,
        "maximum": 3.0,
    }
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "profile_id": "test",
        "created_at": "2026-07-30T00:00:00Z",
        "runtime": {"commit": "test"},
        "host": {"cpu_model": "test"},
        "measurement_semantics": {"clock": "test"},
        "results": {
            "cpu_kernel_gflops": distribution,
            "ram_copy_gib_per_second": distribution.copy(),
        },
    }


class ValidateM6102BenchmarkTests(unittest.TestCase):
    def test_accepts_ordered_positive_distributions(self) -> None:
        self.assertEqual(validate(valid_document()), [])

    def test_rejects_unsorted_or_nonfinite_samples(self) -> None:
        document = valid_document()
        document["results"]["cpu_kernel_gflops"]["samples"] = [2.0, float("inf"), 1.0]
        errors = validate(document)
        self.assertTrue(any("finite and positive" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
