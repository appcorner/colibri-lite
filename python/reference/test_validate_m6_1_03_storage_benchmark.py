from __future__ import annotations

import unittest

from python.reference.validate_m6_1_03_storage_benchmark import SCHEMA, validate


def distribution() -> dict[str, object]:
    return {"samples": [1.0, 2.0, 3.0], "minimum": 1.0, "p10": 1.0, "median": 2.0, "p90": 3.0, "maximum": 3.0}


def document() -> dict[str, object]:
    return {
        "schema": SCHEMA, "schema_version": 1, "profile_id": "test", "created_at": "2026-07-30T00:00:00Z",
        "runtime": {"commit": "test"}, "storage_target": {"path": "D:/"}, "preflight": {"passed": True},
        "test_payload": {"expert_payload_bytes": 18_874_368},
        "cache_semantics": {"first_touch": "not claimed as a cold-device read", "warm": "warm"},
        "results": {
            "sequential_read": {"first_touch_throughput": 1.0, "warm_throughput": distribution()},
            "expert_sized_random_read": {name: distribution() for name in ("first_touch_latency", "first_touch_throughput", "warm_latency", "warm_throughput")},
        },
        "cleanup": {"run_directory_removed": True},
    }


class StorageBenchmarkValidatorTests(unittest.TestCase):
    def test_accepts_complete_evidence(self) -> None:
        self.assertEqual(validate(document()), [])

    def test_rejects_cold_device_claim_or_retained_run(self) -> None:
        evidence = document()
        evidence["cache_semantics"]["first_touch"] = "cold device"
        evidence["cleanup"]["run_directory_removed"] = False
        errors = validate(evidence)
        self.assertTrue(any("cold-device" in error for error in errors))
        self.assertTrue(any("run_directory_removed" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
