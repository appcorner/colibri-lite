from __future__ import annotations

import unittest

from python.reference.validate_m6_1_04_memory_gpu_profile import SCHEMA, validate


def benchmark() -> dict[str, object]:
    return {"status": "not_run", "unit": "GiB/s", "cache_state": "not_applicable", "payload_bytes": 0, "repetitions": 0, "reason": "no backend"}


def document() -> dict[str, object]:
    backend = {
        "backend_id": "cuda", "availability": "unavailable", "usable_vram_bytes": 0,
        "host_to_device": benchmark(), "device_to_host": benchmark(),
    }
    return {
        "schema": SCHEMA, "schema_version": 1, "profile_id": "test", "created_at": "2026-07-30T00:00:00Z",
        "runtime": {"commit": "test"},
        "ram": {"total_physical_bytes": 16, "available_physical_bytes": 12, "safety_reserve_bytes": 4, "usable_budget_bytes": 8},
        "adapters": [], "backends": [backend], "recommendations": {"ram_budget_bytes": 8, "vram_budget_bytes": 0, "confidence": "partial"},
    }


class MemoryGpuProfileValidatorTests(unittest.TestCase):
    def test_accepts_detected_but_unusable_backends(self) -> None:
        self.assertEqual(validate(document()), [])

    def test_rejects_unreviewed_backend_vram_or_transfer_claim(self) -> None:
        evidence = document()
        evidence["backends"][0]["availability"] = "measured"
        evidence["backends"][0]["usable_vram_bytes"] = 1024
        evidence["backends"][0]["host_to_device"]["status"] = "measured"
        errors = validate(evidence)
        self.assertTrue(any("M6.3" in error for error in errors))
        self.assertTrue(any("usable_vram_bytes" in error for error in errors))
        self.assertTrue(any("host_to_device" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
