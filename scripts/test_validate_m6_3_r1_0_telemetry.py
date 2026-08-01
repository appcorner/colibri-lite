import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_m6_3_r1_0_telemetry import validate


def valid_record():
    run = {
        "run_id": "r1",
        "exit_code": 0,
        "samples": 2,
        "memory": {
            "working_set_peak_bytes": 10,
            "peak_working_set_bytes": 12,
            "private_bytes_peak": 20,
        },
        "logical_io": {"bytes_read": 7, "source": "runtime_accounting"},
        "explicit_memory": {"source": "runtime_accounting"},
        "physical_io": {"status": "correlated", "bytes_read": 7},
        "cache_state": {"runtime": "not_applicable", "os_filesystem": "uncontrolled"},
    }
    return {
        "schema": "m6.3-r1-telemetry-v1",
        "runtime": {
            "executable": "clr-cli.exe",
            "command": ["generate"],
            "reference_identity": "reference-f32-v1",
        },
        "collector": {
            "powershell": "7",
            "sample_interval_ms": 100,
            "etw": {"status": "correlated", "trace_path": "trace.etl", "trace_sha256": "a" * 64},
        },
        "runs": [dict(run, run_id=f"r{i}") for i in range(5)],
        "gates": {"five_reference_runs": True, "collector_reconciliation": True, "physical_io": True},
    }


class TelemetryValidatorTests(unittest.TestCase):
    def test_accepts_reconciled_five_runs(self):
        self.assertEqual(validate(valid_record()), [])

    def test_rejects_missing_samples_and_uncorrelated_etw(self):
        record = valid_record()
        record["runs"][0]["samples"] = 0
        record["collector"]["etw"]["status"] = "not_measured"
        errors = validate(record)
        self.assertTrue(any("no independent memory samples" in item for item in errors))
        self.assertTrue(any("ETW" in item for item in errors))

    def test_rejects_runtime_cache_claim_for_os_cache(self):
        record = valid_record()
        record["runs"][0]["cache_state"]["os_filesystem"] = "cold_device"
        self.assertTrue(any("uncontrolled" in item for item in validate(record)))

    def test_rejects_logical_bytes_as_physical_io(self):
        record = valid_record()
        record["runs"][0]["physical_io"] = {"status": "not_measured", "bytes_read": None}
        record["collector"]["etw"]["status"] = "not_measured"
        record["gates"]["physical_io"] = False
        self.assertTrue(any("ETW" in item for item in validate(record)))


if __name__ == "__main__":
    unittest.main()
