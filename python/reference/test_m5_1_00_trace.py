"""Tests for the deterministic ordered expert trace evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts import validate_m5_1_00_trace as trace_validator


class OrderedTraceTests(unittest.TestCase):
    def test_reuse_distance_uses_request_index(self) -> None:
        records = [
            {"layer_expert_key": "a"},
            {"layer_expert_key": "b"},
            {"layer_expert_key": "a"},
            {"layer_expert_key": "c"},
            {"layer_expert_key": "a"},
        ]
        self.assertEqual(trace_validator.reuse_distances(records), [2, 2])

    def test_canonical_trace_hash_and_header(self) -> None:
        path = Path("models/qwen3-30b-a3b/m5.1-00-ordered-expert-trace-v1.json")
        self.assertTrue(path.exists())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "f3f87f05d15424030c9261cdf3e93bd72e9c006a55303bc0c28a92a4fb3ff2d0",
        )
        trace = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(trace["requested_trace_count"], 2304)
        self.assertEqual(len(trace["records"]), 2304)
        self.assertEqual(trace["records"][0]["global_ordinal"], 0)
        self.assertEqual(trace["records"][-1]["global_ordinal"], 2303)


if __name__ == "__main__":
    unittest.main()
