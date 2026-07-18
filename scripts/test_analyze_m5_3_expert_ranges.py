import importlib.util
import sys
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "analyze_m5_3_expert_ranges",
    Path(__file__).with_name("analyze_m5_3_expert_ranges.py"),
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def r(key, layer, expert, start):
    return MODULE.Range(key, layer, expert, layer, start, 10)


class RangeSimulationTests(unittest.TestCase):
    def test_exact_and_bounded_gap_grouping_are_deterministic(self):
        ranges = [r("a", 0, 0, 0), r("b", 0, 1, 10), r("c", 0, 2, 25)]
        exact = MODULE.range_stats(ranges, 0)
        self.assertEqual(exact["read_operations"], 2)
        self.assertEqual(exact["total_bytes_read"], 30)
        self.assertEqual(exact["over_read_bytes"], 0)
        bounded = MODULE.range_stats(ranges, 5)
        self.assertEqual(bounded["read_operations"], 1)
        self.assertEqual(bounded["total_bytes_read"], 35)

    def test_layer_batches_do_not_cross_layers(self):
        ranges = [r("a", 0, 0, 0), r("b", 0, 1, 20), r("c", 1, 0, 0)]
        result = MODULE.range_stats(ranges, 0, layer_batch=True)
        self.assertEqual(result["read_operations"], 2)
        self.assertEqual(result["maximum_temporary_buffer_bytes"], 30)

    def test_one_expert_lru_has_no_hit_for_unique_requests(self):
        trace = {
            "records": [
                {"layer_expert_key": "layer.0.expert.0"},
                {"layer_expert_key": "layer.0.expert.1"},
                {"layer_expert_key": "layer.0.expert.0"},
            ],
            "_ranges": {
                "layer.0.expert.0": r("layer.0.expert.0", 0, 0, 0),
                "layer.0.expert.1": r("layer.0.expert.1", 0, 1, 10),
            },
        }
        _, counters = MODULE.replay_lru(trace, 10)
        self.assertEqual(counters, {"requests": 3, "hits": 0, "misses": 3, "loads": 3, "evictions": 2})


if __name__ == "__main__":
    unittest.main()
