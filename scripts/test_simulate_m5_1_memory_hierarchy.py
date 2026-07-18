import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_m5_1_memory_hierarchy import Cache


def record(ordinal, layer, expert, payload=10):
    return {"global_ordinal": ordinal, "layer_index": layer, "expert_id": expert, "layer_expert_key": f"layer.{layer}.expert.{expert}", "payload_bytes": payload}


class SimulatorTests(unittest.TestCase):
    def test_lru_hit_and_eviction(self):
        records = [record(0, 0, 0), record(1, 0, 1), record(2, 0, 0)]
        cache = Cache(align(10), "lru", records)
        for i, item in enumerate(records):
            cache.request(i, item)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.evictions, 2)

    def test_lru_two_entries_hits(self):
        records = [record(0, 0, 0), record(1, 0, 1), record(2, 0, 0)]
        cache = Cache(2 * align(10), "lru", records)
        for i, item in enumerate(records):
            cache.request(i, item)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.loads, 2)

    def test_oversized_entry_bypasses(self):
        records = [record(0, 0, 0, 5000)]
        cache = Cache(align(10), "lru", records)
        cache.request(0, records[0])
        self.assertEqual(cache.loads, 1)
        self.assertEqual(cache.resident, 0)

    def test_frequency_is_deterministic(self):
        records = [record(0, 0, 0), record(1, 0, 1), record(2, 0, 0), record(3, 0, 2)]
        cache = Cache(2 * align(10), "frequency", records)
        for i, item in enumerate(records):
            cache.request(i, item)
        self.assertEqual(cache.hits, 1)

    def test_belady_upper_bound(self):
        records = [record(0, 0, 0), record(1, 0, 1), record(2, 0, 2), record(3, 0, 0), record(4, 0, 1)]
        cache = Cache(2 * align(10), "belady", records)
        for i, item in enumerate(records):
            cache.request(i, item)
        self.assertGreaterEqual(cache.hits, 1)


def align(payload):
    return ((payload + 64 + 4095) // 4096) * 4096


if __name__ == "__main__":
    unittest.main()
