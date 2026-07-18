import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_m5_4_resident_dense import fixed_bytes


class ResidentDenseSimulationTests(unittest.TestCase):
    def setUp(self):
        self.components = {
            "dense_artifact_bytes": 100,
            "dense_stream_buffer_bytes": 10,
            "decoded_expert_buffer_bytes": 20,
            "runtime_structures_bytes": 30,
            "safety_reserve_bytes": 40,
        }

    def test_streamed_dense_reserves_stream_buffer(self):
        self.assertEqual(fixed_bytes(self.components, "streamed_dense"), 100)

    def test_resident_dense_reserves_full_artifact(self):
        self.assertEqual(fixed_bytes(self.components, "resident_dense"), 190)

    def test_unknown_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            fixed_bytes(self.components, "unknown")


if __name__ == "__main__":
    unittest.main()
