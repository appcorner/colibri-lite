#!/usr/bin/env python3
"""Unit tests for deterministic full-model tensor value sampling."""

from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from python.reference import validate_full_model_tensor_values as values


class TensorValueTests(unittest.TestCase):
    def test_sample_indices_are_deterministic_and_include_boundaries(self) -> None:
        first = values.sample_indices("tensor", 100)
        self.assertEqual(first, values.sample_indices("tensor", 100))
        self.assertIn(0, first)
        self.assertIn(50, first)
        self.assertIn(99, first)

    def test_exact_bf16_to_f32_bit_comparison(self) -> None:
        with tempfile.TemporaryDirectory(prefix="clr-tensor-values-") as directory:
            root = Path(directory)
            source = root / "source.bin"
            artifact = root / "artifact.bin"
            bf16 = [0x3F80, 0x8000, 0x0001, 0x7F80]
            source.write_bytes(b"".join(value.to_bytes(2, "little") for value in bf16))
            artifact.write_bytes(b"".join((value << 16).to_bytes(4, "little") for value in bf16))
            name = next(f"tensor-{index}" for index in range(100) if len(values.sample_indices(f"tensor-{index}", 4)) == 4)
            samples = values.compare_samples("dense", name, [4], source, 0, artifact, 0, 0, "artifact.bin")
            self.assertEqual(len(samples), 4)
            self.assertEqual(samples[0]["f32_bits"], "0x3f800000")
            self.assertEqual(samples[1]["f32_bits"], "0x80000000")
            self.assertEqual(samples[2]["f32_bits"], "0x00010000")
            self.assertEqual(samples[3]["f32_bits"], "0x7f800000")

    def test_value_mismatch_is_structured(self) -> None:
        with tempfile.TemporaryDirectory(prefix="clr-tensor-values-") as directory:
            root = Path(directory)
            source = root / "source.bin"
            artifact = root / "artifact.bin"
            source.write_bytes(struct.pack("<H", 0x3F80))
            artifact.write_bytes(struct.pack("<I", 0))
            with self.assertRaisesRegex(values.TensorValueError, "tensor value mismatch"):
                values.compare_samples("dense", "tensor", [1], source, 0, artifact, 0, 0, "artifact.bin")


if __name__ == "__main__":
    unittest.main()
