#!/usr/bin/env python3
"""Tests for pinned Safetensors source validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import validate_pinned_source as source_validator


class PinnedSourceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="clr-source-validation-")
        self.root = Path(self.temporary.name)
        self.records = []
        names = ["model.safetensors.index.json", *[f"model-{index:05}-of-00016.safetensors" for index in range(1, 17)]]
        for index, name in enumerate(names):
            payload = bytes([index])
            (self.root / name).write_bytes(payload)
            self.records.append(
                {
                    "bytes": len(payload),
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        self.manifest_path = self.root / "source-manifest-v1.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "files": self.records,
                    "model": {"id": "test/model", "revision": "1" * 40},
                    "schema_version": 1,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_validates_index_and_sixteen_shards(self) -> None:
        result = source_validator.validate(self.root, self.manifest_path)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["file_count"], 17)
        self.assertEqual(result["source_bytes"], 17)

    def test_rejects_missing_source(self) -> None:
        (self.root / self.records[-1]["path"]).unlink()
        with self.assertRaisesRegex(source_validator.SourceValidationError, "missing"):
            source_validator.validate(self.root, self.manifest_path)

    def test_rejects_hash_mismatch(self) -> None:
        (self.root / self.records[1]["path"]).write_bytes(b"x")
        with self.assertRaisesRegex(source_validator.SourceValidationError, "hash mismatch"):
            source_validator.validate(self.root, self.manifest_path)


if __name__ == "__main__":
    unittest.main()
