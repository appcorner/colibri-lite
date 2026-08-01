#!/usr/bin/env python3
"""Failure tests for the closed minimal-oracle source validator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import validate_minimal_oracle_source as validator


class MinimalOracleSourceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="clr-minimal-oracle-")
        self.root = Path(self.temporary.name) / "source"
        self.root.mkdir()
        self.shards = [f"model-{number:05}-of-00016.safetensors" for number in range(1, 17)]
        records = []
        for number, name in enumerate(self.shards, start=1):
            payload = bytes([number])
            (self.root / name).write_bytes(payload)
            records.append({"path": name, "bytes": 1, "sha256": hashlib.sha256(payload).hexdigest()})
        index_payload = json.dumps({"weight_map": {f"tensor.{number}": name for number, name in enumerate(self.shards)}}).encode()
        (self.root / "model.safetensors.index.json").write_bytes(index_payload)
        records.append({"path": "model.safetensors.index.json", "bytes": len(index_payload), "sha256": hashlib.sha256(index_payload).hexdigest()})
        self.manifest = self.root.parent / "manifest.json"
        self.manifest.write_text(json.dumps({"files": records, "model": {"id": "Qwen/Qwen3-30B-A3B", "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "license": "Apache-2.0", "architecture": "Qwen3MoeForCausalLM"}, "safetensors": {"index_file": "model.safetensors.index.json", "tensor_count": 16, "shards": self.shards}}), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def validate(self) -> dict[str, object]:
        total = sum(path.stat().st_size for path in self.root.iterdir())
        with patch.object(validator, "MINIMAL_SOURCE_TOTAL_BYTES", total):
            return validator.validate(self.root, self.manifest)

    def test_missing_file_is_rejected(self) -> None:
        (self.root / self.shards[0]).unlink()
        with self.assertRaisesRegex(validator.MinimalOracleSourceError, "missing or unexpected"):
            self.validate()

    def test_tampered_file_is_rejected(self) -> None:
        (self.root / self.shards[0]).write_bytes(b"tampered")
        with self.assertRaisesRegex(validator.MinimalOracleSourceError, "size mismatch"):
            self.validate()

    def test_unexpected_file_is_rejected(self) -> None:
        (self.root / "extra.bin").write_bytes(b"no")
        with self.assertRaisesRegex(validator.MinimalOracleSourceError, "missing or unexpected"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
