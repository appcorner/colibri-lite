#!/usr/bin/env python3
"""Failure and determinism tests for the unified model artifact manifest."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_artifact_manifest as artifact


def write_bytes(path: Path, value: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {
        "byte_length": len(value),
        "sha256": artifact.sha256_bytes(value),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def source_config() -> dict[str, object]:
    return {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "decoder_sparse_step": 1,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "mlp_only_layers": [],
        "moe_intermediate_size": 768,
        "norm_topk_prob": True,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
        "rms_norm_eps": 0.000001,
        "rope_scaling": None,
        "rope_theta": 1000000.0,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "vocab_size": 151936,
    }


def create_fixture(root: Path) -> None:
    source = {
        "schema_version": 1,
        "model": {
            "architecture": artifact.ARCHITECTURE,
            "id": artifact.MODEL_ID,
            "license": artifact.LICENSE,
            "model_type": artifact.MODEL_TYPE,
            "revision": artifact.REVISION,
        },
        "config": source_config(),
    }
    write_json(root / "provenance" / "source-manifest-v1.json", source)

    dense_payload = write_bytes(root / "dense" / "dense-f32.bin", b"D")
    dense = {
        "artifact": {
            "byte_length": dense_payload["byte_length"],
            "path": "dense-f32.bin",
            "sha256": dense_payload["sha256"],
        },
        "artifact_dtype": "F32",
        "endianness": "little",
        "format_version": 1,
        "model_id": artifact.MODEL_ID,
        "model_revision": artifact.REVISION,
        "source_dtype": "BF16",
        "tensors": [{} for _ in range(435)],
    }
    write_json(root / "dense" / "dense-manifest-v1.json", dense)

    shards = []
    for index in range(48):
        name = f"experts-layer-{index:05}-of-00048.bin"
        metadata = write_bytes(root / "experts" / name, bytes([index]))
        shards.append(
            {
                "byte_length": metadata["byte_length"],
                "path": name,
                "sha256": metadata["sha256"],
                "shard_id": index,
            }
        )
    experts = {
        "artifact_dtype": "F32",
        "endianness": "little",
        "experts": [
            {"expert": expert, "layer": layer}
            for layer in range(48)
            for expert in range(128)
        ],
        "format_version": 1,
        "model_id": artifact.MODEL_ID,
        "model_revision": artifact.REVISION,
        "shard_policy": "one container per selected layer",
        "shards": shards,
        "source_dtype": "BF16",
    }
    write_json(root / "experts" / "expert-manifest-v1.json", experts)

    tokenizer_files = []
    for index, name in enumerate(("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")):
        payload = bytes([ord("a") + index])
        path = root / "tokenizer" / name
        metadata = write_bytes(path, payload)
        tokenizer_files.append(
            {
                "bytes": metadata["byte_length"],
                "path": name,
                "sha256": metadata["sha256"],
            }
        )
    tokenizer = {
        "chat_template": {
            "bytes": 1,
            "characters": 1,
            "preserved": True,
            "rendering_implemented": False,
            "sha256": artifact.sha256_bytes(b"c"),
            "source_path": "tokenizer_config.json",
        },
        "files": tokenizer_files,
        "format_version": 1,
        "model_id": artifact.MODEL_ID,
        "revision": artifact.REVISION,
        "tokenizer": {
            "added_token_count": 26,
            "base_vocabulary_size": 151643,
            "class": "Qwen2Tokenizer",
            "tokenizer_size_with_added_tokens": 151669,
        },
    }
    write_json(root / "tokenizer" / "tokenizer-artifact-manifest-v1.json", tokenizer)

    document = artifact.generate_document(root)
    (root / artifact.ROOT_MANIFEST_NAME).write_bytes(artifact.canonical_json(document))


class UnifiedManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="clr-manifest-")
        self.root = Path(self.temporary.name) / "artifact"
        create_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def document(self) -> dict[str, object]:
        return artifact.read_json(self.root / artifact.ROOT_MANIFEST_NAME)

    def test_generation_is_deterministic_and_relative_after_move(self) -> None:
        first = artifact.canonical_json(artifact.generate_document(self.root))
        second = artifact.canonical_json(artifact.generate_document(self.root))
        self.assertEqual(first, second)
        self.assertEqual(artifact.sha256_bytes(first), artifact.sha256_bytes(second))
        self.assertEqual(artifact.validate_artifact(self.root, False)["status"], "passed")
        self.assertEqual(artifact.validate_artifact(self.root, True)["status"], "passed")

        moved = self.root.with_name("moved-artifact")
        shutil.move(self.root, moved)
        self.root = moved
        self.assertEqual(artifact.validate_artifact(self.root, False)["status"], "passed")
        self.assertEqual(artifact.validate_artifact(self.root, True)["status"], "passed")

    def test_missing_renamed_truncated_and_corrupted_files_fail(self) -> None:
        payload = self.root / "dense" / "dense-f32.bin"
        renamed = payload.with_name("renamed.bin")
        payload.rename(renamed)
        with self.assertRaisesRegex(artifact.ManifestError, "missing or renamed"):
            artifact.validate_artifact(self.root, False)
        renamed.rename(payload)

        payload.write_bytes(b"")
        with self.assertRaisesRegex(artifact.ManifestError, "size mismatch"):
            artifact.validate_artifact(self.root, False)
        payload.write_bytes(b"D")

        payload.write_bytes(b"X")
        self.assertEqual(artifact.validate_artifact(self.root, False)["status"], "passed")
        with self.assertRaisesRegex(artifact.ManifestError, "hash mismatch"):
            artifact.validate_artifact(self.root, True)

    def test_missing_and_duplicate_expert_shards_fail(self) -> None:
        document = self.document()
        missing = copy.deepcopy(document)
        missing["components"]["experts"]["shards"].pop()
        with self.assertRaisesRegex(artifact.ManifestError, "incomplete"):
            artifact.validate_root_document(missing)

        duplicate = copy.deepcopy(document)
        duplicate["components"]["experts"]["shards"][1]["shard_id"] = 0
        with self.assertRaisesRegex(artifact.ManifestError, "missing, duplicated, or out of order"):
            artifact.validate_root_document(duplicate)

        path = self.root / "experts" / "experts-layer-00047-of-00048.bin"
        path.unlink()
        with self.assertRaisesRegex(artifact.ManifestError, "missing or renamed"):
            artifact.validate_artifact(self.root, False)

    def test_versions_identity_unknown_fields_and_incomplete_output_fail(self) -> None:
        document = self.document()
        unsupported = copy.deepcopy(document)
        unsupported["format_version"] = 2
        with self.assertRaisesRegex(artifact.ManifestError, "unsupported root"):
            artifact.validate_root_document(unsupported)

        dense_version = copy.deepcopy(document)
        dense_version["components"]["dense"]["format_version"] = 2
        with self.assertRaisesRegex(artifact.ManifestError, "unsupported dense"):
            artifact.validate_root_document(dense_version)

        for field, value in (("architecture", "OtherModel"), ("model_type", "other")):
            incompatible = copy.deepcopy(document)
            incompatible[field] = value
            with self.assertRaisesRegex(artifact.ManifestError, "incompatible"):
                artifact.validate_root_document(incompatible)

        unknown = copy.deepcopy(document)
        unknown["unknown_critical_field"] = True
        with self.assertRaisesRegex(artifact.ManifestError, "fields differ"):
            artifact.validate_root_document(unknown)

        incomplete = self.root / "experts" / ".layer.incomplete"
        incomplete.write_bytes(b"")
        with self.assertRaisesRegex(artifact.ManifestError, "incomplete temporary output"):
            artifact.validate_artifact(self.root, False)


if __name__ == "__main__":
    unittest.main()
