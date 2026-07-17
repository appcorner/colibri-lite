"""Focused tests for the deterministic M4 release-provenance record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from python.reference.build_m4_release_provenance import (
    BASELINE_ID,
    BASELINE_SCHEMA,
    BASELINE_SHA256,
    RELEASE_ID,
    RELEASE_TAG,
    RUNTIME_COMMIT,
    build,
    canonical_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models" / "qwen3-30b-a3b"
PROVENANCE = MODEL_ROOT / "m4-release-provenance-v1.json"


class M4ReleaseProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    def test_identity_and_verdict(self) -> None:
        self.assertEqual(self.document["release_id"], RELEASE_ID)
        self.assertEqual(self.document["release_tag"], RELEASE_TAG)
        self.assertEqual(self.document["runtime_source"]["runtime_commit"], RUNTIME_COMMIT)
        self.assertEqual(self.document["m4_baseline"]["baseline_id"], BASELINE_ID)
        self.assertEqual(self.document["m4_baseline"]["baseline_schema"], BASELINE_SCHEMA)
        self.assertEqual(self.document["m4_baseline"]["baseline_sha256"], BASELINE_SHA256)
        self.assertEqual(
            self.document["m4_baseline"]["f32_manifest"]["path"],
            "models/qwen3-30b-a3b/m4.3-01-f32-baseline-manifest-v1.json",
        )
        self.assertEqual(self.document["m4_verdict"]["performance_readiness"], "not_ready")
        self.assertEqual(self.document["m4_verdict"]["first_full_model_int8_candidate"], "rejected")
        self.assertFalse(self.document["m5_gate"]["started"])

    def test_all_references_exist_and_match(self) -> None:
        self.assertGreaterEqual(len(self.document["references"]), 21)
        roles = set()
        for item in self.document["references"]:
            roles.add(item["role"])
            path = ROOT / item["path"]
            payload = path.read_bytes()
            self.assertEqual(len(payload), item["bytes"], item["path"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"], item["path"])
        self.assertEqual(len(roles), len(self.document["references"]))

    def test_artifact_and_external_identity(self) -> None:
        artifact = self.document["canonical_artifact"]
        self.assertEqual(artifact["root_manifest_sha256"], "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2")
        self.assertEqual(artifact["total_bytes"], 122147666917)
        self.assertEqual(artifact["dtype"], {"source": "BF16", "storage": "F32", "compute": "F32"})
        external = self.document["external_reference"]
        self.assertEqual(external["commit"], "1fddd12ba861c4815a8633f14d9c5670692099cc")
        self.assertEqual(external["model"]["format"], "Q4_K_M")
        self.assertFalse(external["quality_equivalence_to_f32"])

    def test_repeated_build_is_byte_identical(self) -> None:
        expected = PROVENANCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            first = canonical_bytes(build(ROOT))
            second = canonical_bytes(build(ROOT))
            Path(directory, "first.json").write_bytes(first)
            self.assertEqual(first, second)
            self.assertEqual(first, expected)
            self.assertEqual(hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest())


if __name__ == "__main__":
    unittest.main()
