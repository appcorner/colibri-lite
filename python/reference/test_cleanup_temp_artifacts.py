#!/usr/bin/env python3
"""Safety tests for the dry-run-first temporary-artifact cleanup tool."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import cleanup_temp_artifacts as cleanup


def sha256_zeros() -> str:
    return "0" * 64


class CleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="clr-cleanup-")
        self.workspace = Path(self.temporary.name).resolve()
        self.temp_root = self.workspace / "tmp"
        self.temp_root.mkdir()
        self.canonical = self.workspace / "stable" / "canonical"
        self.canonical.parent.mkdir()
        self.canonical.mkdir()
        final = self.canonical / "final.bin"
        final.write_bytes(b"final")
        manifest = {
            "artifact_format": "test",
            "file": {"bytes": 5, "path": "final.bin", "sha256": sha256_zeros()},
        }
        (self.canonical / "model-manifest-v1.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.source = self.temp_root / "source.bin"
        self.source.write_bytes(b"source")
        self.candidate = self.temp_root / "completed-run"
        self.candidate.mkdir()
        (self.candidate / "temp.bin").write_bytes(b"temporary")
        self.plan_path = self.temp_root / "plan.json"
        self.registry_path = self.workspace / "registry.json"
        self.write_registry()
        self.write_plan(self.candidate)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def candidate_record(self, path: Path) -> dict[str, object]:
        measured = cleanup.inventory(path)
        return {
            "classification": "completed-task temporary output",
            "expected_file_count": measured["file_count"],
            "expected_logical_bytes": measured["logical_bytes"],
            "expected_reclaimable_bytes": measured["reclaimable_bytes"],
            "expected_shared_hardlink_bytes": measured["shared_hardlink_bytes"],
            "kind": "file" if path.is_file() else "directory",
            "path": str(path),
        }

    def write_plan(self, candidate: Path, protected: list[Path] | None = None) -> None:
        plan = {
            "canonical_artifact_root": str(self.canonical),
            "candidates": [self.candidate_record(candidate)],
            "generated_at": "test",
            "protected_paths": [str(path) for path in (protected or [self.source])],
            "schema_version": 1,
            "temp_root": str(self.temp_root),
        }
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def write_registry(self, canonical: Path | None = None) -> None:
        root = canonical or self.canonical
        manifest = self.canonical / "model-manifest-v1.json"
        registry = {
            "canonical_artifact_root": str(root),
            "canonical_file_count": 2,
            "model_id": "test/model",
            "revision": "1" * 40,
            "root_manifest_bytes": manifest.stat().st_size,
            "root_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "schema_version": 1,
        }
        self.registry_path.write_text(json.dumps(registry), encoding="utf-8")

    def test_dry_run_is_default_and_changes_nothing(self) -> None:
        result = cleanup.execute(self.plan_path, self.registry_path)
        self.assertEqual(result["mode"], "dry-run")
        self.assertTrue(self.candidate.exists())
        self.assertTrue((self.canonical / "final.bin").exists())
        self.assertTrue(self.source.exists())

    def test_apply_removes_only_reviewed_candidate(self) -> None:
        result = cleanup.execute(self.plan_path, self.registry_path, apply=True)
        self.assertEqual(result["mode"], "apply")
        self.assertFalse(self.candidate.exists())
        self.assertTrue(self.canonical.exists())
        self.assertTrue((self.canonical / "final.bin").exists())
        self.assertTrue(self.source.exists())

    def test_canonical_root_and_referenced_files_cannot_be_candidates(self) -> None:
        for candidate in (self.canonical, self.canonical / "final.bin", self.temp_root):
            self.write_plan(candidate)
            with self.assertRaises(cleanup.CleanupError):
                cleanup.execute(self.plan_path, self.registry_path, apply=True)
            self.assertTrue(self.canonical.exists())
            self.assertTrue((self.canonical / "final.bin").exists())

    def test_protected_source_and_parent_cannot_be_candidates(self) -> None:
        for candidate in (self.source, self.temp_root):
            self.write_plan(candidate, [self.source])
            with self.assertRaises(cleanup.CleanupError):
                cleanup.execute(self.plan_path, self.registry_path, apply=True)
            self.assertTrue(self.source.exists())

    def test_inventory_drift_refuses_cleanup(self) -> None:
        (self.candidate / "new.bin").write_bytes(b"drift")
        with self.assertRaisesRegex(cleanup.CleanupError, "drifted"):
            cleanup.execute(self.plan_path, self.registry_path, apply=True)
        self.assertTrue(self.candidate.exists())
        self.assertTrue(self.canonical.exists())

    def test_registry_mismatch_refuses_cleanup(self) -> None:
        self.write_registry(self.workspace / "other-stable-root")
        with self.assertRaisesRegex(cleanup.CleanupError, "differs from registry"):
            cleanup.execute(self.plan_path, self.registry_path, apply=True)
        self.assertTrue(self.candidate.exists())
        self.assertTrue(self.canonical.exists())


if __name__ == "__main__":
    unittest.main()
