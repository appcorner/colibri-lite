#!/usr/bin/env python3
"""Build one verified full-projection grouped artifact for an R2.3b layer/group."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from convert_m6_3_r2_2_d2_grouped_layer import (
    EXPERTS,
    SOURCE_LAYER_BYTES,
    convert,
    packed_expert_bytes,
)

ROOT_MANIFEST_SHA256 = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
LAYERS = tuple(range(1, 24)) + tuple(range(25, 47))
GROUPS = (32, 16, 8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_source(artifact_root: Path, layer: int) -> tuple[Path, str]:
    manifest_path = artifact_root / "model-manifest-v1.json"
    require(manifest_path.is_file(), "canonical root manifest is missing")
    require(sha256(manifest_path) == ROOT_MANIFEST_SHA256, "canonical root manifest SHA-256 mismatch")
    expert_manifest_path = artifact_root / "experts" / "expert-manifest-v1.json"
    manifest = load_json(expert_manifest_path)
    require(manifest["model_id"] == MODEL_ID, "expert manifest model mismatch")
    require(manifest["model_revision"] == MODEL_REVISION, "expert manifest revision mismatch")
    shard = manifest["shards"][layer]
    require(shard["shard_id"] == layer, "expert shard ordering mismatch")
    require(shard["byte_length"] == SOURCE_LAYER_BYTES, "expert shard byte length mismatch")
    source = (artifact_root / "experts" / shard["path"]).resolve()
    require(source.parent == (artifact_root / "experts").resolve(), "expert shard escaped canonical root")
    return source, shard["sha256"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=LAYERS, required=True)
    parser.add_argument("--group-size", type=int, choices=GROUPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    output = args.output.resolve()
    provenance = args.provenance.resolve()
    require(artifact_root.is_dir(), "canonical artifact root is missing")
    require(not output.exists(), "artifact output must be new")
    require(not provenance.exists(), "provenance output must be new")
    require(output.parent == provenance.parent, "artifact and provenance must share one flat run directory")
    require(output.parent.is_dir(), "flat run directory must already exist")
    source, expected_source_sha = canonical_source(artifact_root, args.layer)
    source_sha, output_sha = convert(source, output, expected_source_sha, args.group_size)
    expected_output_bytes = packed_expert_bytes(args.group_size) * EXPERTS
    require(output.stat().st_size == expected_output_bytes, "R2.3b packed output byte length mismatch")
    require(sha256(output) == output_sha, "R2.3b packed output post-write hash mismatch")

    record = {
        "schema": "colibri-m6.3-r2.3b-artifact-provenance-v1",
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license": "Apache-2.0",
        "canonical_root_manifest_sha256": ROOT_MANIFEST_SHA256,
        "layer": args.layer,
        "group_size": args.group_size,
        "layout": "r1.1-packed-expert-grouped-int8-values-f32-scales-v1",
        "source_relative_path": source.relative_to(artifact_root).as_posix(),
        "source_bytes": SOURCE_LAYER_BYTES,
        "source_sha256": source_sha,
        "artifact_name": output.name,
        "artifact_bytes": output.stat().st_size,
        "artifact_sha256": output_sha,
        "packed_expert_bytes": packed_expert_bytes(args.group_size),
        "projection_order": ["gate", "up", "down"],
        "rounding": "IEEE-754 f32 round-half-away-from-zero",
        "saturation": "[-127,127]",
        "byte_order": "little-endian",
    }
    temporary = provenance.with_suffix(provenance.suffix + ".incomplete")
    require(not temporary.exists(), "provenance incomplete path must be new")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(provenance)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
