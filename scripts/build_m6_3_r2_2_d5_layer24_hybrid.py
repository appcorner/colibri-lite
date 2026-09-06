#!/usr/bin/env python3
"""Build the frozen D5 Layer24 hybrid artifact without requantization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

EXPERTS = 128
HIDDEN = 2048
INTERMEDIATE = 768
GROUP_SIZE = 8
VALUES_PER_PROJECTION = HIDDEN * INTERMEDIATE
F32_PROJECTION_BYTES = VALUES_PER_PROJECTION * 4
F32_EXPERT_BYTES = F32_PROJECTION_BYTES * 3
PACKED_VALUES_BYTES = VALUES_PER_PROJECTION
PACKED_SCALES_BYTES = (VALUES_PER_PROJECTION // GROUP_SIZE) * 4
PACKED_PROJECTION_BYTES = PACKED_VALUES_BYTES + PACKED_SCALES_BYTES
PACKED_EXPERT_BYTES = PACKED_PROJECTION_BYTES * 3
HYBRID_EXPERT_BYTES = 2 * F32_PROJECTION_BYTES + PACKED_PROJECTION_BYTES
F32_LAYER_BYTES = F32_EXPERT_BYTES * EXPERTS
PACKED_LAYER_BYTES = PACKED_EXPERT_BYTES * EXPERTS
HYBRID_LAYER_BYTES = HYBRID_EXPERT_BYTES * EXPERTS
COPY_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def copy_exact(reader, writer, length: int, digest: hashlib._Hash) -> None:
    remaining = length
    while remaining:
        chunk = reader.read(min(COPY_CHUNK_BYTES, remaining))
        if not chunk:
            raise RuntimeError("short source read while building D5 hybrid artifact")
        writer.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def build(
    f32_source: Path,
    group8_source: Path,
    output: Path,
    manifest: Path,
    expected_f32_sha256: str,
    expected_group8_sha256: str,
) -> dict:
    if f32_source.stat().st_size != F32_LAYER_BYTES:
        raise RuntimeError("canonical Layer24 F32 byte length mismatch")
    if group8_source.stat().st_size != PACKED_LAYER_BYTES:
        raise RuntimeError("frozen Layer24 group8 byte length mismatch")
    actual_f32_sha256 = sha256_file(f32_source)
    if actual_f32_sha256 != expected_f32_sha256:
        raise RuntimeError("canonical Layer24 F32 SHA-256 mismatch")
    actual_group8_sha256 = sha256_file(group8_source)
    if actual_group8_sha256 != expected_group8_sha256:
        raise RuntimeError("frozen Layer24 group8 SHA-256 mismatch")
    if output.exists() or manifest.exists():
        raise RuntimeError("D5 hybrid output and manifest must be new")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    manifest_partial = manifest.with_suffix(manifest.suffix + ".partial")
    if partial.exists() or manifest_partial.exists():
        raise RuntimeError("D5 partial output already exists")

    digest = hashlib.sha256()
    try:
        with f32_source.open("rb") as f32, group8_source.open("rb") as group8, partial.open("xb") as writer:
            for expert in range(EXPERTS):
                f32.seek(expert * F32_EXPERT_BYTES)
                copy_exact(f32, writer, 2 * F32_PROJECTION_BYTES, digest)
                group8.seek(expert * PACKED_EXPERT_BYTES + 2 * PACKED_PROJECTION_BYTES)
                copy_exact(group8, writer, PACKED_PROJECTION_BYTES, digest)
            writer.flush()
            os.fsync(writer.fileno())
        if partial.stat().st_size != HYBRID_LAYER_BYTES:
            raise RuntimeError("D5 hybrid Layer24 byte length mismatch")
        output_sha256 = digest.hexdigest()
        result = {
            "schema": "m6.3-r2.2-d5-layer24-hybrid-artifact-v1",
            "schema_version": 1,
            "layer": 24,
            "experts": EXPERTS,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "group_size": GROUP_SIZE,
            "source_f32": {
                "path": str(f32_source),
                "bytes": F32_LAYER_BYTES,
                "sha256": actual_f32_sha256,
            },
            "source_group8": {
                "path": str(group8_source),
                "bytes": PACKED_LAYER_BYTES,
                "sha256": actual_group8_sha256,
            },
            "expert_layout": {
                "expert_bytes": HYBRID_EXPERT_BYTES,
                "gate": {"offset": 0, "length": F32_PROJECTION_BYTES, "dtype": "f32_le"},
                "up": {"offset": F32_PROJECTION_BYTES, "length": F32_PROJECTION_BYTES, "dtype": "f32_le"},
                "down_values": {
                    "offset": 2 * F32_PROJECTION_BYTES,
                    "length": PACKED_VALUES_BYTES,
                    "dtype": "i8",
                },
                "down_scales": {
                    "offset": 2 * F32_PROJECTION_BYTES + PACKED_VALUES_BYTES,
                    "length": PACKED_SCALES_BYTES,
                    "dtype": "f32_le",
                },
            },
            "artifact": {"path": str(output), "bytes": HYBRID_LAYER_BYTES, "sha256": output_sha256},
            "construction": "copy_f32_gate_up_and_frozen_group8_down_without_requantization",
            "whole_f32_expert_materialized": False,
            "f32_down_materialized": False,
        }
        manifest_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with manifest_partial.open("xb") as writer:
            writer.write(manifest_bytes)
            writer.flush()
            os.fsync(writer.fileno())
        partial.replace(output)
        manifest_partial.replace(manifest)
        return result
    except Exception:
        partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f32-source", type=Path, required=True)
    parser.add_argument("--group8-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-f32-sha256", required=True)
    parser.add_argument("--expected-group8-sha256", required=True)
    args = parser.parse_args()
    result = build(
        args.f32_source,
        args.group8_source,
        args.output,
        args.manifest,
        args.expected_f32_sha256,
        args.expected_group8_sha256,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
