#!/usr/bin/env python3
"""Compare selected pinned Safetensors BF16 values with stable F32 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any, Iterable


class TensorValueError(RuntimeError):
    """Selected source and artifact tensor values do not match."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TensorValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TensorValueError(f"cannot read JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_count(shape: list[int]) -> int:
    count = 1
    require(shape, "tensor shape must not be empty")
    for dimension in shape:
        require(isinstance(dimension, int) and dimension > 0, "tensor shape dimensions must be positive integers")
        count *= dimension
    return count


def sample_indices(name: str, element_count: int) -> list[int]:
    require(element_count > 0, "sampled tensor must not be empty")
    hashed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little") % element_count
    return sorted({0, hashed, element_count // 2, element_count - 1})


def read_integer(path: Path, offset: int, width: int) -> int:
    require(offset >= 0 and width in (2, 4), "invalid sample range")
    try:
        with path.open("rb") as source:
            source.seek(offset)
            payload = source.read(width)
    except OSError as error:
        raise TensorValueError(f"cannot read sample from {path}: {error}") from error
    require(len(payload) == width, f"truncated sample range: {path}")
    return int.from_bytes(payload, "little")


def parse_shape(value: str) -> list[int]:
    try:
        return [int(part) for part in value.split(",")]
    except ValueError as error:
        raise TensorValueError(f"invalid plan shape: {value}") from error


def parse_plan(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[int, int, str], dict[str, Any]]]:
    shards: dict[int, dict[str, Any]] = {}
    dense: dict[str, dict[str, Any]] = {}
    experts: dict[tuple[int, int, str], dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise TensorValueError(f"cannot read source plan {path}: {error}") from error
    for line in lines:
        fields = line.split("\t")
        if fields[0] == "shard":
            require(len(fields) == 5, "invalid shard plan record")
            index = int(fields[1])
            require(index not in shards, f"duplicate shard plan record: {index}")
            shards[index] = {"path": fields[2], "bytes": int(fields[3]), "sha256": fields[4]}
        elif fields[0] == "tensor":
            require(len(fields) == 6, "invalid dense tensor plan record")
            name = fields[1]
            require(name not in dense, f"duplicate dense tensor plan record: {name}")
            dense[name] = {
                "name": name,
                "shard_id": int(fields[2]),
                "offset": int(fields[3]),
                "length": int(fields[4]),
                "shape": parse_shape(fields[5]),
            }
        elif fields[0] == "projection":
            require(len(fields) == 9, "invalid expert projection plan record")
            key = (int(fields[1]), int(fields[2]), fields[3])
            require(key not in experts, f"duplicate expert projection plan record: {key}")
            experts[key] = {
                "name": fields[4],
                "shard_id": int(fields[5]),
                "offset": int(fields[6]),
                "length": int(fields[7]),
                "shape": parse_shape(fields[8]),
            }
    require(shards, f"source plan has no shards: {path}")
    return shards, dense, experts


def compare_samples(
    component: str,
    name: str,
    shape: list[int],
    source_path: Path,
    source_offset: int,
    artifact_path: Path,
    artifact_offset: int,
    source_shard_id: int,
    artifact_relative_path: str,
) -> list[dict[str, Any]]:
    samples = []
    for index in sample_indices(name, shape_count(shape)):
        bf16_bits = read_integer(source_path, source_offset + index * 2, 2)
        expected_f32_bits = bf16_bits << 16
        actual_f32_bits = read_integer(artifact_path, artifact_offset + index * 4, 4)
        require(
            actual_f32_bits == expected_f32_bits,
            f"tensor value mismatch: {name} element {index}: expected 0x{expected_f32_bits:08x}, got 0x{actual_f32_bits:08x}",
        )
        value = struct.unpack("<f", actual_f32_bits.to_bytes(4, "little"))[0]
        samples.append(
            {
                "artifact_offset": artifact_offset + index * 4,
                "artifact_path": artifact_relative_path,
                "bf16_bits": f"0x{bf16_bits:04x}",
                "component": component,
                "element_index": index,
                "f32_bits": f"0x{actual_f32_bits:08x}",
                "source_offset": source_offset + index * 2,
                "source_shard_id": source_shard_id,
                "tensor": name,
                "value": repr(value),
            }
        )
    return samples


def validate(
    source_root: Path,
    registry_path: Path,
    source_manifest_path: Path,
    dense_plan_path: Path,
    expert_plan_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    registry = read_json(registry_path)
    selection = read_json(selection_path)
    source_manifest = read_json(source_manifest_path)
    artifact_root = Path(registry["canonical_artifact_root"]).resolve()
    root_manifest_path = artifact_root / "model-manifest-v1.json"
    require(root_manifest_path.is_file(), "canonical root manifest is missing")
    require(root_manifest_path.stat().st_size == registry["root_manifest_bytes"], "canonical root manifest size mismatch")
    require(sha256_file(root_manifest_path) == registry["root_manifest_sha256"], "canonical root manifest hash mismatch")
    root_manifest = read_json(root_manifest_path)
    for document, context in ((selection, "selection"), (source_manifest["model"], "source"), (root_manifest, "artifact")):
        model_id = document.get("model_id", document.get("id"))
        revision = document.get("revision")
        require(model_id == registry["model_id"], f"{context} model ID mismatch")
        require(revision == registry["revision"], f"{context} revision mismatch")

    dense_shards, dense_plan, _ = parse_plan(dense_plan_path)
    expert_shards, _, expert_plan = parse_plan(expert_plan_path)
    require(dense_shards == expert_shards, "dense and expert source shard contracts differ")
    source_records = {record["path"]: record for record in source_manifest["files"]}

    dense_manifest_relative = root_manifest["components"]["dense"]["manifest"]["path"]
    expert_manifest_relative = root_manifest["components"]["experts"]["manifest"]["path"]
    dense_manifest = read_json(artifact_root / dense_manifest_relative)
    expert_manifest = read_json(artifact_root / expert_manifest_relative)
    require(dense_manifest["source_dtype"] == "BF16" and dense_manifest["artifact_dtype"] == "F32", "dense dtype contract mismatch")
    require(expert_manifest["source_dtype"] == "BF16" and expert_manifest["artifact_dtype"] == "F32", "expert dtype contract mismatch")
    dense_records = {record["name"]: record for record in dense_manifest["tensors"]}
    expert_records = {(record["layer"], record["expert"]): record for record in expert_manifest["experts"]}
    expert_artifact_shards = {record["shard_id"]: record for record in expert_manifest["shards"]}

    selected_shards: set[int] = set()
    for name in selection["dense_tensors"]:
        require(name in dense_plan and name in dense_records, f"selected dense tensor is missing: {name}")
        selected_shards.add(dense_plan[name]["shard_id"])
    for identity in selection["experts"]:
        for role in ("gate", "up", "down"):
            key = (identity["layer"], identity["expert"], role)
            require(key in expert_plan, f"selected expert projection is missing: {key}")
            selected_shards.add(expert_plan[key]["shard_id"])

    verified_shards = []
    for shard_id in sorted(selected_shards):
        shard = dense_shards[shard_id]
        record = source_records.get(shard["path"])
        require(record is not None, f"selected source shard is absent from provenance: {shard['path']}")
        require(record["bytes"] == shard["bytes"] and record["sha256"] == shard["sha256"], "source plan/provenance mismatch")
        path = source_root / shard["path"]
        require(path.is_file() and path.stat().st_size == shard["bytes"], f"selected source shard size mismatch: {shard['path']}")
        require(sha256_file(path) == shard["sha256"], f"selected source shard hash mismatch: {shard['path']}")
        verified_shards.append({"bytes": shard["bytes"], "path": shard["path"], "sha256": shard["sha256"], "shard_id": shard_id})

    dense_payload_relative = "dense/" + dense_manifest["artifact"]["path"]
    dense_payload = artifact_root / dense_payload_relative
    samples: list[dict[str, Any]] = []
    for name in selection["dense_tensors"]:
        source = dense_plan[name]
        artifact = dense_records[name]
        require(source["shape"] == artifact["shape"], f"dense shape mismatch: {name}")
        require(source["length"] * 2 == artifact["byte_length"], f"dense byte length mismatch: {name}")
        require(source["shard_id"] == artifact["source_shard_index"], f"dense source shard mismatch: {name}")
        require(source["offset"] == artifact["source_offset"], f"dense source offset mismatch: {name}")
        samples.extend(
            compare_samples(
                "dense", name, source["shape"], source_root / dense_shards[source["shard_id"]]["path"],
                source["offset"], dense_payload, artifact["offset"], source["shard_id"], dense_payload_relative,
            )
        )

    projection_count = 0
    for identity in selection["experts"]:
        expert_key = (identity["layer"], identity["expert"])
        require(expert_key in expert_records, f"selected expert is absent from artifact: {expert_key}")
        artifact_expert = expert_records[expert_key]
        artifact_shard = expert_artifact_shards[artifact_expert["shard_id"]]
        artifact_relative = "experts/" + artifact_shard["path"]
        artifact_path = artifact_root / artifact_relative
        for role in ("gate", "up", "down"):
            projection_count += 1
            source = expert_plan[(identity["layer"], identity["expert"], role)]
            artifact = artifact_expert[role]
            require(source["shape"] == artifact["shape"], f"expert shape mismatch: {source['name']}")
            require(source["length"] * 2 == artifact["length"], f"expert byte length mismatch: {source['name']}")
            samples.extend(
                compare_samples(
                    "expert", source["name"], source["shape"], source_root / expert_shards[source["shard_id"]]["path"],
                    source["offset"], artifact_path, artifact_expert["payload_offset"] + artifact["offset"],
                    source["shard_id"], artifact_relative,
                )
            )

    return {
        "artifact_root_manifest_sha256": registry["root_manifest_sha256"],
        "comparison": "exact BF16 bits shifted left 16 equal stored little-endian F32 bits",
        "dense_tensor_count": len(selection["dense_tensors"]),
        "expert_count": len(selection["experts"]),
        "expert_projection_count": projection_count,
        "model_id": registry["model_id"],
        "revision": registry["revision"],
        "sample_count": len(samples),
        "samples": samples,
        "schema_version": 1,
        "selected_source_shards": verified_shards,
        "status": "passed",
    }


def canonical_json(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_evidence(path: Path, payload: bytes) -> str:
    if path.exists():
        require(path.read_bytes() == payload, "existing evidence differs from deterministic output")
        return "unchanged"
    temporary = path.with_name(f".{path.name}.incomplete")
    require(not temporary.exists(), f"incomplete evidence output exists: {temporary}")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise TensorValueError(f"cannot write evidence: {error}") from error
    return "written"


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dense-plan", type=Path, required=True)
    parser.add_argument("--expert-plan", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    try:
        document = validate(
            args.source_root.resolve(), args.registry.resolve(), args.source_manifest.resolve(),
            args.dense_plan.resolve(), args.expert_plan.resolve(), args.selection.resolve(),
        )
        payload = canonical_json(document)
        disposition = write_evidence(args.output.resolve(), payload)
    except TensorValueError as error:
        print(f"tensor value validation error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"bytes": len(payload), "output": str(args.output), "sha256": hashlib.sha256(payload).hexdigest(), "status": document["status"], "write": disposition}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
