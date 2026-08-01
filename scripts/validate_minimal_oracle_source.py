#!/usr/bin/env python3
"""Validate the closed 17-file minimal Safetensors oracle source subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


MINIMAL_SOURCE_TOTAL_BYTES = 61_068_275_406


class MinimalOracleSourceError(RuntimeError):
    """The minimal source root is not the verified pinned subset."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MinimalOracleSourceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_without_duplicate_keys(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, json.JSONDecodeError) as error:
        raise MinimalOracleSourceError(f"cannot parse {path.name}: {error}") from error
    require(isinstance(value, dict), f"{path.name} must be a JSON object")
    return value


def expected_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = [
        record
        for record in manifest.get("files", [])
        if record.get("path") == "model.safetensors.index.json"
        or str(record.get("path", "")).endswith(".safetensors")
    ]
    require(len(records) == 17, "manifest must define exactly one index and 16 shards")
    names = [record.get("path") for record in records]
    require(len(names) == len(set(names)), "manifest has duplicate minimal-source paths")
    return records


def validate(source_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = parse_json_without_duplicate_keys(manifest_path)
    records = expected_records(manifest)
    expected = {record["path"]: record for record in records}
    require(source_root.is_dir(), "source root is not a directory")
    entries = list(source_root.iterdir())
    require(not any(entry.is_dir() for entry in entries), "source root must not contain child directories")
    actual = {entry.name for entry in entries if entry.is_file()}
    require(actual == set(expected), "source root has missing or unexpected files")

    total_bytes = 0
    for name, record in expected.items():
        path = source_root / name
        require(path.stat().st_size == record["bytes"], f"size mismatch: {name}")
        require(sha256_file(path) == record["sha256"], f"SHA-256 mismatch: {name}")
        total_bytes += record["bytes"]
    require(total_bytes == MINIMAL_SOURCE_TOTAL_BYTES, "minimal-source total bytes mismatch")

    index_name = manifest["safetensors"]["index_file"]
    index = parse_json_without_duplicate_keys(source_root / index_name)
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and weight_map, "index has no tensor weight_map")
    require(len(weight_map) == manifest["safetensors"]["tensor_count"], "index tensor count mismatch")
    referenced = set(weight_map.values())
    require(all(isinstance(name, str) for name in referenced), "index has non-string shard reference")
    require(
        all(Path(name).name == name and "/" not in name and "\\" not in name and ".." not in name for name in referenced),
        "index contains unsafe shard path",
    )
    expected_shards = set(manifest["safetensors"]["shards"])
    require(referenced == expected_shards, "index shard references differ from manifest")
    require(len(referenced) == 16, "index must reference exactly 16 shards")
    require(actual - {index_name} == referenced, "source root has unreferenced shard")

    model = manifest["model"]
    require(model["id"] == "Qwen/Qwen3-30B-A3B", "unexpected model ID")
    require(model["revision"] == "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "unexpected revision")
    require(model["license"] == "Apache-2.0", "unexpected license")
    require(model["architecture"] == "Qwen3MoeForCausalLM", "unexpected architecture")
    return {
        "file_count": len(actual), "model_id": model["id"], "revision": model["revision"],
        "total_bytes": total_bytes, "tensor_mapping_count": len(weight_map),
        "referenced_shard_count": len(referenced), "status": "passed",
    }


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("source_manifest", type=Path)
    args = parser.parse_args(arguments)
    try:
        print(json.dumps(validate(args.source_root.resolve(), args.source_manifest.resolve()), sort_keys=True))
    except MinimalOracleSourceError as error:
        print(f"minimal oracle source validation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
