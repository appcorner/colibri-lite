#!/usr/bin/env python3
"""Validate the pinned Safetensors index and shards against source provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


class SourceValidationError(RuntimeError):
    """Pinned source files are missing or differ from provenance."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(source_root: Path, manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceValidationError(f"cannot read source manifest: {error}") from error
    require(manifest.get("schema_version") == 1, "unsupported source manifest version")
    records = [
        record
        for record in manifest.get("files", [])
        if record.get("path") == "model.safetensors.index.json"
        or str(record.get("path", "")).endswith(".safetensors")
    ]
    require(len(records) == 17, "source manifest must identify one index and 16 shards")
    total_bytes = 0
    for record in records:
        relative = record.get("path")
        require(isinstance(relative, str) and Path(relative).name == relative, "invalid source file path")
        path = source_root / relative
        require(path.is_file(), f"pinned source file is missing: {relative}")
        expected_bytes = record.get("bytes")
        expected_hash = record.get("sha256")
        require(path.stat().st_size == expected_bytes, f"pinned source size mismatch: {relative}")
        require(sha256_file(path) == expected_hash, f"pinned source hash mismatch: {relative}")
        total_bytes += expected_bytes
    return {
        "file_count": len(records),
        "model_id": manifest["model"]["id"],
        "revision": manifest["model"]["revision"],
        "source_bytes": total_bytes,
        "source_root": str(source_root.resolve()),
        "status": "passed",
    }


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("source_manifest", type=Path)
    args = parser.parse_args(arguments)
    try:
        result = validate(args.source_root.resolve(), args.source_manifest.resolve())
    except SourceValidationError as error:
        print(f"source validation error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
