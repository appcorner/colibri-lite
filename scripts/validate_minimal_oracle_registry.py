#!/usr/bin/env python3
"""Validate the tracked canonical minimal-oracle registry against its root."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from validate_minimal_oracle_source import validate


def canonical_root_set(records: list[dict[str, object]]) -> str:
    payload = (json.dumps(sorted(records, key=lambda record: str(record["path"])), separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("source_manifest", type=Path)
    args = parser.parse_args()
    try:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        if registry.get("schema_version") != 1 or registry.get("full_transformers_snapshot") is not False:
            raise ValueError("invalid registry version or snapshot claim")
        records = registry.get("files")
        if not isinstance(records, list) or len(records) != 17:
            raise ValueError("registry must list exactly 17 files")
        if canonical_root_set(records) != registry.get("root_set_sha256"):
            raise ValueError("registry root-set hash mismatch")
        result = validate(Path(registry["stable_root"]), args.source_manifest)
        if result["total_bytes"] != registry.get("total_bytes"):
            raise ValueError("registry total bytes mismatch")
        print(json.dumps({"registry": str(args.registry), "status": "passed", **result}, sort_keys=True))
    except (OSError, ValueError, KeyError) as error:
        print(f"minimal oracle registry validation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
