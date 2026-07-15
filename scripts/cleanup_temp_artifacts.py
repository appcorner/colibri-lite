#!/usr/bin/env python3
"""Safely audit or apply a reviewed temporary-artifact cleanup plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable


class CleanupError(RuntimeError):
    """The cleanup plan is unsafe, stale, or invalid."""


PLAN_KEYS = {
    "canonical_artifact_root",
    "candidates",
    "generated_at",
    "protected_paths",
    "schema_version",
    "temp_root",
}
CANDIDATE_KEYS = {
    "classification",
    "expected_file_count",
    "expected_logical_bytes",
    "expected_reclaimable_bytes",
    "expected_shared_hardlink_bytes",
    "kind",
    "path",
}
DELETABLE_CLASSIFICATIONS = {
    "completed-task temporary output",
    "incomplete/orphaned temporary output",
    "reproducibility evidence",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CleanupError(message)


def strict_keys(value: Any, expected: set[str], context: str) -> None:
    require(isinstance(value, dict), f"{context} must be an object")
    actual = set(value)
    require(actual == expected, f"{context} fields differ: expected {sorted(expected)}, got {sorted(actual)}")


def resolved(value: str) -> Path:
    require(isinstance(value, str) and value != "", "plan path must be a non-empty string")
    return Path(value).resolve()


def overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def files_under(path: Path) -> list[Path]:
    is_junction = getattr(path, "is_junction", lambda: False)()
    require(not path.is_symlink() and not is_junction, f"cleanup candidate itself is a link/reparse directory: {path}")
    if path.is_file():
        return [path]
    require(path.is_dir(), f"candidate is neither a file nor directory: {path}")
    entries = list(path.rglob("*"))
    for entry in entries:
        is_junction = getattr(entry, "is_junction", lambda: False)()
        require(not entry.is_symlink() and not is_junction, f"candidate contains a link/reparse directory: {entry}")
    return [entry for entry in entries if entry.is_file()]


def inventory(path: Path) -> dict[str, int]:
    files = files_under(path)
    logical_bytes = 0
    reclaimable_bytes = 0
    shared_hardlink_bytes = 0
    for file in files:
        metadata = file.stat()
        logical_bytes += metadata.st_size
        if metadata.st_nlink == 1:
            reclaimable_bytes += metadata.st_size
        else:
            shared_hardlink_bytes += metadata.st_size
    return {
        "file_count": len(files),
        "logical_bytes": logical_bytes,
        "reclaimable_bytes": reclaimable_bytes,
        "shared_hardlink_bytes": shared_hardlink_bytes,
    }


def manifest_file_paths(canonical_root: Path) -> list[Path]:
    manifest_path = canonical_root / "model-manifest-v1.json"
    require(manifest_path.is_file(), f"canonical root manifest is missing: {manifest_path}")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CleanupError(f"cannot read canonical root manifest: {error}") from error
    records: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if set(value) >= {"bytes", "path", "sha256"}:
                records.append(value["path"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    require(records, "canonical root manifest contains no file records")
    output = [manifest_path.resolve()]
    for value in records:
        require(isinstance(value, str) and value != "", "canonical manifest contains an invalid path")
        path = (canonical_root / Path(value)).resolve()
        require(path == canonical_root or canonical_root in path.parents, f"canonical manifest path escapes root: {value}")
        require(path.is_file(), f"canonical referenced file is missing: {value}")
        output.append(path)
    return output


def load_plan(path: Path) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CleanupError(f"cannot read cleanup plan: {error}") from error
    strict_keys(plan, PLAN_KEYS, "cleanup plan")
    require(plan["schema_version"] == 1, "unsupported cleanup plan version")
    require(isinstance(plan["candidates"], list) and plan["candidates"], "cleanup plan has no candidates")
    require(isinstance(plan["protected_paths"], list), "protected_paths must be an array")
    return plan


def validate_plan(plan: dict[str, Any]) -> tuple[Path, Path, list[dict[str, Any]]]:
    temp_root = resolved(plan["temp_root"])
    canonical_root = resolved(plan["canonical_artifact_root"])
    require(temp_root.is_dir(), f"temporary root is missing: {temp_root}")
    require(canonical_root.is_dir(), f"canonical artifact root is missing: {canonical_root}")
    require(temp_root == canonical_root or temp_root in canonical_root.parents, "canonical root must be under the temporary volume root")
    protected = [canonical_root, *[resolved(value) for value in plan["protected_paths"]]]
    for path in protected:
        require(path.exists(), f"protected path is missing: {path}")
    referenced = manifest_file_paths(canonical_root)

    validated: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    for index, candidate in enumerate(plan["candidates"]):
        strict_keys(candidate, CANDIDATE_KEYS, f"candidate {index}")
        require(candidate["classification"] in DELETABLE_CLASSIFICATIONS, f"candidate {index} classification is not deletable")
        path = resolved(candidate["path"])
        require(path.exists(), f"cleanup candidate is missing: {path}")
        require(temp_root in path.parents, f"cleanup candidate is outside the temporary root: {path}")
        require(path != temp_root, "cleanup candidate cannot be the temporary root")
        require(candidate["kind"] in ("file", "directory"), f"candidate {index} kind is invalid")
        require((candidate["kind"] == "file") == path.is_file(), f"candidate {index} kind does not match filesystem")
        for protected_path in protected:
            require(not overlaps(path, protected_path), f"cleanup candidate overlaps protected path: {path}")
        for referenced_file in referenced:
            require(not overlaps(path, referenced_file), f"cleanup candidate overlaps canonical referenced file: {path}")
        candidate_paths.append(path)
        actual = inventory(path)
        require(actual["file_count"] == candidate["expected_file_count"], f"candidate file count drifted: {path}")
        require(actual["logical_bytes"] == candidate["expected_logical_bytes"], f"candidate logical bytes drifted: {path}")
        require(actual["reclaimable_bytes"] == candidate["expected_reclaimable_bytes"], f"candidate reclaimable bytes drifted: {path}")
        require(actual["shared_hardlink_bytes"] == candidate["expected_shared_hardlink_bytes"], f"candidate hard-link bytes drifted: {path}")
        validated.append({**candidate, **actual, "resolved_path": str(path)})

    require(len(candidate_paths) == len(set(candidate_paths)), "cleanup candidates are duplicated")
    for index, first in enumerate(candidate_paths):
        for second in candidate_paths[index + 1 :]:
            require(not overlaps(first, second), f"cleanup candidates overlap or are nested: {first} and {second}")
    return temp_root, canonical_root, validated


def delete_candidate(path: Path) -> None:
    if path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def execute(plan_path: Path, apply: bool = False) -> dict[str, Any]:
    plan = load_plan(plan_path)
    temp_root, canonical_root, candidates = validate_plan(plan)
    free_before = shutil.disk_usage(temp_root).free
    totals = {
        "file_count": sum(candidate["file_count"] for candidate in candidates),
        "logical_bytes": sum(candidate["logical_bytes"] for candidate in candidates),
        "reclaimable_bytes": sum(candidate["reclaimable_bytes"] for candidate in candidates),
        "shared_hardlink_bytes": sum(candidate["shared_hardlink_bytes"] for candidate in candidates),
    }
    if apply:
        for candidate in candidates:
            delete_candidate(Path(candidate["resolved_path"]))
        for candidate in candidates:
            require(not Path(candidate["resolved_path"]).exists(), f"candidate still exists after cleanup: {candidate['resolved_path']}")
        require(canonical_root.is_dir(), "canonical artifact root disappeared during cleanup")
        manifest_file_paths(canonical_root)
    free_after = shutil.disk_usage(temp_root).free
    return {
        "apply": apply,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "canonical_artifact_root": str(canonical_root),
        "disk_free_after": free_after,
        "disk_free_before": free_before,
        "disk_free_delta": free_after - free_before,
        "mode": "apply" if apply else "dry-run",
        "temp_root": str(temp_root),
        "totals": totals,
    }


def parse_arguments(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="reviewed JSON cleanup plan")
    parser.add_argument("--apply", action="store_true", help="perform deletion; default is dry-run")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    args = parse_arguments(arguments)
    try:
        result = execute(args.plan.resolve(), args.apply)
    except CleanupError as error:
        print(f"cleanup refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
