#!/usr/bin/env python3
"""Validate a recorded M6.1-03 storage benchmark evidence file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SCHEMA = "colibri-lite-m6.1-03-storage-benchmark-v1"


def validate_distribution(value: object, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path} must be an object"]
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) < 3:
        return [f"{path}.samples must contain at least three values"]
    values = [*samples, *(value.get(key) for key in ("minimum", "p10", "median", "p90", "maximum"))]
    if not all(isinstance(item, (int, float)) and math.isfinite(item) and item > 0 for item in values):
        return [f"{path} values must be finite and positive"]
    if samples != sorted(samples):
        return [f"{path}.samples must be sorted ascending"]
    if not value["minimum"] <= value["p10"] <= value["median"] <= value["p90"] <= value["maximum"]:
        return [f"{path} percentile ordering is invalid"]
    return []


def validate(document: object) -> list[str]:
    if not isinstance(document, dict):
        return ["document must be a JSON object"]
    errors: list[str] = []
    if document.get("schema") != SCHEMA:
        errors.append(f"schema must equal {SCHEMA}")
    if document.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    for key in ("profile_id", "created_at", "runtime", "storage_target", "preflight", "test_payload", "cache_semantics", "results", "cleanup"):
        if not document.get(key):
            errors.append(f"{key} is required")
    if document.get("cleanup", {}).get("run_directory_removed") is not True:
        errors.append("cleanup.run_directory_removed must be true")
    semantics = document.get("cache_semantics")
    if isinstance(semantics, dict) and "not claimed as a cold-device" not in semantics.get("first_touch", ""):
        errors.append("cache_semantics.first_touch must not claim a controlled cold-device read")
    results = document.get("results")
    if not isinstance(results, dict):
        return errors + ["results must be an object"]
    sequential = results.get("sequential_read")
    random_read = results.get("expert_sized_random_read")
    if not isinstance(sequential, dict) or not isinstance(random_read, dict):
        return errors + ["sequential_read and expert_sized_random_read are required"]
    if not isinstance(sequential.get("first_touch_throughput"), (int, float)) or sequential["first_touch_throughput"] <= 0:
        errors.append("sequential_read.first_touch_throughput must be positive")
    errors.extend(validate_distribution(sequential.get("warm_throughput"), "sequential_read.warm_throughput"))
    for name in ("first_touch_latency", "first_touch_throughput", "warm_latency", "warm_throughput"):
        errors.extend(validate_distribution(random_read.get(name), f"expert_sized_random_read.{name}"))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    arguments = parser.parse_args()
    errors = validate(json.loads(arguments.input.read_text(encoding="utf-8")))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"M6.1-03 storage benchmark evidence is valid: {arguments.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
