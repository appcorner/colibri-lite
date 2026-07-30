#!/usr/bin/env python3
"""Validate a recorded M6.1-02 CPU/RAM microbenchmark evidence file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SCHEMA = "colibri-lite-m6.1-02-cpu-ram-benchmark-v1"


def validate(document: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["document must be a JSON object"]
    if document.get("schema") != SCHEMA:
        errors.append(f"schema must equal {SCHEMA}")
    if document.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    for key in ("profile_id", "created_at", "runtime", "host", "measurement_semantics", "results"):
        if not document.get(key):
            errors.append(f"{key} is required")

    results = document.get("results")
    if not isinstance(results, dict):
        return errors + ["results must be an object"]
    for result_name in ("cpu_kernel_gflops", "ram_copy_gib_per_second"):
        distribution = results.get(result_name)
        if not isinstance(distribution, dict):
            errors.append(f"results.{result_name} must be an object")
            continue
        samples = distribution.get("samples")
        if not isinstance(samples, list) or len(samples) < 3:
            errors.append(f"results.{result_name}.samples must contain at least three values")
            continue
        values = [*samples, *(distribution.get(key) for key in ("minimum", "p10", "median", "p90", "maximum"))]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in values):
            errors.append(f"results.{result_name} values must be finite and positive")
            continue
        if samples != sorted(samples):
            errors.append(f"results.{result_name}.samples must be sorted ascending")
        if not (distribution["minimum"] <= distribution["p10"] <= distribution["median"] <= distribution["p90"] <= distribution["maximum"]):
            errors.append(f"results.{result_name} percentile ordering is invalid")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    arguments = parser.parse_args()
    document = json.loads(arguments.input.read_text(encoding="utf-8"))
    errors = validate(document)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"M6.1-02 benchmark evidence is valid: {arguments.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
