#!/usr/bin/env python3
"""Validate M6.1-04 RAM/GPU discovery evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA = "colibri-lite-m6.1-04-memory-gpu-profile-v1"


def validate(document: object) -> list[str]:
    if not isinstance(document, dict):
        return ["document must be a JSON object"]
    errors: list[str] = []
    if document.get("schema") != SCHEMA:
        errors.append(f"schema must equal {SCHEMA}")
    if document.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    for key in ("profile_id", "created_at", "runtime", "ram", "adapters", "backends", "recommendations"):
        if key not in document:
            errors.append(f"{key} is required")
    ram = document.get("ram")
    if isinstance(ram, dict):
        for key in ("total_physical_bytes", "available_physical_bytes", "safety_reserve_bytes", "usable_budget_bytes"):
            if not isinstance(ram.get(key), int) or ram[key] < 0:
                errors.append(f"ram.{key} must be a non-negative integer")
        if ram.get("usable_budget_bytes", 0) > ram.get("available_physical_bytes", 0):
            errors.append("ram.usable_budget_bytes cannot exceed available_physical_bytes")
    backends = document.get("backends")
    if not isinstance(backends, list) or not backends:
        errors.append("backends must be a non-empty list")
        return errors
    for backend in backends:
        if not isinstance(backend, dict):
            errors.append("backend must be an object")
            continue
        if backend.get("availability") not in {"unavailable", "not_run"}:
            errors.append("M6.1-04 must not claim a usable backend before M6.3 review")
        if backend.get("usable_vram_bytes") != 0:
            errors.append("usable_vram_bytes must be zero without a usable backend")
        for direction in ("host_to_device", "device_to_host"):
            benchmark = backend.get(direction)
            if not isinstance(benchmark, dict) or benchmark.get("status") != "not_run" or not benchmark.get("reason"):
                errors.append(f"{backend.get('backend_id', 'backend')}.{direction} must be explicit not_run with a reason")
    recommendations = document.get("recommendations")
    if isinstance(recommendations, dict) and recommendations.get("vram_budget_bytes") != 0:
        errors.append("recommendations.vram_budget_bytes must be zero without a usable backend")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    arguments = parser.parse_args()
    errors = validate(json.loads(arguments.input.read_text(encoding="utf-8-sig")))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"M6.1-04 memory/GPU evidence is valid: {arguments.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
