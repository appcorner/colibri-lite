"""Validator for the M6.3-R1.0 release telemetry record."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def validate(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if document.get("schema") != "m6.3-r1-telemetry-v1":
        errors.append("schema must be m6.3-r1-telemetry-v1")
    runtime = document.get("runtime", {})
    if runtime.get("reference_identity") != "reference-f32-v1":
        errors.append("runtime reference identity is not reference-f32-v1")
    collector = document.get("collector", {})
    if collector.get("sample_interval_ms") != 100:
        errors.append("sample interval must be 100 ms")
    runs = document.get("runs", [])
    if len(runs) < 5:
        errors.append("five reference runs are required")
    for index, run in enumerate(runs):
        prefix = f"runs[{index}]"
        if run.get("exit_code") != 0:
            errors.append(f"{prefix} failed with exit_code={run.get('exit_code')}")
        if run.get("samples", 0) < 1:
            errors.append(f"{prefix} has no independent memory samples")
        memory = run.get("memory", {})
        if memory.get("working_set_peak_bytes", 0) <= 0:
            errors.append(f"{prefix} missing working-set peak")
        if memory.get("peak_working_set_bytes", 0) < memory.get("working_set_peak_bytes", 0):
            errors.append(f"{prefix} peak working set is below sampled peak")
        if run.get("logical_io", {}).get("source") != "runtime_accounting":
            errors.append(f"{prefix} logical I/O source is not runtime_accounting")
        if run.get("explicit_memory", {}).get("source") != "runtime_accounting":
            errors.append(f"{prefix} explicit memory source is not runtime_accounting")
        if run.get("cache_state", {}).get("os_filesystem") != "uncontrolled":
            errors.append(f"{prefix} must label OS filesystem cache uncontrolled")
    etw = collector.get("etw", {})
    if etw.get("status") != "correlated":
        errors.append("process-correlated ETW file/disk I/O is not measured")
    gates = document.get("gates", {})
    if not all(gates.get(name) is True for name in ("five_reference_runs", "collector_reconciliation", "physical_io")):
        errors.append("one or more R1.0 gates failed")
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_m6_3_r1_0_telemetry.py RECORD.json", file=sys.stderr)
        return 2
    document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    errors = validate(document)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("m6.3-r1.0 telemetry valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
