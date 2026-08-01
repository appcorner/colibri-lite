#!/usr/bin/env python3
"""Validate the pre-candidate R1.1a code_newline control calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def fail(message: str) -> None:
    raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upward_f32(value: float) -> tuple[float, str]:
    rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    bits = struct.unpack("<I", struct.pack("<f", rounded))[0]
    if rounded < value:
        bits += 1
        rounded = struct.unpack("<f", struct.pack("<I", bits))[0]
    return rounded, f"0x{bits:08x}"


def validate(record: dict, plan: dict, repository: Path, verify_evidence: bool) -> None:
    if record["schema"] != "colibri-qwen3-moe-m6.3-r1.1a-control-calibration-result-v1":
        fail("schema mismatch")
    if record["status"] != "control_calibrated":
        fail("status mismatch")
    if record["fixture"] != {
        "id": "code_newline",
        "token_ids": [87, 28, 16, 198],
        "checkpoint_payload_sha256": "6a614737eab3c775fac1ed02f3eeabbf5f0d69eb7ab20ba9010bba0c1b9b0978",
    }:
        fail("fixture mismatch")
    if record["candidate_results_seen"] or record["authorizes_candidate_execution"]:
        fail("candidate scope violation")
    if plan["status"] != "pre_registered_not_executed" or plan["independent_release_process_runs"] != 2:
        fail("pre-registration plan mismatch")
    preregistration = record["preregistration"]
    if sha256_file(repository / preregistration["adr"]) != preregistration["adr_sha256"]:
        fail("ADR hash mismatch")
    if sha256_file(repository / preregistration["plan"]) != preregistration["plan_sha256"]:
        fail("plan hash mismatch")

    runs = record["runs"]
    if len(runs) != 2 or len({run["run_id"] for run in runs}) != 2:
        fail("independent run count mismatch")
    if any(run["status"] != "completed" or run["exit_code"] != 0 for run in runs):
        fail("run failure")
    if len({run["checkpoint_sha256"] for run in runs}) != 1:
        fail("checkpoint determinism mismatch")
    for run in runs:
        for stream in ("stdout", "stderr"):
            evidence = run[stream]
            if evidence["bytes"] < 0 or len(evidence["sha256"]) != 64:
                fail(f"invalid {stream} evidence")
        if verify_evidence:
            run_root = Path(r"D:\tmp\colibri-lite-runs") / run["run_id"]
            for stream in ("stdout", "stderr"):
                path = run_root / f"{stream}.log"
                if path.stat().st_size != run[stream]["bytes"] or sha256_file(path) != run[stream]["sha256"]:
                    fail(f"{run['run_id']} {stream} evidence mismatch")

    observed = record["maximum_observed_absolute_errors"]
    if any(not math.isfinite(value) or value < 0.0 for value in observed.values()):
        fail("non-finite observation")
    formulas = {
        "layer0.post_attention_rmsnorm": 3.0 * observed["layer0.post_attention_rmsnorm"] + 5e-7,
        "layer0.router_logits": 3.0 * observed["layer0.post_attention_rmsnorm"] + 1.430511474609375e-5,
        "layer0.routing_weights": 0.5 * observed["layer0.router_logits"] + 1e-7,
        "layer0.selected_expert_output": 3.0 * observed["layer0.selected_expert_output"] + 1e-6,
        "layer0.aggregated_moe_output": 3.0 * observed["layer0.aggregated_moe_output"] + 1e-6,
    }
    budgets = record["derived_f32_budgets"]
    if set(budgets) != set(formulas):
        fail("budget checkpoint set mismatch")
    for checkpoint, unrounded in formulas.items():
        expected_value, expected_bits = upward_f32(unrounded)
        if budgets[checkpoint]["value"] != expected_value or budgets[checkpoint]["bits"] != expected_bits:
            fail(f"{checkpoint} derived budget mismatch")
        if budgets[checkpoint]["value"] <= observed[checkpoint]:
            fail(f"{checkpoint} budget does not cover control")
    guard = record["fixed_logit_guard"]
    if not guard["passed"] or guard["maximum_observed_absolute_error"] > guard["frozen_budget"]:
        fail("fixed-logit guard failed")
    if set(record["exact_gates"].values()) != {"passed", True}:
        fail("exact gate failure")
    final = record["final_control_verification"]
    if (
        final["status"] != "completed"
        or final["exit_code"] != 0
        or final["tests_passed"] != 1
        or final["tests_failed"] != 0
    ):
        fail("final control verification failed")
    for stream in ("stdout", "stderr"):
        evidence = final[stream]
        if evidence["bytes"] < 0 or len(evidence["sha256"]) != 64:
            fail(f"invalid final control {stream} evidence")
        if verify_evidence:
            path = Path(r"D:\tmp\colibri-lite-runs") / final["run_id"] / f"{stream}.log"
            if path.stat().st_size != evidence["bytes"] or sha256_file(path) != evidence["sha256"]:
                fail(f"final control {stream} evidence mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--verify-evidence", action="store_true")
    args = parser.parse_args()
    validate(
        json.loads(args.record.read_text(encoding="utf-8")),
        json.loads(args.plan.read_text(encoding="utf-8")),
        args.repository.resolve(),
        args.verify_evidence,
    )
    print(json.dumps({"status": "passed", "record": str(args.record)}))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"control calibration validation error: {error}")
        raise SystemExit(1) from error
