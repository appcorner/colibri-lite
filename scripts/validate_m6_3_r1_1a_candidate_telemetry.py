#!/usr/bin/env python3
"""Validate the pre-registered R1.1a telemetry contract and run records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath


def fail(message: str) -> None:
    raise ValueError(message)


def valid_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def candidates(contract: dict) -> dict[str, dict]:
    return {item["candidate_id"]: item for item in contract["candidates"]}


def validate_contract(contract: dict) -> None:
    if contract["schema"] != "m6.3-r1.1a-candidate-telemetry-contract-v1":
        fail("contract schema mismatch")
    if contract["status"] != "pre_registered_not_executed":
        fail("contract status mismatch")
    if contract["fixture"] != {"id": "code_newline", "token_ids": [87, 28, 16, 198]}:
        fail("fixture mismatch")
    expected = {
        "cpu-safe-rust-int8-group64-layer0-r1-1a": (64, 641728512, "35a3ef6aba723d302fb1a7fded6ede4543a0c3dfcd35651596b78f1c5158cad2", 5013504),
        "cpu-safe-rust-int8-group32-layer0-r1-1a": (32, 679477248, "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2", 5308416),
    }
    actual = candidates(contract)
    if set(actual) != set(expected):
        fail("candidate set mismatch")
    for candidate_id, values in expected.items():
        item = actual[candidate_id]
        if (item["group_size"], item["artifact_bytes"], item["artifact_sha256"], item["packed_expert_bytes"]) != values:
            fail(f"{candidate_id} identity mismatch")
    if contract["execution"]["independent_runs_per_candidate"] != 3:
        fail("run count mismatch")
    if contract["required_metrics"]["complete_f32_weight_materializations"] != 0:
        fail("F32 materialization contract mismatch")
    if contract["required_metrics"]["sample_interval_ms"] != 100:
        fail("sample interval mismatch")
    gate = contract["physical_io_gate"]
    if not gate["required_for_admission"] or gate["minimum_candidate_disk_read_bytes"] != 1 or gate["total_events_lost"] != 0:
        fail("physical I/O contract mismatch")
    if any(contract[key] for key in ("candidate_results_seen", "authorizes_candidate_execution", "authorizes_r1_2")):
        fail("pre-execution scope violation")


def validate_run(record: dict, contract: dict) -> None:
    validate_contract(contract)
    if record["schema"] != "m6.3-r1.1a-candidate-run-telemetry-v1":
        fail("run schema mismatch")
    candidate = candidates(contract).get(record["candidate_id"])
    if candidate is None:
        fail("unknown candidate")
    if record["fixture"] != contract["fixture"]:
        fail("run fixture mismatch")
    run_root = PureWindowsPath(record["run_directory"])
    artifact_path = PureWindowsPath(record["artifact"]["path"])
    if artifact_path.parent != run_root or run_root.parent.name != "colibri-lite-runs":
        fail("artifact is not inside one flat run directory")
    artifact = record["artifact"]
    if (artifact["group_size"], artifact["bytes"], artifact["sha256"]) != (
        candidate["group_size"], candidate["artifact_bytes"], candidate["artifact_sha256"]
    ):
        fail("artifact identity mismatch")
    process = record["persisted_process"]
    if process["status"] != "completed" or process["exit_code"] != 0 or process["child_pid"] <= 0:
        fail("persisted process failure")
    if not valid_hash(process["stdout_sha256"]) or not valid_hash(process["stderr_sha256"]):
        fail("persisted log hash mismatch")
    direct = record["direct_consumption"]
    if direct["verification_bytes_read"] != candidate["artifact_bytes"]:
        fail("verification byte accounting mismatch")
    if direct["payload_bytes_read"] <= 0 or direct["payload_bytes_read"] % candidate["packed_expert_bytes"] != 0:
        fail("payload byte accounting mismatch")
    if direct["peak_packed_expert_bytes"] != candidate["packed_expert_bytes"]:
        fail("packed expert peak mismatch")
    if direct["complete_f32_weight_materializations"] != 0:
        fail("complete F32 weight materialization")
    cold = record["cold_cache"]
    if cold["status"] != "consumed_before_etw_launch" or cold["authorization_reused"]:
        fail("cold-cache authorization mismatch")
    if cold["prelaunch_payload_reads"] != 0 or cold["prepared_sha256"] != artifact["sha256"]:
        fail("cold-cache artifact preparation mismatch")
    if PureWindowsPath(cold["authorization_path"]).parent != run_root:
        fail("cold-cache authorization is outside flat run directory")
    if PureWindowsPath(cold["consumption_path"]).parent != run_root:
        fail("cold-cache consumption evidence is outside flat run directory")
    if cold["boot_time_separation_seconds"] < 5:
        fail("cold-cache reboot boundary missing")
    memory = record["process_memory"]
    if memory["samples"] < 1 or memory["sample_interval_ms"] != 100:
        fail("memory sampling mismatch")
    if memory["working_set_peak_bytes"] <= 0 or memory["private_bytes_peak"] <= 0:
        fail("memory peak missing")
    physical = record["physical_io"]
    if physical["status"] != "correlated" or physical["pid"] != process["child_pid"]:
        fail("physical I/O PID correlation mismatch")
    if PureWindowsPath(physical["artifact_path"]) != artifact_path or physical["candidate_disk_read_bytes"] < 1:
        fail("physical I/O artifact correlation mismatch")
    if physical["total_events_lost"] != 0:
        fail("physical I/O event loss")
    for checkpoint in ("layer0_checkpoint_sha256", "final_logits_sha256"):
        if not valid_hash(record[checkpoint]):
            fail(f"invalid {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("run", type=Path, nargs="?")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    validate_contract(contract)
    if args.run:
        validate_run(json.loads(args.run.read_text(encoding="utf-8")), contract)
    print(json.dumps({"status": "passed", "run_validated": args.run is not None}))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"candidate telemetry validation error: {error}")
        raise SystemExit(1) from error
