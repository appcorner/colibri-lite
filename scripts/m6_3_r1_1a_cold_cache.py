#!/usr/bin/env python3
"""Prepare, arm, and consume one-shot M6.3-R1.1a cold-cache evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import stat
import time
from typing import Any


class ColdCacheError(RuntimeError):
    """The cold-cache contract or evidence is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ColdCacheError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    require(not path.exists() and not temporary.exists(), f"output already exists: {path}")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ColdCacheError(f"cannot read JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_contract(contract: dict[str, Any]) -> None:
    require(contract.get("schema") == "m6.3-r1.1a-cold-cache-contract-v1", "contract schema mismatch")
    require(contract.get("schema_version") == 1, "contract version mismatch")
    require(contract.get("status") == "pre_registered_after_invalid_warm_cache_run", "contract status mismatch")
    expected_candidates = {
        "cpu-safe-rust-int8-group64-layer0-r1-1a": (
            641728512,
            "35a3ef6aba723d302fb1a7fded6ede4543a0c3dfcd35651596b78f1c5158cad2",
        ),
        "cpu-safe-rust-int8-group32-layer0-r1-1a": (
            679477248,
            "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2",
        ),
    }
    actual_candidates = {
        item["candidate_id"]: (item["artifact_bytes"], item["artifact_sha256"])
        for item in contract["candidates"]
    }
    require(actual_candidates == expected_candidates, "candidate identity contract mismatch")
    require(contract["prepare_phase"] == {
        "full_sha256_required": True,
        "read_only_required": True,
        "stable_file_identity_required": True,
    }, "prepare contract mismatch")
    require(contract["arm_phase"] == {
        "payload_reads_permitted": False,
        "metadata_identity_only": True,
        "one_shot_authorization": True,
    }, "arm contract mismatch")
    require(contract["reboot_boundary"] == {
        "minimum_boot_time_separation_seconds": 5,
        "maximum_arm_uptime_seconds": 1800,
    }, "reboot boundary contract mismatch")
    require(contract["physical_io_gate"] == {
        "status": "correlated",
        "minimum_candidate_disk_read_bytes": 1,
        "total_events_lost": 0,
    }, "physical I/O gate changed")
    require(contract.get("numerical_contract_changed") is False, "numerical contract changed")
    require(contract.get("authorizes_group32_before_group64_closure") is False, "group-32 scope violation")
    require(contract.get("authorizes_r1_2") is False, "R1.2 scope violation")


def candidate_for(contract: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    candidates = [item for item in contract["candidates"] if item.get("candidate_id") == candidate_id]
    require(len(candidates) == 1, f"unknown or duplicate candidate: {candidate_id}")
    return candidates[0]


def is_read_only(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", None)
    if attributes is not None:
        return bool(attributes & stat.FILE_ATTRIBUTE_READONLY)
    return metadata.st_mode & stat.S_IWUSR == 0


def file_identity(path: Path) -> dict[str, Any]:
    metadata = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": metadata.st_size,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "modified_ns": metadata.st_mtime_ns,
        "read_only": is_read_only(metadata),
    }


def current_boot_marker() -> dict[str, int]:
    require(os.name == "nt", "cold-cache boot evidence is Windows-only")
    uptime_ms = int(ctypes.windll.kernel32.GetTickCount64())
    observed_ns = time.time_ns()
    return {
        "observed_unix_ns": observed_ns,
        "uptime_ms": uptime_ms,
        "boot_time_unix_ns": observed_ns - uptime_ms * 1_000_000,
    }


def validate_flat_path(contract: dict[str, Any], artifact: Path, output: Path) -> Path:
    run_directory = artifact.resolve().parent
    expected_temp = PureWindowsPath(str(Path(contract["temp_root"]).resolve()))
    actual_run = PureWindowsPath(str(run_directory))
    require(actual_run.parent == expected_temp, "artifact is not in one flat run directory")
    require(output.resolve().parent == run_directory, "evidence output must be in the artifact run directory")
    return run_directory


def prepare(
    contract: dict[str, Any], candidate_id: str, artifact: Path, output: Path,
    boot: dict[str, int] | None = None,
) -> dict[str, Any]:
    validate_contract(contract)
    candidate = candidate_for(contract, candidate_id)
    run_directory = validate_flat_path(contract, artifact, output)
    identity = file_identity(artifact)
    require(identity["bytes"] == candidate["artifact_bytes"], "artifact size mismatch")
    require(identity["read_only"], "artifact must be read-only before prepare")
    actual_hash = sha256_file(artifact)
    require(actual_hash == candidate["artifact_sha256"], "artifact hash mismatch")
    document = {
        "schema": "m6.3-r1.1a-cold-cache-prepare-v1",
        "candidate_id": candidate_id,
        "run_directory": str(run_directory),
        "artifact": {**identity, "sha256": actual_hash},
        "boot": boot or current_boot_marker(),
        "status": "prepared_requires_reboot",
    }
    atomic_json(output, document)
    return document


def arm(
    contract: dict[str, Any], prepare_record: dict[str, Any], output: Path,
    boot: dict[str, int] | None = None,
) -> dict[str, Any]:
    validate_contract(contract)
    require(prepare_record.get("schema") == "m6.3-r1.1a-cold-cache-prepare-v1", "prepare schema mismatch")
    require(prepare_record.get("status") == "prepared_requires_reboot", "prepare status mismatch")
    candidate_for(contract, prepare_record["candidate_id"])
    artifact = Path(prepare_record["artifact"]["path"])
    validate_flat_path(contract, artifact, output)
    actual_identity = file_identity(artifact)
    expected_identity = {key: prepare_record["artifact"][key] for key in actual_identity}
    require(actual_identity == expected_identity, "artifact identity changed across reboot")
    require(actual_identity["read_only"], "artifact is no longer read-only")
    current = boot or current_boot_marker()
    reboot = contract["reboot_boundary"]
    separation_ns = abs(current["boot_time_unix_ns"] - prepare_record["boot"]["boot_time_unix_ns"])
    require(separation_ns >= reboot["minimum_boot_time_separation_seconds"] * 1_000_000_000,
            "reboot boundary was not observed")
    require(current["uptime_ms"] <= reboot["maximum_arm_uptime_seconds"] * 1000,
            "arm phase exceeded maximum post-reboot uptime")
    document = {
        "schema": "m6.3-r1.1a-cold-cache-authorization-v1",
        "candidate_id": prepare_record["candidate_id"],
        "run_directory": prepare_record["run_directory"],
        "artifact": actual_identity,
        "prepared_sha256": prepare_record["artifact"]["sha256"],
        "prepare_boot": prepare_record["boot"],
        "arm_boot": current,
        "status": "armed_for_one_etw_launch",
    }
    atomic_json(output, document)
    return document


def verify_launch(
    contract: dict[str, Any], authorization: dict[str, Any], candidate_id: str,
    artifact: Path, authorization_path: Path, consume: bool = False,
    boot: dict[str, int] | None = None,
) -> dict[str, Any]:
    validate_contract(contract)
    candidate = candidate_for(contract, candidate_id)
    require(authorization.get("schema") == "m6.3-r1.1a-cold-cache-authorization-v1", "authorization schema mismatch")
    require(authorization.get("status") == "armed_for_one_etw_launch", "authorization status mismatch")
    require(authorization.get("candidate_id") == candidate_id, "authorization candidate mismatch")
    actual_identity = file_identity(artifact)
    require(actual_identity == authorization["artifact"], "artifact identity changed after arm")
    require(authorization["prepared_sha256"] == candidate["artifact_sha256"], "prepared hash mismatch")
    require(authorization_path.resolve().parent == artifact.resolve().parent, "authorization is outside run directory")
    current = boot or current_boot_marker()
    arm_boot = authorization["arm_boot"]
    boot_drift_ns = abs(current["boot_time_unix_ns"] - arm_boot["boot_time_unix_ns"])
    require(boot_drift_ns < contract["reboot_boundary"]["minimum_boot_time_separation_seconds"] * 1_000_000_000,
            "launch is not in the arm boot session")
    require(current["uptime_ms"] >= arm_boot["uptime_ms"], "launch uptime precedes arm uptime")
    consumed_path = authorization_path.with_name(authorization_path.name + ".consumed.json")
    require(not consumed_path.exists(), "cold-cache authorization was already consumed")
    result = {"status": "valid", "consumed": consume, "consumed_path": str(consumed_path)}
    if consume:
        atomic_json(consumed_path, {
            "schema": "m6.3-r1.1a-cold-cache-consumption-v1",
            "candidate_id": candidate_id,
            "artifact": str(artifact.resolve()),
            "boot": current,
            "status": "consumed_before_etw_launch",
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "arm", "verify-launch"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--prepare-record", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--consume", action="store_true")
    args = parser.parse_args()
    contract = load_json(args.contract)
    if args.command == "prepare":
        require(args.candidate_id and args.artifact and args.output, "prepare arguments are incomplete")
        result = prepare(contract, args.candidate_id, args.artifact, args.output)
    elif args.command == "arm":
        require(args.prepare_record and args.output, "arm arguments are incomplete")
        result = arm(contract, load_json(args.prepare_record), args.output)
    else:
        require(args.candidate_id and args.artifact and args.authorization, "verify-launch arguments are incomplete")
        result = verify_launch(
            contract, load_json(args.authorization), args.candidate_id, args.artifact,
            args.authorization, args.consume,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ColdCacheError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"cold-cache protocol error: {error}")
        raise SystemExit(1) from error
