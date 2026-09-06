#!/usr/bin/env python3
"""Durable, no-automatic-retry orchestrator for official M6.3-R2.3b."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from validate_m6_3_r2_3b import (
    CONTRACT_SHA256,
    GROUPS,
    LAYERS,
    REFERENCE_SHA256,
    ROOT_MANIFEST_SHA256,
    SUBSETS,
    selected_candidate,
    validate,
)

TEST_FILTER = "full_model_validation_tests::r2_3b_characterization_tests::m6_3_r2_3b_characterize_layer_group_candidates"
BOOL_FIELDS = ("same_input_pass", "sequence_pass", "candidate_finite", "prompt_first_token_exact")
INT_FIELDS = (
    "layer", "group_size", "subset_index", "packed_count", "logical_expert_bytes",
    "logical_layer_bytes", "logical_bytes_saved", "artifact_bytes",
)
FLOAT_FIELDS = (
    "same_input_english_max_abs", "same_input_thai_max_abs", "same_input_max_abs",
    "sequence_english_max_abs", "sequence_thai_max_abs", "sequence_max_abs",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    require(not temporary.exists(), f"stale incomplete output: {temporary}")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def verify_execution(repo: Path, artifact_root: Path, reference: Path, binary: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == "colibri-m6.3-r2.3b-execution-manifest-v1", "execution manifest schema mismatch")
    identities = manifest["identities"]
    require(identities["implementation_contract_sha256"] == CONTRACT_SHA256, "execution contract mismatch")
    require(identities["r2_3a_reference_sha256"] == REFERENCE_SHA256, "execution reference mismatch")
    require(identities["canonical_root_manifest_sha256"] == ROOT_MANIFEST_SHA256, "execution root mismatch")
    require(sha256(repo / identities["implementation_contract_path"]) == CONTRACT_SHA256, "contract bytes mismatch")
    require(reference == (repo / identities["r2_3a_reference_path"]).resolve(), "reference path differs from manifest")
    require(sha256(reference) == REFERENCE_SHA256, "R2.3a reference bytes mismatch")
    require(sha256(artifact_root / "model-manifest-v1.json") == ROOT_MANIFEST_SHA256, "canonical root manifest bytes mismatch")
    for record in manifest["files"]:
        path = repo / record["path"]
        require(path.stat().st_size == record["bytes"], f"machinery size mismatch: {record['path']}")
        require(sha256(path) == record["sha256"], f"machinery hash mismatch: {record['path']}")
    binary_record = manifest["release_test_binary"]
    require(binary.name == binary_record["name"], "release binary name mismatch")
    require(binary.stat().st_size == binary_record["bytes"], "release binary size mismatch")
    require(sha256(binary) == binary_record["sha256"], "release binary hash mismatch")
    require(manifest["grid"]["layers"] == list(LAYERS), "execution layer grid mismatch")
    require(manifest["grid"]["groups"] == list(GROUPS), "execution group grid mismatch")
    require(manifest["grid"]["subsets"] == list(SUBSETS), "execution subset grid mismatch")
    require(manifest["grid"]["candidate_count"] == 945, "execution candidate count mismatch")
    require(manifest["gates"] == {
        "same_input_max_abs": 0.001,
        "sequence_aware_bilingual_max_abs": 0.001,
        "generated_tokens_propagated": 1,
        "automatic_retry": False,
    }, "execution gates mismatch")
    require(manifest["test_filter"] == TEST_FILTER, "execution test filter mismatch")
    return manifest


def parse_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == 7, "R2.3b harness must emit seven subset rows")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for name in INT_FIELDS:
            item[name] = int(item[name])
        for name in FLOAT_FIELDS:
            item[name] = float(item[name])
        for name in BOOL_FIELDS:
            require(item[name] in ("true", "false"), f"invalid boolean field {name}")
            item[name] = item[name] == "true"
        item["canonical_f32_control_exact"] = True
        parsed.append(item)
    require([row["subset"] for row in parsed] == list(SUBSETS), "R2.3b harness subset order mismatch")
    require([row["subset_index"] for row in parsed] == list(range(7)), "R2.3b harness subset indices mismatch")
    return parsed


def decision(layer: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = selected_candidate(rows)
    result: dict[str, Any] = {
        "layer": layer,
        "candidates_evaluated": len(rows),
        "candidates_passed": sum(
            row["same_input_pass"] and row["sequence_pass"] and row["candidate_finite"]
            and row["canonical_f32_control_exact"]
            for row in rows
        ),
        "worst_same_input_max_abs": max(row["same_input_max_abs"] for row in rows),
        "worst_sequence_max_abs": max(row["sequence_max_abs"] for row in rows),
    }
    result["candidates_failed"] = len(rows) - result["candidates_passed"]
    if selected is None:
        result.update(
            selected_policy="canonical_f32", selected_candidate_id=None, group_size=None,
            subset=None, logical_bytes_saved=0, artifact_sha256=None,
        )
    else:
        result.update(
            selected_policy="packed_subset",
            selected_candidate_id=f"layer{layer:02d}-group{selected['group_size']}-{selected['subset']}",
            group_size=selected["group_size"], subset=selected["subset"],
            logical_bytes_saved=selected["logical_bytes_saved"], artifact_sha256=selected["artifact_sha256"],
        )
    return result


def completed_prefix(state: dict[str, Any]) -> list[tuple[int, int]]:
    completed = [(item["layer"], item["group_size"]) for item in state.get("completed_runs", [])]
    expected = [(layer, group) for layer in LAYERS for group in GROUPS]
    require(completed == expected[: len(completed)], "completed runs are not an exact frozen prefix")
    return completed


def write_failure(root: Path, state: dict[str, Any], phase: str, reason: str) -> None:
    state["status"] = "transport_blocked"
    state["failure"] = {"phase": phase, "reason": reason, "time_unix": time.time()}
    atomic_json(root / "state.json", state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    artifact_root = args.artifact_root.resolve()
    reference = args.reference.resolve()
    binary = args.binary.resolve()
    manifest_path = args.execution_manifest.resolve()
    root = args.output_root.resolve()
    require(sys.platform == "win32", "official R2.3b execution requires Windows")
    manifest = verify_execution(repo, artifact_root, reference, binary, manifest_path)
    manifest_sha = sha256(manifest_path)
    if args.validate_only:
        print(json.dumps({"status": "passed", "execution_manifest_sha256": manifest_sha, "candidates": 945}, sort_keys=True))
        return 0

    if args.resume:
        require(root.is_dir(), "resume output root is missing")
        state = load_json(root / "state.json")
        require(state.get("status") == "running", "resume requires a clean running state; failed in-progress work needs a documented recovery decision")
        require(state.get("in_progress") is None, "resume refuses an in-progress scientific run")
        require(state["execution_manifest_sha256"] == manifest_sha, "resume execution manifest mismatch")
    else:
        require(not root.exists(), "official output root must be new")
        root.mkdir(parents=True)
        state = {
            "schema": "colibri-m6.3-r2.3b-run-state-v1",
            "status": "running",
            "execution_manifest_sha256": manifest_sha,
            "machinery_commit": manifest["machinery_commit"],
            "started_unix": time.time(),
            "in_progress": None,
            "completed_runs": [],
            "candidates": [],
        }
        atomic_json(root / "state.json", state)

    completed = completed_prefix(state)
    expected_runs = [(layer, group) for layer in LAYERS for group in GROUPS]
    builder = repo / "scripts" / "build_m6_3_r2_3b_artifact.py"
    for run_index, (layer, group) in enumerate(expected_runs):
        if run_index < len(completed):
            continue
        run_id = f"layer{layer:02d}-group{group}"
        artifact = root / f"{run_id}.bin"
        provenance = root / f"{run_id}.provenance.json"
        evidence = root / f"{run_id}.tsv"
        stdout_path = root / f"{run_id}.stdout.log"
        stderr_path = root / f"{run_id}.stderr.log"
        require(not any(path.exists() for path in (artifact, provenance, evidence, stdout_path, stderr_path)), f"new run paths already exist: {run_id}")
        free_before = shutil.disk_usage(root).free
        reserve = max(1024**3, int(free_before * 0.05))
        expected_artifact = {32: 679_477_248, 16: 754_974_720, 8: 905_969_664}[group]
        require(free_before >= expected_artifact + reserve, f"insufficient preflight disk for {run_id}")
        state["in_progress"] = {"layer": layer, "group_size": group, "run_id": run_id, "started_unix": time.time()}
        atomic_json(root / "state.json", state)
        try:
            build = subprocess.run(
                [sys.executable, str(builder), "--artifact-root", str(artifact_root), "--layer", str(layer),
                 "--group-size", str(group), "--output", str(artifact), "--provenance", str(provenance)],
                cwd=repo, check=False, capture_output=True, text=True,
            )
            stdout_path.write_text(build.stdout, encoding="utf-8", newline="\n")
            stderr_path.write_text(build.stderr, encoding="utf-8", newline="\n")
            require(build.returncode == 0, f"builder exited {build.returncode}")
            provenance_record = load_json(provenance)
            environment = os.environ.copy()
            environment.update({
                "COLIBRI_ARTIFACT_ROOT": str(artifact_root),
                "COLIBRI_R2_3A_REFERENCE_PATH": str(reference),
                "COLIBRI_R2_3B_LAYER": str(layer),
                "COLIBRI_R2_3B_GROUP_SIZE": str(group),
                "COLIBRI_R2_3B_ARTIFACT_PATH": str(artifact),
                "COLIBRI_R2_3B_ARTIFACT_SHA256": provenance_record["artifact_sha256"],
                "COLIBRI_R2_3B_SOURCE_SHA256": provenance_record["source_sha256"],
                "COLIBRI_R2_3B_OUTPUT": str(evidence),
            })
            with stdout_path.open("a", encoding="utf-8", newline="\n") as stdout_handle, stderr_path.open("a", encoding="utf-8", newline="\n") as stderr_handle:
                test = subprocess.run(
                    [str(binary), TEST_FILTER, "--exact", "--nocapture"], cwd=repo, env=environment,
                    stdout=stdout_handle, stderr=stderr_handle, check=False,
                )
            require(test.returncode == 0, f"characterization test exited {test.returncode}")
            rows = parse_tsv(evidence)
            require(all(row["layer"] == layer and row["group_size"] == group for row in rows), "harness run identity mismatch")
            state["candidates"].extend(rows)
            state["completed_runs"].append({
                "layer": layer, "group_size": group, "run_id": run_id,
                "artifact_sha256": provenance_record["artifact_sha256"],
                "source_sha256": provenance_record["source_sha256"],
                "evidence_sha256": sha256(evidence),
                "stdout_sha256": sha256(stdout_path), "stderr_sha256": sha256(stderr_path),
                "free_bytes_before": free_before, "free_bytes_after": shutil.disk_usage(root).free,
            })
            state["in_progress"] = None
            atomic_json(root / "state.json", state)
            layer_rows = [row for row in state["candidates"] if row["layer"] == layer]
            current = selected_candidate(layer_rows)
            keep_group = current["group_size"] if current is not None else None
            for completed_group in GROUPS:
                completed_artifact = root / f"layer{layer:02d}-group{completed_group}.bin"
                if completed_artifact.exists() and completed_group != keep_group:
                    completed_artifact.unlink()
        except Exception as error:
            write_failure(root, state, run_id, str(error))
            raise

    decisions = [decision(layer, [row for row in state["candidates"] if row["layer"] == layer]) for layer in LAYERS]
    result = {
        "schema": "colibri-m6.3-r2.3b-result-v1",
        "schema_version": 1,
        "status": "pass",
        "identities": {
            "implementation_contract_sha256": CONTRACT_SHA256,
            "r2_3a_reference_sha256": REFERENCE_SHA256,
            "canonical_root_manifest_sha256": ROOT_MANIFEST_SHA256,
            "execution_manifest_sha256": manifest_sha,
            "release_test_binary_sha256": manifest["release_test_binary"]["sha256"],
        },
        "candidates": state["candidates"],
        "decisions": decisions,
        "execution": {"official_candidate_count": 945, "automatic_retries": 0, "completed_runs": len(state["completed_runs"])},
    }
    errors = validate(result)
    require(not errors, "strict result validation failed: " + "; ".join(errors[:10]))
    atomic_json(root / "result.json", result)
    state["status"] = "complete"
    state["completed_unix"] = time.time()
    state["result_sha256"] = sha256(root / "result.json")
    atomic_json(root / "state.json", state)
    print(json.dumps({"status": "pass", "candidates": 945, "decisions": 45, "result_sha256": state["result_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
