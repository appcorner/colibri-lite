"""Capture the M5.4-02 streamed/resident-dense paired runtime matrix.

This measurement-only harness reuses the canonical M5.2 fixtures, strict
global-LRU cache, and full-model assertions. It never writes to the canonical
artifact root and keeps all transient output in one flat policy-owned run
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import capture_m5_2_03_runtime_validation as m52


GIB = 1024**3
FIXED_RUNTIME_MEMORY_BYTES = 377_384_088
TASK = "M5.4-02"
SCHEMA = "colibri-qwen3-moe-m5.4-02-resident-dense-runtime-results-v1"
MODES = ("streamed_dense", "resident_dense")
BUDGETS = (8, 16)
MAX_DIAGNOSTIC_STREAM_BYTES = 64 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json_atomic(path: Path, value: Any) -> None:
    incomplete = path.with_name(path.name + ".incomplete")
    incomplete.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(incomplete, path)


def bounded_text(path: Path) -> str:
    if not path.is_file():
        return "<not written>"
    return path.read_bytes()[:MAX_DIAGNOSTIC_STREAM_BYTES].decode("utf-8", errors="replace")


def retain_failure_diagnostic(
    evidence_dir: Path,
    stem: str,
    command: list[str],
    environment: dict[str, str],
    error: Exception,
    run_dir: Path,
) -> tuple[Path, str]:
    relevant = {key: environment[key] for key in sorted(environment) if key.startswith("COLIBRI_")}
    diagnostic = {
        "schema": "colibri-qwen3-moe-m5.4-02-failure-diagnostic-v1",
        "command": command,
        "environment": relevant,
        "failure_stage": "subprocess",
        "error": str(error),
        "stdout": bounded_text(run_dir / f"{stem}.stdout.log"),
        "stderr": bounded_text(run_dir / f"{stem}.stderr.log"),
        "stream_truncation_bytes": MAX_DIAGNOSTIC_STREAM_BYTES,
    }
    payload = (json.dumps(diagnostic, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path = evidence_dir / f"{stem}.failure.json"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def cache_budget(total_budget: int, dense_payload_bytes: int, mode: str) -> int:
    resident_dense = dense_payload_bytes if mode == "resident_dense" else 0
    budget = total_budget - FIXED_RUNTIME_MEMORY_BYTES - resident_dense
    require(budget > 0, f"{mode} fixed reservation exceeds total budget")
    return budget


def document(
    artifact: dict[str, Any],
    source_commit: str,
    preflight: dict[str, Any],
    postflight: dict[str, Any] | None,
    runs: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "task": TASK,
        "status": status,
        "measurement_only": True,
        "production_adoption_authorized": False,
        "artifact": {
            "model_id": artifact["model_id"],
            "revision": artifact["revision"],
            "root_sha256": artifact["root_manifest_sha256"],
            "root": artifact["root"],
            "validation": artifact,
        },
        "references": {
            "m5_2_runtime_results_sha256": "0a0b964eaca9de55f3244f45b275b8d386b66b448a701a35377fbf85631ae870",
            "m5_4_01_simulation": "models/qwen3-30b-a3b/m5.4-01-resident-dense-simulation-v1.json",
        },
        "fixtures": {
            "selected": m52.SELECTED_FIXTURES,
            "unavailable": {
                "tier_b_short_english": "no M5.2 full-runtime dense-read evidence",
                "tier_b_repeated_pattern": "no M5.2 full-runtime dense-read evidence",
            },
        },
        "accounting": {
            "formula": "total = resident_dense + fixed_runtime_memory + expert_cache",
            "fixed_runtime_memory_bytes": FIXED_RUNTIME_MEMORY_BYTES,
            "fixed_runtime_memory_basis": "M5.4-01 allowance plus baseline peak dense decode buffer",
            "cache_policy": "strict_global_lru",
            "physical_io_measured": False,
        },
        "capture_source_commit": source_commit,
        "preflight": preflight,
        "postflight": postflight,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--test-executable", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path(r"D:\tmp\colibri-lite-runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--max-new-runs", type=int, default=0)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--budget-gib", type=int, choices=BUDGETS)
    parser.add_argument("--fixture", choices=m52.SELECTED_FIXTURES)
    args = parser.parse_args()
    require(args.max_new_runs >= 0, "--max-new-runs must be non-negative")

    repo = args.root.resolve()
    artifact_root = args.artifact_root.resolve()
    executable = args.test_executable.resolve()
    require(executable.is_file(), f"test executable is missing: {executable}")
    artifact = m52.validate_artifact(repo, artifact_root)
    _, fixtures, _ = m52.validate_corpus(repo)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dense_payload_bytes = artifact["dense_payload_bytes"]

    run_parent = args.run_root.resolve()
    run_dir = run_parent / f"m5.4-02-{args.run_id}"
    require(not run_dir.exists(), f"run ID is not reusable: {run_dir}")
    free_before = shutil.disk_usage(run_parent.anchor).free
    expected_new_bytes = 256 * 1024 * 1024
    safety_reserve = max(GIB, expected_new_bytes // 20)
    require(free_before >= expected_new_bytes + safety_reserve, "disk preflight failed")
    preflight = {
        "run_directory": str(run_dir),
        "free_bytes_before": free_before,
        "expected_new_output_bytes": expected_new_bytes,
        "expected_peak_temporary_bytes": 0,
        "safety_reserve_bytes": safety_reserve,
        "canonical_artifact_root": str(artifact_root),
        "canonical_artifact_bytes": artifact["root_total_bytes"],
        "canonical_input_modified": False,
    }
    run_parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    evidence_dir = (repo / args.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path = (repo / args.result_path).resolve()
    runs: list[dict[str, Any]] = []
    if result_path.is_file():
        prior = m52.load_json(result_path)
        require(prior.get("schema") == SCHEMA, "existing result schema mismatch")
        runs = prior.get("runs", [])
    completed = {(run.get("mode"), run.get("fixture_id"), run.get("total_budget_gib")) for run in runs if run.get("status", "passed") == "passed"}
    new_runs = 0
    postflight: dict[str, Any] | None = None
    try:
        for total_gib in BUDGETS:
            if args.budget_gib is not None and total_gib != args.budget_gib:
                continue
            total_budget = total_gib * GIB
            for mode in MODES:
                if args.mode is not None and mode != args.mode:
                    continue
                expert_cache_budget = cache_budget(total_budget, dense_payload_bytes, mode)
                for fixture_id in m52.SELECTED_FIXTURES:
                    if args.fixture is not None and fixture_id != args.fixture:
                        continue
                    if (mode, fixture_id, total_gib) in completed:
                        continue
                    fixture = fixtures[fixture_id]
                    stem = f"{mode}__{fixture_id}__{total_gib}gib__repeat-1"
                    trace_path = run_dir / f"{stem}.trace.json.incomplete"
                    metrics_path = run_dir / f"{stem}.runtime.json.incomplete"
                    environment = os.environ.copy()
                    environment.update({
                        "COLIBRI_ARTIFACT_ROOT": str(artifact_root),
                        "COLIBRI_EXPERT_CACHE_BUDGET_BYTES": str(expert_cache_budget),
                        "COLIBRI_TOTAL_RAM_BUDGET_BYTES": str(total_budget),
                        "COLIBRI_DENSE_RESIDENCY_MODE": mode,
                        "COLIBRI_RUNTIME_VALIDATION": "1",
                        "COLIBRI_TRACE_ONLY": "1",
                        "COLIBRI_FS_CACHE_ASSUMPTION": "uncontrolled",
                        "COLIBRI_TRACE_INSTRUMENTATION_COMMIT": source_commit,
                        "COLIBRI_EXPERT_TRACE_OUTPUT": str(trace_path),
                    })
                    if fixture_id == "tier_a_control":
                        environment.update({
                            "COLIBRI_RMS_DIAGNOSTIC_ROOT": str(run_dir),
                            "COLIBRI_FULL_LOGITS_ROOT": str(run_dir),
                            "COLIBRI_METRICS_OUTPUT": str(metrics_path),
                        })
                        command = [str(executable), m52.CONTROL_TEST, "--exact", "--nocapture"]
                    else:
                        environment.update({
                            "COLIBRI_RUNTIME_METRICS_OUTPUT": str(metrics_path),
                            "COLIBRI_TRACE_FIXTURE_ID": fixture_id,
                            "COLIBRI_TRACE_WORKLOAD_CLASS": fixture["workload_class"],
                            "COLIBRI_TRACE_INPUT_TOKEN_IDS": ",".join(map(str, fixture["token_ids"])),
                            "COLIBRI_TRACE_REQUESTED_GENERATION_LENGTH": str(fixture["requested_generation_length"]),
                            "COLIBRI_TRACE_SEED": str(fixture["seed"]),
                            "COLIBRI_TRACE_DECODING_MODE": fixture["decoding_mode"],
                            "COLIBRI_TRACE_KV_CACHE_CAPACITY": str(fixture["kv_cache_capacity"]),
                        })
                        command = [str(executable), m52.TRACE_TEST, "--exact", "--nocapture"]
                    try:
                        process = m52.run_process(command, environment, repo, run_dir, stem, args.timeout_seconds)
                    except RuntimeError as error:
                        diagnostic_path, diagnostic_hash = retain_failure_diagnostic(
                            evidence_dir, stem, command, environment, error, run_dir
                        )
                        runs.append({
                            "fixture_id": fixture_id,
                            "mode": mode,
                            "total_budget_gib": total_gib,
                            "total_budget_bytes": total_budget,
                            "expert_cache_budget_bytes": expert_cache_budget,
                            "resident_dense_bytes": dense_payload_bytes if mode == "resident_dense" else 0,
                            "status": "failed",
                            "exit_code": 101,
                            "failure_stage": "subprocess",
                            "diagnostic_path": diagnostic_path.relative_to(repo).as_posix(),
                            "diagnostic_sha256": diagnostic_hash,
                        })
                        write_json_atomic(result_path, document(artifact, source_commit, preflight, None, runs, "partial"))
                        return 1
                    require(trace_path.is_file() and metrics_path.is_file(), f"missing runtime evidence: {stem}")
                    trace_final = trace_path.with_suffix("")
                    metrics_final = metrics_path.with_suffix("")
                    os.replace(trace_path, trace_final)
                    os.replace(metrics_path, metrics_final)
                    trace = m52.load_json(trace_final)
                    expected_trace_path = repo / next(
                        item["trace_path"] for item in m52.load_json(repo / "models/qwen3-30b-a3b/m5.2-01-trace-corpus-manifest-v1.json")["fixtures"]
                        if item["fixture_id"] == fixture_id
                    )
                    require(m52.trace_signature(trace) == m52.trace_signature(m52.load_json(expected_trace_path)), f"router trace mismatch: {stem}")
                    runtime = (
                        m52.parse_control_metrics(metrics_final, fixture, expert_cache_budget, process, {"path": trace_final})
                        if fixture_id == "tier_a_control"
                        else m52.parse_generic_metrics(metrics_final, fixture, expert_cache_budget, process, {"path": trace_final})
                    )
                    cache = runtime["cache"]
                    require(cache["resident_bytes"] <= expert_cache_budget, f"expert cache budget exceeded: {stem}")
                    requests = runtime["io"]["expert_payload_bytes_requested"] // m52.PAYLOAD_BYTES
                    require(cache["hits"] + cache["misses"] == requests, f"cache accounting mismatch: {stem}")
                    trace_hash = m52.copy_evidence(trace_final, evidence_dir / f"{stem}.trace.json")
                    metrics_hash = m52.copy_evidence(metrics_final, evidence_dir / f"{stem}.runtime.json")
                    accounted_peak = dense_payload_bytes if mode == "resident_dense" else 0
                    accounted_peak += FIXED_RUNTIME_MEMORY_BYTES + cache["peak_resident_bytes"]
                    require(accounted_peak <= total_budget, f"total budget exceeded: {stem}")
                    runs.append({
                        "fixture_id": fixture_id,
                        "mode": mode,
                        "total_budget_gib": total_gib,
                        "total_budget_bytes": total_budget,
                        "expert_cache_budget_bytes": expert_cache_budget,
                        "resident_dense_bytes": dense_payload_bytes if mode == "resident_dense" else 0,
                        "accounted_peak_budget_bytes": accounted_peak,
                        "repeat": 1,
                        "status": "passed",
                        "trace_sha256": trace_hash,
                        "runtime_metrics_sha256": metrics_hash,
                        "runtime": runtime,
                    })
                    new_runs += 1
                    write_json_atomic(result_path, document(artifact, source_commit, preflight, None, runs, "partial"))
                    if args.max_new_runs and new_runs >= args.max_new_runs:
                        return 0
        return 0
    finally:
        free_after = shutil.disk_usage(run_parent.anchor).free
        postflight = {"free_bytes_after": free_after, "run_directory_removed": True, "canonical_input_modified": False}
        write_json_atomic(result_path, document(artifact, source_commit, preflight, postflight, runs, "complete" if len(runs) == 24 else "partial"))
        if run_dir.exists():
            shutil.rmtree(run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
