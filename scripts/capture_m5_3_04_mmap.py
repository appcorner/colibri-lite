"""Capture the isolated M5.3-04 reference/mmap full-runtime matrix.

The Rust validation tests remain authoritative.  This driver validates the
canonical artifact, frozen traces, and M5.2 simulation before each one-mode,
one-fixture-at-a-time run.  Timing and process-memory observations are kept
outside the deterministic correctness fingerprint.
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

import capture_m5_2_03_runtime_validation as m52
import capture_m5_3_03_compute_profile as m53


GIB = 1024**3
ARTIFACT_ROOT_SHA256 = m52.ARTIFACT_ROOT_SHA256
M4_BASELINE_SHA256 = m52.M4_BASELINE_SHA256
M5_3_03_RESULTS_SHA256 = "25036c06623f16cb84cfa681e9697f4ef291951eea89b7b92e3d1a8017aae9c1"
M5_3_03_AGGREGATE_SHA256 = "9800aa25181e843e53fd3989f8a4edec315cab33ae68c52ccb75cba05d89390b"
SELECTED_FIXTURES = (
    "tier_a_control",
    "tier_b_code_newline",
    "long_english_context",
    "long_decode_english",
)
BUDGETS = {8: 8 * GIB, 16: 16 * GIB}
READER_MODES = ("reference_allocated", "mmap_read_only")
PROFILE_MODE = "detailed"
CONTROL_TEST = m52.CONTROL_TEST
TRACE_TEST = m52.TRACE_TEST


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    incomplete = path.with_name(path.name + ".incomplete")
    incomplete.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(incomplete, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_storage(
    storage: dict[str, Any], runtime: dict[str, Any], reader_mode: str, budget: int
) -> dict[str, Any]:
    expected_schema = (
        "colibri-qwen3-moe-m5.3-04-storage-metrics-v1"
        if reader_mode == "mmap_read_only"
        else "colibri-qwen3-moe-m5.3-02-storage-metrics-v1"
    )
    require(storage["schema"] == expected_schema, "storage schema mismatch")
    require(storage["reader_mode"] == reader_mode, "storage reader mode mismatch")
    require(storage["cache"]["configured_budget_bytes"] == budget, "storage cache budget mismatch")
    for key in ("hits", "misses", "loads", "evictions", "bytes_read", "peak_resident_bytes"):
        require(storage["cache"][key] == runtime["cache"][key], f"storage/runtime cache mismatch: {key}")
    requests = runtime["io"]["expert_payload_bytes_requested"] // m52.PAYLOAD_BYTES
    require(storage["path"]["request_count"] == requests, "storage path request count mismatch")
    require(storage["path"]["expert_load_count"] == runtime["cache"]["loads"], "storage load count mismatch")
    reader = storage["reader"]
    require(reader["tensor_reads"] == runtime["cache"]["loads"], "reader tensor count mismatch")
    require(reader["returned_read_bytes"] == reader["requested_read_bytes"], "reader byte mismatch")
    require(reader["mmap_copy_bytes"] == reader["requested_read_bytes"], "mmap copy byte mismatch")
    if reader_mode == "mmap_read_only":
        require(reader["read_call_count"] == 0, "mmap path unexpectedly used read calls")
        require(reader["mmap_mapping_count"] > 0, "mmap path created no mappings")
        require(reader["mmap_active_mapping_count"] == reader["mmap_mapping_count"], "mmap active count mismatch")
        require(reader["mmap_mapped_virtual_bytes"] > 0, "mmap virtual-byte accounting is empty")
    else:
        require(reader["mmap_mapping_count"] == 0, "reference path reported mappings")
        require(reader["mmap_mapped_virtual_bytes"] == 0, "reference path reported mapped bytes")
        require(reader["read_call_count"] == runtime["cache"]["loads"], "reference read-call count mismatch")
    return {
        "schema": storage["schema"],
        "reader_mode": storage["reader_mode"],
        "cache": storage["cache"],
        "path": storage["path"],
        "reader": reader,
        "timing": storage["timing"],
    }


def storage_non_timing(storage: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in storage.items() if key != "timing"}


def profile_non_timing(profile: dict[str, Any]) -> dict[str, Any]:
    return m53.profile_non_timing(profile)


def result_document(
    artifact: dict[str, Any], runs: list[dict[str, Any]], source_commit: str, status: str
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": "colibri-qwen3-moe-m5.3-04-mmap-results-v1",
        "schema_version": 1,
        "task": "M5.3-04",
        "status": status,
        "artifact": {
            "model_id": "Qwen/Qwen3-30B-A3B",
            "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "root_sha256": ARTIFACT_ROOT_SHA256,
            "validation": artifact,
        },
        "references": {
            "m4_baseline_sha256": M4_BASELINE_SHA256,
            "m5_3_03_results_sha256": M5_3_03_RESULTS_SHA256,
            "m5_3_03_aggregate_sha256": M5_3_03_AGGREGATE_SHA256,
            "corpus_id": m52.CORPUS_ID,
            "simulation_results_sha256": m52.SIMULATION_RESULTS_SHA256,
        },
        "selected_fixture_ids": list(SELECTED_FIXTURES),
        "budgets_binary_gib": {str(key): value for key, value in BUDGETS.items()},
        "reader_modes": list(READER_MODES),
        "runtime_configuration": {
            "cache_policy": "strict_global_lru",
            "reader_default": "reference_allocated",
            "profile_mode": PROFILE_MODE,
            "compute_dtype": "F32",
            "kv_cache_dtype": "F32",
            "threads": 8,
            "target": "x86_64-pc-windows-msvc",
            "build_profile": "release",
            "filesystem_cache_assumption": "uncontrolled",
            "mmap_scope": "expert_shards_only",
            "prefetch": False,
            "simd": False,
            "threading": False,
            "quantization": False,
            "gpu": False,
        },
        "capture_source_commit": source_commit,
        "runs": runs,
    }
    if status == "complete":
        document["summary"] = summarize(runs)
    return document


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(run["reader_mode"], run["fixture_id"], run["budget_gib"]): run for run in runs}
    paired = []
    for fixture_id in SELECTED_FIXTURES:
        for budget_gib in BUDGETS:
            reference = by_key[("reference_allocated", fixture_id, budget_gib)]
            mmap = by_key[("mmap_read_only", fixture_id, budget_gib)]
            reference_seconds = reference["runtime"]["timing"]["total_seconds"]
            mmap_seconds = mmap["runtime"]["timing"]["total_seconds"]
            paired.append(
                {
                    "fixture_id": fixture_id,
                    "budget_gib": budget_gib,
                    "reference_total_seconds": reference_seconds,
                    "mmap_total_seconds": mmap_seconds,
                    "relative_change": mmap_seconds / reference_seconds - 1.0,
                    "non_timing_identical": reference["non_timing_fingerprint"]
                    == mmap["non_timing_fingerprint"],
                }
            )
    return {
        "run_count": len(runs),
        "paired_comparisons": paired,
        "all_correctness_pass": all(
            bool(run["runtime"]["correctness"].get(key))
            for run in runs
            for key in run["runtime"]["correctness"]
            if key not in {"oversized_entry_events", "blocked_eviction_events"}
        ),
        "all_simulation_comparisons_exact": all(
            run["runtime"]["simulation_comparison"]["classification"] == "exact"
            for run in runs
        ),
    }


def capture(args: argparse.Namespace) -> int:
    repo = args.root.resolve()
    artifact_root = args.artifact_root.resolve()
    executable = args.test_executable.resolve()
    require(executable.is_file(), f"test executable is missing: {executable}")
    artifact = m52.validate_artifact(repo, artifact_root)
    corpus, fixtures, simulation_data = m52.validate_corpus(repo)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    corpus_paths = {item["fixture_id"]: item["trace_path"] for item in corpus["fixtures"]}
    run_parent = args.run_root.resolve()
    run_parent.mkdir(parents=True, exist_ok=True)
    run_dir = run_parent / f"m5.3-04-{args.run_id}"
    require(not run_dir.exists(), f"run ID is not reusable: {run_dir}")
    run_dir.mkdir()
    evidence_dir = (repo / args.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path = (repo / args.result_path).resolve()
    runs: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if result_path.is_file():
        prior = load_json(result_path)
        require(
            prior.get("references", {}).get("simulation_results_sha256") == m52.SIMULATION_RESULTS_SHA256,
            "existing result uses another simulation",
        )
        runs = prior.get("runs", [])
        completed = {(run["reader_mode"], run["fixture_id"], run["budget_gib"]) for run in runs}

    normal_exit = False
    try:
        for reader_mode in READER_MODES:
            for fixture_id in SELECTED_FIXTURES:
                fixture = fixtures[fixture_id]
                for budget_gib, budget in BUDGETS.items():
                    key = (reader_mode, fixture_id, budget_gib)
                    if key in completed:
                        continue
                    stem = f"{reader_mode}__{fixture_id}__{budget_gib}gib"
                    trace_path = run_dir / f"{stem}.trace.json.incomplete"
                    metrics_suffix = "tsv" if fixture_id == "tier_a_control" else "json"
                    metrics_path = run_dir / f"{stem}.runtime-metrics.{metrics_suffix}.incomplete"
                    storage_path = run_dir / f"{stem}.storage-metrics.json.incomplete"
                    profile_path = run_dir / f"{stem}.compute-profile.json.incomplete"
                    diagnostic_root = run_dir / f"{stem}.diagnostics"
                    diagnostic_root.mkdir()
                    environment = os.environ.copy()
                    environment.update(
                        {
                            "COLIBRI_ARTIFACT_ROOT": str(artifact_root),
                            "COLIBRI_EXPERT_CACHE_BUDGET_BYTES": str(budget),
                            "COLIBRI_EXPERT_READER_MODE": reader_mode,
                            "COLIBRI_RUNTIME_VALIDATION": "1",
                            "COLIBRI_TRACE_ONLY": "1",
                            "COLIBRI_FS_CACHE_ASSUMPTION": "uncontrolled",
                            "COLIBRI_TRACE_INSTRUMENTATION_COMMIT": source_commit,
                            "COLIBRI_COMPUTE_PROFILE_MODE": PROFILE_MODE,
                            "COLIBRI_COMPUTE_PROFILE_OUTPUT": str(profile_path),
                            "COLIBRI_M5_3_STORAGE_METRICS_OUTPUT": str(storage_path),
                            "COLIBRI_RMS_DIAGNOSTIC_ROOT": str(diagnostic_root),
                            "COLIBRI_FULL_LOGITS_ROOT": str(diagnostic_root),
                            "COLIBRI_RUNTIME_METRICS_OUTPUT": str(metrics_path),
                            "COLIBRI_EXPERT_TRACE_OUTPUT": str(trace_path),
                        }
                    )
                    if fixture_id == "tier_a_control":
                        command = [str(executable), CONTROL_TEST, "--exact", "--nocapture"]
                        environment["COLIBRI_METRICS_OUTPUT"] = str(metrics_path)
                    else:
                        command = [str(executable), TRACE_TEST, "--exact", "--nocapture"]
                        environment.update(
                            {
                                "COLIBRI_TRACE_FIXTURE_ID": fixture_id,
                                "COLIBRI_TRACE_WORKLOAD_CLASS": fixture["workload_class"],
                                "COLIBRI_TRACE_INPUT_TOKEN_IDS": ",".join(str(value) for value in fixture["token_ids"]),
                                "COLIBRI_TRACE_REQUESTED_GENERATION_LENGTH": str(fixture["requested_generation_length"]),
                                "COLIBRI_TRACE_SEED": str(fixture["seed"]),
                                "COLIBRI_TRACE_DECODING_MODE": fixture["decoding_mode"],
                                "COLIBRI_TRACE_KV_CACHE_CAPACITY": str(fixture["kv_cache_capacity"]),
                            }
                        )
                    process = m52.run_process(command, environment, repo, run_dir, stem, args.timeout_seconds)
                    for path in (trace_path, metrics_path, storage_path, profile_path):
                        require(path.is_file(), f"missing runtime evidence: {path.name}")
                    trace_final = trace_path.with_suffix("")
                    metrics_final = metrics_path.with_suffix("")
                    storage_final = storage_path.with_suffix("")
                    profile_final = profile_path.with_suffix("")
                    for source, target in (
                        (trace_path, trace_final),
                        (metrics_path, metrics_final),
                        (storage_path, storage_final),
                        (profile_path, profile_final),
                    ):
                        os.replace(source, target)
                    trace = load_json(trace_final)
                    expected_trace = load_json(repo / corpus_paths[fixture_id])
                    require(
                        m52.trace_signature(trace) == m52.trace_signature(expected_trace),
                        f"runtime request sequence differs: {stem}",
                    )
                    if fixture_id == "tier_a_control":
                        runtime = m52.parse_control_metrics(metrics_final, fixture, budget, process, {"path": trace_final})
                    else:
                        runtime = m52.parse_generic_metrics(metrics_final, fixture, budget, process, {"path": trace_final})
                    expected = simulation_data["expected"][(fixture_id, budget)]
                    runtime["simulation_comparison"] = m52.validate_runtime_against_simulation(runtime, expected)
                    storage = validate_storage(load_json(storage_final), runtime, reader_mode, budget)
                    runtime["storage_metrics"] = storage
                    profile = load_json(profile_final)
                    m53.validate_profile(profile, fixture_id, budget_gib, PROFILE_MODE)
                    require(profile["generated_token_ids"] == runtime["generated_token_ids"], f"profile IDs differ: {stem}")
                    trace_hash = m52.copy_evidence(trace_final, evidence_dir / f"{stem}.trace.json")
                    metrics_hash = m52.copy_evidence(metrics_final, evidence_dir / f"{stem}.runtime.{metrics_suffix}")
                    storage_hash = m52.copy_evidence(storage_final, evidence_dir / f"{stem}.storage.json")
                    profile_hash = m52.copy_evidence(profile_final, evidence_dir / f"{stem}.profile.json")
                    fingerprint = {
                        "generated_token_ids": runtime["generated_token_ids"],
                        "cache": runtime["cache"],
                        "io": runtime["io"],
                        "kv_cache": runtime["kv_cache"],
                        "correctness": runtime["correctness"],
                        "trace_signature": m52.trace_signature(trace),
                        "storage": storage_non_timing(storage),
                    }
                    run = {
                        "reader_mode": reader_mode,
                        "fixture_id": fixture_id,
                        "budget_gib": budget_gib,
                        "budget_bytes": budget,
                        "trace_sha256": trace_hash,
                        "runtime_metrics_sha256": metrics_hash,
                        "storage_metrics_sha256": storage_hash,
                        "profile_sha256": profile_hash,
                        "runtime": runtime,
                        "profile": profile,
                        "non_timing_fingerprint": fingerprint,
                        "profile_non_timing": profile_non_timing(profile),
                    }
                    runs.append(run)
                    completed.add(key)
                    write_json_atomic(result_path, result_document(artifact, runs, source_commit, "partial"))
                    print(
                        json.dumps(
                            {
                                "reader_mode": reader_mode,
                                "fixture_id": fixture_id,
                                "budget_gib": budget_gib,
                                "hits": runtime["cache"]["hits"],
                                "total_seconds": runtime["timing"]["total_seconds"],
                                "storage_sha256": storage_hash,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        expected_keys = {(mode, fixture_id, budget) for mode in READER_MODES for fixture_id in SELECTED_FIXTURES for budget in BUDGETS}
        status = "complete" if expected_keys <= completed else "partial"
        write_json_atomic(result_path, result_document(artifact, runs, source_commit, status))
        print(json.dumps({"status": status, "result_path": str(result_path), "run_count": len(runs)}, sort_keys=True))
        normal_exit = True
        return 0
    finally:
        if run_dir.exists() and normal_exit:
            shutil.rmtree(run_dir)
        elif run_dir.exists():
            print(f"retained failed run directory: {run_dir}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, default=Path(r"D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1"))
    parser.add_argument("--test-executable", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, default=Path("models/qwen3-30b-a3b/m5.3-04-mmap-results-v1.json"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("models/qwen3-30b-a3b/m5.3-04-runtime-evidence"))
    parser.add_argument("--run-root", type=Path, default=Path(r"D:\tmp\colibri-lite-m5.3-04"))
    parser.add_argument("--run-id", default="full-matrix")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return capture(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
