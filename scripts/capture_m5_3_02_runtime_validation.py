"""Capture a bounded M5.3-02 reference/reusable-reader runtime comparison.

The harness reuses the M5.2 validation parsers and fixture validation, but runs
one fixture at a time for the two reader modes at 8 and 16 GiB. It preserves
completed traces and metrics in the repository evidence directory and writes a
canonical partial result after every successful run.
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


GIB = 1024**3
PAYLOAD_BYTES = m52.PAYLOAD_BYTES
ARTIFACT_ROOT_SHA256 = m52.ARTIFACT_ROOT_SHA256
CORPUS_ID = m52.CORPUS_ID
SIMULATION_RESULTS_SHA256 = m52.SIMULATION_RESULTS_SHA256
SIMULATION_INPUT_SHA256 = m52.SIMULATION_INPUT_SHA256
SELECTED_FIXTURES = [
    "tier_a_control",
    "tier_b_short_thai",
    "tier_b_special_token",
    "tier_b_code_newline",
    "long_english_context",
    "long_decode_english",
]
BUDGETS = {8: 8 * GIB, 16: 16 * GIB}
READER_MODES = {
    "reference_allocated": "reference_allocated",
    "reusable_aligned_buffer": "reusable_aligned_buffer",
}
CONTROL_TEST = m52.CONTROL_TEST
TRACE_TEST = m52.TRACE_TEST


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def non_timing_storage(storage: dict[str, Any]) -> dict[str, Any]:
    return {
        "reader_mode": storage["reader_mode"],
        "cache": storage["cache"],
        "path": {
            key: storage["path"][key]
            for key in ("request_count", "cache_hit_count", "expert_load_count", "bytes_copied_after_read")
        },
        "reader": {
            key: storage["reader"][key]
            for key in (
                "tensor_reads",
                "file_open_count",
                "file_handle_reuse_count",
                "metadata_count",
                "seek_count",
                "read_call_count",
                "requested_read_bytes",
                "returned_read_bytes",
                "buffer_allocation_count",
                "allocated_bytes",
                "copied_bytes",
                "buffer_growth_events",
                "buffer_reuse_count",
                "bytes_read_into_reusable_buffers",
                "bytes_copied_after_read",
                "peak_buffer_capacity",
                "fallback_allocations",
                "alignment_failures",
                "hash_bytes",
            )
        },
    }


def storage_timing(storage: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": {
            key: storage["path"][key]
            for key in ("total_nanos", "cache_lookup_nanos", "expert_load_nanos")
        },
        "reader": {
            key: storage["reader"][key]
            for key in ("open_nanos", "metadata_nanos", "seek_nanos", "read_nanos", "hash_nanos")
        },
    }


def validate_storage_metrics(
    storage: dict[str, Any],
    runtime: dict[str, Any],
    reader_mode: str,
) -> dict[str, Any]:
    require(storage["schema"] == "colibri-qwen3-moe-m5.3-02-storage-metrics-v1", "storage metric schema mismatch")
    require(storage["reader_mode"] == reader_mode, "storage reader mode mismatch")
    runtime_cache = runtime["cache"]
    storage_cache = storage["cache"]
    for key in ("hits", "misses", "loads", "evictions", "bytes_read", "peak_resident_bytes"):
        runtime_key = "bytes_read" if key == "bytes_read" else key
        require(storage_cache[key] == runtime_cache[runtime_key], f"storage/runtime cache mismatch: {key}")
    requests = runtime["io"]["expert_payload_bytes_requested"] // PAYLOAD_BYTES
    path = storage["path"]
    require(path["request_count"] == requests, "storage path request count mismatch")
    require(path["cache_hit_count"] == runtime_cache["hits"], "storage path hit count mismatch")
    require(path["expert_load_count"] == runtime_cache["loads"], "storage path load count mismatch")
    require(path["bytes_copied_after_read"] == runtime["io"]["expert_bytes_loaded"], "storage path copy accounting mismatch")
    reader = storage["reader"]
    require(reader["tensor_reads"] == runtime_cache["loads"], "reader tensor count mismatch")
    require(reader["requested_read_bytes"] == runtime["io"]["expert_bytes_loaded"], "reader requested bytes mismatch")
    require(reader["returned_read_bytes"] == reader["requested_read_bytes"], "reader returned bytes mismatch")
    require(reader["read_call_count"] == reader["tensor_reads"], "reader read-call accounting mismatch")
    require(reader["fallback_allocations"] == 0, "unexpected reusable-reader fallback allocation")
    require(reader["alignment_failures"] == 0, "reusable-reader alignment failure")
    require(reader["peak_buffer_capacity"] <= PAYLOAD_BYTES, "reusable buffer exceeded payload capacity")
    if reader_mode == "reference_allocated":
        require(reader["buffer_reuse_count"] == 0, "reference reader reported buffer reuse")
        require(reader["buffer_allocation_count"] == reader["tensor_reads"], "reference allocation count mismatch")
    else:
        require(reader["buffer_allocation_count"] <= 1, "reusable reader allocated more than one staging buffer")
        require(reader["buffer_reuse_count"] == max(0, reader["tensor_reads"] - 1), "reusable count mismatch")
    return non_timing_storage(storage)


def result_document(
    artifact: dict[str, Any],
    runs: list[dict[str, Any]],
    source_commit: str,
    status: str,
    benchmark_path: Path,
    repo: Path,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": "colibri-qwen3-moe-m5.3-02-reusable-buffer-runtime-results-v1",
        "schema_version": 1,
        "task": "M5.3-02",
        "status": status,
        "artifact": {
            "model_id": "Qwen/Qwen3-30B-A3B",
            "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "root_sha256": ARTIFACT_ROOT_SHA256,
            "validation": artifact,
        },
        "references": {
            "corpus_id": CORPUS_ID,
            "simulation_results_sha256": SIMULATION_RESULTS_SHA256,
            "simulation_input_sha256": SIMULATION_INPUT_SHA256,
            "m5_3_01_results_sha256": "2d7cab4e69d6063bebbd9c392c5635aa56183ee8a2055ad6d28e8e7a210f0ca0",
            "m5_3_01_benchmark_sha256": "59fcc85be74158497492d4c05334a490e19fbbbb5cec1b0da2c2651ec67119c",
            "benchmark_sha256": sha256_file(benchmark_path),
        },
        "selected_fixture_ids": SELECTED_FIXTURES,
        "budgets_binary_gib": {str(key): value for key, value in BUDGETS.items()},
        "reader_modes": list(READER_MODES),
        "runtime_configuration": {
            "cache_policy": "strict_global_lru",
            "compute_dtype": "F32",
            "kv_cache_dtype": "F32",
            "threads": 8,
            "target": "x86_64-pc-windows-msvc",
            "build_profile": "release",
            "filesystem_cache_assumption": "uncontrolled",
            "mmap": False,
            "persistent_handles": False,
            "coalescing": False,
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
    first = {(run["reader_mode"], run["fixture_id"], run["budget_gib"]): run for run in runs}
    per_fixture: dict[str, Any] = {}
    for fixture_id in SELECTED_FIXTURES:
        per_fixture[fixture_id] = {}
        for mode in READER_MODES:
            low = first[(mode, fixture_id, 8)]["runtime"]
            high = first[(mode, fixture_id, 16)]["runtime"]
            per_fixture[fixture_id][mode] = {
                "8_gib_byte_hit_rate": low["io"]["expert_bytes_avoided"] / low["io"]["expert_payload_bytes_requested"],
                "16_gib_byte_hit_rate": high["io"]["expert_bytes_avoided"] / high["io"]["expert_payload_bytes_requested"],
                "additional_hits_at_16_gib": high["cache"]["hits"] - low["cache"]["hits"],
                "additional_expert_bytes_avoided_at_16_gib": high["io"]["expert_bytes_avoided"] - low["io"]["expert_bytes_avoided"],
                "additional_expert_bytes_loaded_at_16_gib": high["io"]["expert_bytes_loaded"] - low["io"]["expert_bytes_loaded"],
            }
    macro_micro: dict[str, Any] = {}
    for mode in READER_MODES:
        macro_micro[mode] = {}
        for budget_gib in BUDGETS:
            rows = [first[(mode, fixture_id, budget_gib)]["runtime"] for fixture_id in SELECTED_FIXTURES]
            request_total = sum(row["io"]["expert_payload_bytes_requested"] // PAYLOAD_BYTES for row in rows)
            bytes_total = sum(row["io"]["expert_payload_bytes_requested"] for row in rows)
            macro_micro[mode][str(budget_gib)] = {
                "fixture_count": len(rows),
                "macro_request_hit_rate": sum(row["cache"]["hits"] / (row["io"]["expert_payload_bytes_requested"] // PAYLOAD_BYTES) for row in rows) / len(rows),
                "macro_byte_hit_rate": sum(row["io"]["expert_bytes_avoided"] / row["io"]["expert_payload_bytes_requested"] for row in rows) / len(rows),
                "micro_request_hit_rate": sum(row["cache"]["hits"] for row in rows) / request_total,
                "micro_byte_hit_rate": sum(row["io"]["expert_bytes_avoided"] for row in rows) / bytes_total,
                "zero_hit_fixture_count": sum(row["cache"]["hits"] == 0 for row in rows),
            }
    return {"per_fixture_8_vs_16": per_fixture, "subset_macro_micro": macro_micro}


def capture(args: argparse.Namespace) -> int:
    repo = args.root.resolve()
    artifact_root = args.artifact_root.resolve()
    executable = args.test_executable.resolve()
    benchmark_path = (repo / args.benchmark_path).resolve()
    require(executable.is_file(), f"test executable is missing: {executable}")
    require(benchmark_path.is_file(), f"benchmark evidence is missing: {benchmark_path}")
    artifact = m52.validate_artifact(repo, artifact_root)
    corpus, fixtures, simulation_data = m52.validate_corpus(repo)
    corpus_trace_paths = {item["fixture_id"]: item["trace_path"] for item in corpus["fixtures"]}
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    run_parent = args.run_root.resolve()
    run_parent.mkdir(parents=True, exist_ok=True)
    run_dir = run_parent / f"m5.3-02-{args.run_id}"
    require(not run_dir.exists(), f"run ID is not reusable: {run_dir}")
    run_dir.mkdir()
    evidence_dir = (repo / args.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path = (repo / args.result_path).resolve()
    runs: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if result_path.is_file():
        prior = load_json(result_path)
        require(prior.get("references", {}).get("simulation_results_sha256") == SIMULATION_RESULTS_SHA256, "existing result uses another simulation")
        runs = prior.get("runs", [])
        completed = {(run["reader_mode"], run["fixture_id"], run["budget_gib"]) for run in runs}
        for run in runs:
            if "storage_timing" not in run["runtime"]:
                evidence_path = evidence_dir / f"{run['reader_mode']}__{run['fixture_id']}__{run['budget_gib']}gib.storage.json"
                require(evidence_path.is_file(), f"storage evidence missing for existing run: {evidence_path.name}")
                run["runtime"]["storage_timing"] = storage_timing(load_json(evidence_path))
    normal_exit = False
    selected_modes = [args.only_reader_mode] if args.only_reader_mode else list(READER_MODES)
    selected_fixtures = [args.only_fixture] if args.only_fixture else SELECTED_FIXTURES
    require(all(mode in READER_MODES for mode in selected_modes), "unknown reader mode filter")
    require(all(fixture_id in SELECTED_FIXTURES for fixture_id in selected_fixtures), "unknown fixture filter")
    try:
        for reader_mode in selected_modes:
            mode_value = READER_MODES[reader_mode]
            for fixture_id in selected_fixtures:
                fixture = fixtures[fixture_id]
                for budget_gib, budget in BUDGETS.items():
                    key = (reader_mode, fixture_id, budget_gib)
                    if key in completed:
                        continue
                    stem = f"{reader_mode}__{fixture_id}__{budget_gib}gib"
                    trace_path = run_dir / f"{stem}.trace.json.incomplete"
                    metrics_path = run_dir / f"{stem}.runtime-metrics.incomplete"
                    storage_path = run_dir / f"{stem}.storage-metrics.json.incomplete"
                    env = os.environ.copy()
                    env.update({
                        "COLIBRI_ARTIFACT_ROOT": str(artifact_root),
                        "COLIBRI_EXPERT_CACHE_BUDGET_BYTES": str(budget),
                        "COLIBRI_EXPERT_READER_MODE": mode_value,
                        "COLIBRI_RUNTIME_VALIDATION": "1",
                        "COLIBRI_TRACE_ONLY": "1",
                        "COLIBRI_FS_CACHE_ASSUMPTION": "uncontrolled",
                        "COLIBRI_TRACE_INSTRUMENTATION_COMMIT": source_commit,
                        "COLIBRI_M5_3_STORAGE_METRICS_OUTPUT": str(storage_path),
                    })
                    if fixture_id == "tier_a_control":
                        command = [str(executable), CONTROL_TEST, "--exact", "--nocapture"]
                        diagnostic_root = run_dir / f"{stem}.diagnostics"
                        diagnostic_root.mkdir()
                        env.update({
                            "COLIBRI_RMS_DIAGNOSTIC_ROOT": str(diagnostic_root),
                            "COLIBRI_FULL_LOGITS_ROOT": str(diagnostic_root),
                            "COLIBRI_METRICS_OUTPUT": str(metrics_path),
                            "COLIBRI_EXPERT_TRACE_OUTPUT": str(trace_path),
                        })
                    else:
                        command = [str(executable), TRACE_TEST, "--exact", "--nocapture"]
                        env.update({
                            "COLIBRI_EXPERT_TRACE_OUTPUT": str(trace_path),
                            "COLIBRI_RUNTIME_METRICS_OUTPUT": str(metrics_path),
                            "COLIBRI_TRACE_FIXTURE_ID": fixture_id,
                            "COLIBRI_TRACE_WORKLOAD_CLASS": fixture["workload_class"],
                            "COLIBRI_TRACE_INPUT_TOKEN_IDS": ",".join(str(value) for value in fixture["token_ids"]),
                            "COLIBRI_TRACE_REQUESTED_GENERATION_LENGTH": str(fixture["requested_generation_length"]),
                            "COLIBRI_TRACE_SEED": str(fixture["seed"]),
                            "COLIBRI_TRACE_DECODING_MODE": fixture["decoding_mode"],
                            "COLIBRI_TRACE_KV_CACHE_CAPACITY": str(fixture["kv_cache_capacity"]),
                        })
                    process = m52.run_process(command, env, repo, run_dir, stem, args.timeout_seconds)
                    require(trace_path.is_file(), f"runtime trace missing: {stem}")
                    trace_final = trace_path.with_suffix("")
                    os.replace(trace_path, trace_final)
                    require(metrics_path.is_file(), f"runtime metrics missing: {stem}")
                    metrics_final = metrics_path.with_suffix("")
                    os.replace(metrics_path, metrics_final)
                    require(storage_path.is_file(), f"storage metrics missing: {stem}")
                    storage_final = storage_path.with_suffix("")
                    os.replace(storage_path, storage_final)
                    trace = load_json(trace_final)
                    expected_trace = load_json(repo / corpus_trace_paths[fixture_id])
                    require(m52.trace_signature(trace) == m52.trace_signature(expected_trace), f"runtime request sequence differs: {stem}")
                    if fixture_id != "tier_a_control":
                        m52.validate_trace_record_contract(trace, fixture_id)
                    else:
                        require(trace["requested_trace_count"] == len(expected_trace["records"]), f"control trace count mismatch: {stem}")
                    if fixture_id == "tier_a_control":
                        runtime = m52.parse_control_metrics(metrics_final, fixture, budget, process, {"path": trace_final})
                    else:
                        runtime = m52.parse_generic_metrics(metrics_final, fixture, budget, process, {"path": trace_final})
                    expected = simulation_data["expected"][(fixture_id, budget)]
                    runtime["simulation_comparison"] = m52.validate_runtime_against_simulation(runtime, expected)
                    storage = load_json(storage_final)
                    runtime["storage_metrics"] = validate_storage_metrics(storage, runtime, mode_value)
                    runtime["storage_timing"] = storage_timing(storage)
                    trace_hash = m52.copy_evidence(trace_final, evidence_dir / f"{stem}.trace.json")
                    metric_extension = "tsv" if fixture_id == "tier_a_control" else "json"
                    metrics_hash = m52.copy_evidence(metrics_final, evidence_dir / f"{stem}.runtime.{metric_extension}")
                    storage_hash = m52.copy_evidence(storage_final, evidence_dir / f"{stem}.storage.json")
                    non_timing = {
                        "generated_token_ids": runtime["generated_token_ids"],
                        "cache": runtime["cache"],
                        "io": runtime["io"],
                        "kv_cache": runtime["kv_cache"],
                        "correctness": runtime["correctness"],
                        "storage": runtime["storage_metrics"],
                    }
                    run = {
                        "reader_mode": reader_mode,
                        "fixture_id": fixture_id,
                        "budget_gib": budget_gib,
                        "budget_bytes": budget,
                        "trace_sha256": trace_hash,
                        "runtime_metrics_sha256": metrics_hash,
                        "storage_metrics_sha256": storage_hash,
                        "runtime": runtime,
                        "non_timing_fingerprint": non_timing,
                    }
                    runs.append(run)
                    write_json_atomic(result_path, result_document(artifact, runs, source_commit, "partial", benchmark_path, repo))
                    print(json.dumps({"reader_mode": reader_mode, "fixture_id": fixture_id, "budget_gib": budget_gib, "hits": runtime["cache"]["hits"], "loads": runtime["cache"]["loads"], "trace_sha256": trace_hash}, sort_keys=True), flush=True)
        expected_keys = {
            (reader_mode, fixture_id, budget_gib)
            for reader_mode in READER_MODES
            for fixture_id in SELECTED_FIXTURES
            for budget_gib in BUDGETS
        }
        actual_keys = {(run["reader_mode"], run["fixture_id"], run["budget_gib"]) for run in runs}
        status = "complete" if expected_keys <= actual_keys else "partial"
        write_json_atomic(result_path, result_document(artifact, runs, source_commit, status, benchmark_path, repo))
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
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--test-executable", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path(r"D:\tmp\colibri-lite-runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark-path", type=Path, default=Path("models/qwen3-30b-a3b/m5.3-02-reusable-buffer-benchmark-v1.json"))
    parser.add_argument("--result-path", type=Path, default=Path("models/qwen3-30b-a3b/m5.3-02-reusable-buffer-results-v1.json"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("models/qwen3-30b-a3b/m5.3-02-runtime-evidence"))
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--only-reader-mode", choices=tuple(READER_MODES), default=None)
    parser.add_argument("--only-fixture", choices=tuple(SELECTED_FIXTURES), default=None)
    args = parser.parse_args()
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
