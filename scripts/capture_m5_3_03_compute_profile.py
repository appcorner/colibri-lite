"""Capture the bounded M5.3-03 compute-profile matrix.

The Rust test binary remains the authoritative full-model execution path. This
driver validates the committed artifact, corpus, traces, and M5.2 simulation
before each one-fixture-at-a-time run. Profile timing is intentionally kept
outside deterministic trace fingerprints.
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
ARTIFACT_ROOT_SHA256 = m52.ARTIFACT_ROOT_SHA256
M4_BASELINE_SHA256 = m52.M4_BASELINE_SHA256
M5_3_02_RESULTS_SHA256 = "69121543607046c2c88bf312cae8c506840e74832cad4ac2d328c2658a97641a"
SELECTED_FIXTURES = [
    "tier_a_control",
    "tier_b_code_newline",
    "long_english_context",
    "long_decode_english",
]
BUDGETS = {8: 8 * GIB, 16: 16 * GIB}
PROFILE_MODES = ("disabled", "coarse", "detailed")
DETAILED_FIXTURES = tuple(SELECTED_FIXTURES)
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


def validate_profile(profile: dict[str, Any], fixture_id: str, budget_gib: int, mode: str) -> None:
    require(profile["schema"] == "colibri-qwen3-moe-m5.3-03-compute-profile-v1", "profile schema mismatch")
    require(profile["schema_version"] == 1, "profile schema version mismatch")
    require(profile["fixture_id"] == fixture_id, "profile fixture mismatch")
    require(profile["cache_budget_bytes"] == BUDGETS[budget_gib], "profile cache budget mismatch")
    require(profile["mode"] == mode, "profile mode mismatch")
    events = profile["events"]
    require(isinstance(events, list), "profile events must be a list")
    previous: tuple[str, int, str] | None = None
    for event in events:
        key = (event["phase"], -1 if event["layer"] is None else event["layer"], event["operation"])
        require(previous is None or previous <= key, "profile events are not deterministically ordered")
        previous = key
        for field in ("calls", "total_nanos", "exclusive_nanos", "min_nanos", "max_nanos"):
            require(isinstance(event[field], int) and event[field] >= 0, f"invalid profile counter: {field}")
        require(event["calls"] == 0 or event["min_nanos"] <= event["max_nanos"], "profile min/max mismatch")
        require(event["exclusive_nanos"] <= event["total_nanos"], "profile exclusive time exceeds inclusive time")
        for matrix in event["matrices"]:
            require(all(isinstance(matrix[field], int) and matrix[field] >= 0 for field in ("rows", "outputs", "inputs", "calls")), "invalid matrix shape")


def profile_non_timing(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": profile["mode"],
        "fixture_id": profile["fixture_id"],
        "cache_budget_bytes": profile["cache_budget_bytes"],
        "input_token_ids": profile["input_token_ids"],
        "generated_token_ids": profile["generated_token_ids"],
        "scope_count": profile["scope_count"],
        "events": [
            {
                key: event[key]
                for key in ("phase", "layer", "operation", "calls", "estimated_flops", "input_bytes", "output_bytes", "matrices")
            }
            for event in profile["events"]
        ],
    }


def validate_storage(storage: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    require(storage["schema"] == "colibri-qwen3-moe-m5.3-02-storage-metrics-v1", "storage schema mismatch")
    require(storage["reader_mode"] == "reference_allocated", "compute profile must use reference reader")
    cache = storage["cache"]
    runtime_cache = runtime["cache"]
    for key in ("hits", "misses", "loads", "evictions", "bytes_read", "peak_resident_bytes"):
        require(cache[key] == runtime_cache[key], f"storage/runtime cache mismatch: {key}")
    requests = runtime["io"]["expert_payload_bytes_requested"] // m52.PAYLOAD_BYTES
    require(storage["path"]["request_count"] == requests, "storage path request count mismatch")
    require(storage["path"]["expert_load_count"] == runtime_cache["loads"], "storage load count mismatch")
    require(storage["reader"]["tensor_reads"] == runtime_cache["loads"], "reader tensor count mismatch")
    require(storage["reader"]["returned_read_bytes"] == storage["reader"]["requested_read_bytes"], "reader byte mismatch")
    return {
        "reader_mode": storage["reader_mode"],
        "cache": cache,
        "path": {
            key: storage["path"][key]
            for key in ("request_count", "cache_hit_count", "expert_load_count", "bytes_copied_after_read")
        },
        "reader": {
            key: storage["reader"][key]
            for key in (
                "tensor_reads", "file_open_count", "file_handle_reuse_count", "seek_count", "read_call_count",
                "requested_read_bytes", "returned_read_bytes", "buffer_allocation_count", "allocated_bytes",
                "copied_bytes", "buffer_growth_events", "buffer_reuse_count", "bytes_read_into_reusable_buffers",
                "bytes_copied_after_read", "peak_buffer_capacity", "fallback_allocations", "alignment_failures",
                "hash_bytes",
            )
        },
        "timing": {
            "path": {key: storage["path"][key] for key in ("total_nanos", "cache_lookup_nanos", "expert_load_nanos")},
            "reader": {key: storage["reader"][key] for key in ("open_nanos", "metadata_nanos", "seek_nanos", "read_nanos", "hash_nanos")},
        },
    }


def storage_non_timing(storage: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in storage.items() if key != "timing"}


def result_document(artifact: dict[str, Any], runs: list[dict[str, Any]], source_commit: str, status: str) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": "colibri-qwen3-moe-m5.3-03-compute-profile-results-v1",
        "schema_version": 1,
        "task": "M5.3-03",
        "status": status,
        "artifact": {
            "model_id": "Qwen/Qwen3-30B-A3B",
            "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "root_sha256": ARTIFACT_ROOT_SHA256,
            "validation": artifact,
        },
        "references": {
            "m4_baseline_sha256": M4_BASELINE_SHA256,
            "m5_3_02_results_sha256": M5_3_02_RESULTS_SHA256,
            "corpus_id": m52.CORPUS_ID,
            "simulation_results_sha256": m52.SIMULATION_RESULTS_SHA256,
        },
        "selected_fixture_ids": SELECTED_FIXTURES,
        "budgets_binary_gib": {str(key): value for key, value in BUDGETS.items()},
        "profile_modes": list(PROFILE_MODES),
        "runtime_configuration": {
            "cache_policy": "strict_global_lru",
            "reader_mode": "reference_allocated",
            "compute_dtype": "F32",
            "kv_cache_dtype": "F32",
            "threads": 8,
            "target": "x86_64-pc-windows-msvc",
            "build_profile": "release",
            "filesystem_cache_assumption": "uncontrolled",
            "mmap": False,
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
    return {
        "run_count": len(runs),
        "modes": sorted({run["profile_mode"] for run in runs}),
        "detailed_run_count": sum(run["profile_mode"] == "detailed" for run in runs),
        "all_correctness_pass": all(run["runtime"]["correctness"][key] in (True, "pass") for run in runs for key in run["runtime"]["correctness"] if key not in {"oversized_entry_events", "blocked_eviction_events"}),
        "all_simulation_comparisons_exact": all(run["runtime"]["simulation_comparison"]["classification"] == "exact" for run in runs),
        "mode_comparisons": mode_comparisons(runs),
    }


def mode_comparisons(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(run["fixture_id"], run["budget_gib"], run["profile_mode"]): run for run in runs}
    comparisons = []
    for fixture_id in SELECTED_FIXTURES:
        for budget_gib in BUDGETS:
            key = (fixture_id, budget_gib, "detailed")
            if key not in by_key:
                continue
            reference = by_key[key]["non_timing_fingerprint"]
            for mode in ("disabled", "coarse"):
                candidate = by_key.get((fixture_id, budget_gib, mode))
                if candidate is not None:
                    comparisons.append({
                        "fixture_id": fixture_id,
                        "budget_gib": budget_gib,
                        "mode": mode,
                        "non_timing_identical": candidate["non_timing_fingerprint"] == reference,
                    })
    return comparisons


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
    run_dir = run_parent / f"m5.3-03-{args.run_id}"
    require(not run_dir.exists(), f"run ID is not reusable: {run_dir}")
    run_dir.mkdir()
    evidence_dir = (repo / args.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path = (repo / args.result_path).resolve()
    runs: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if result_path.is_file():
        prior = load_json(result_path)
        require(prior.get("references", {}).get("simulation_results_sha256") == m52.SIMULATION_RESULTS_SHA256, "existing result uses another simulation")
        runs = prior.get("runs", [])
        for run in runs:
            if "non_timing_fingerprint" in run and "storage_metrics" in run["runtime"]:
                run["non_timing_fingerprint"]["storage"] = storage_non_timing(run["runtime"]["storage_metrics"])
        completed = {(run["profile_mode"], run["fixture_id"], run["budget_gib"]) for run in runs}

    required_keys = {
        ("detailed", fixture_id, budget_gib)
        for fixture_id in DETAILED_FIXTURES
        for budget_gib in BUDGETS
    } | {("disabled", "tier_a_control", 8), ("coarse", "tier_a_control", 8)}
    normal_exit = False
    try:
        for mode, fixture_ids, budgets in (
            ("disabled", ("tier_a_control",), (8,)),
            ("coarse", ("tier_a_control",), (8,)),
            ("detailed", DETAILED_FIXTURES, tuple(BUDGETS)),
        ):
            for fixture_id in fixture_ids:
                fixture = fixtures[fixture_id]
                for budget_gib in budgets:
                    key = (mode, fixture_id, budget_gib)
                    if key in completed:
                        continue
                    budget = BUDGETS[budget_gib]
                    stem = f"{mode}__{fixture_id}__{budget_gib}gib"
                    trace_path = run_dir / f"{stem}.trace.json.incomplete"
                    metrics_path = run_dir / f"{stem}.runtime-metrics.incomplete"
                    storage_path = run_dir / f"{stem}.storage-metrics.json.incomplete"
                    profile_path = run_dir / f"{stem}.compute-profile.json.incomplete"
                    diagnostic_root = run_dir / f"{stem}.diagnostics"
                    diagnostic_root.mkdir()
                    environment = os.environ.copy()
                    environment.update({
                        "COLIBRI_ARTIFACT_ROOT": str(artifact_root),
                        "COLIBRI_EXPERT_CACHE_BUDGET_BYTES": str(budget),
                        "COLIBRI_RUNTIME_VALIDATION": "1",
                        "COLIBRI_TRACE_ONLY": "1",
                        "COLIBRI_FS_CACHE_ASSUMPTION": "uncontrolled",
                        "COLIBRI_TRACE_INSTRUMENTATION_COMMIT": source_commit,
                        "COLIBRI_COMPUTE_PROFILE_MODE": mode,
                        "COLIBRI_COMPUTE_PROFILE_OUTPUT": str(profile_path),
                        "COLIBRI_M5_3_STORAGE_METRICS_OUTPUT": str(storage_path),
                        "COLIBRI_RMS_DIAGNOSTIC_ROOT": str(diagnostic_root),
                        "COLIBRI_FULL_LOGITS_ROOT": str(diagnostic_root),
                        "COLIBRI_RUNTIME_METRICS_OUTPUT": str(metrics_path),
                        "COLIBRI_EXPERT_TRACE_OUTPUT": str(trace_path),
                    })
                    if fixture_id == "tier_a_control":
                        command = [str(executable), CONTROL_TEST, "--exact", "--nocapture"]
                        environment["COLIBRI_METRICS_OUTPUT"] = str(metrics_path)
                    else:
                        command = [str(executable), TRACE_TEST, "--exact", "--nocapture"]
                        environment.update({
                            "COLIBRI_TRACE_FIXTURE_ID": fixture_id,
                            "COLIBRI_TRACE_WORKLOAD_CLASS": fixture["workload_class"],
                            "COLIBRI_TRACE_INPUT_TOKEN_IDS": ",".join(str(value) for value in fixture["token_ids"]),
                            "COLIBRI_TRACE_REQUESTED_GENERATION_LENGTH": str(fixture["requested_generation_length"]),
                            "COLIBRI_TRACE_SEED": str(fixture["seed"]),
                            "COLIBRI_TRACE_DECODING_MODE": fixture["decoding_mode"],
                            "COLIBRI_TRACE_KV_CACHE_CAPACITY": str(fixture["kv_cache_capacity"]),
                        })
                    process = m52.run_process(command, environment, repo, run_dir, stem, args.timeout_seconds)
                    for path in (trace_path, metrics_path, storage_path, profile_path):
                        require(path.is_file(), f"missing runtime evidence: {path.name}")
                    trace_final = trace_path.with_suffix("")
                    metrics_final = metrics_path.with_suffix("")
                    storage_final = storage_path.with_suffix("")
                    profile_final = profile_path.with_suffix("")
                    for source, target in ((trace_path, trace_final), (metrics_path, metrics_final), (storage_path, storage_final), (profile_path, profile_final)):
                        os.replace(source, target)
                    trace = load_json(trace_final)
                    expected_trace = load_json(repo / corpus_paths[fixture_id])
                    require(m52.trace_signature(trace) == m52.trace_signature(expected_trace), f"runtime request sequence differs: {stem}")
                    if fixture_id == "tier_a_control":
                        runtime = m52.parse_control_metrics(metrics_final, fixture, budget, process, {"path": trace_final})
                    else:
                        runtime = m52.parse_generic_metrics(metrics_final, fixture, budget, process, {"path": trace_final})
                    expected = simulation_data["expected"][(fixture_id, budget)]
                    runtime["simulation_comparison"] = m52.validate_runtime_against_simulation(runtime, expected)
                    storage = load_json(storage_final)
                    runtime["storage_metrics"] = validate_storage(storage, runtime)
                    profile = load_json(profile_final)
                    validate_profile(profile, fixture_id, budget_gib, mode)
                    require(profile["generated_token_ids"] == runtime["generated_token_ids"], f"profile output IDs differ: {stem}")
                    trace_hash = m52.copy_evidence(trace_final, evidence_dir / f"{stem}.trace.json")
                    metrics_extension = "tsv" if fixture_id == "tier_a_control" else "json"
                    metrics_hash = m52.copy_evidence(metrics_final, evidence_dir / f"{stem}.runtime.{metrics_extension}")
                    storage_hash = m52.copy_evidence(storage_final, evidence_dir / f"{stem}.storage.json")
                    profile_hash = m52.copy_evidence(profile_final, evidence_dir / f"{stem}.profile.json")
                    fingerprint = {
                        "generated_token_ids": runtime["generated_token_ids"],
                        "cache": runtime["cache"],
                        "io": runtime["io"],
                        "kv_cache": runtime["kv_cache"],
                        "correctness": runtime["correctness"],
                        "storage": storage_non_timing(runtime["storage_metrics"]),
                    }
                    run = {
                        "profile_mode": mode,
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
                    print(json.dumps({"mode": mode, "fixture_id": fixture_id, "budget_gib": budget_gib, "hits": runtime["cache"]["hits"], "total_seconds": runtime["timing"]["total_seconds"], "profile_sha256": profile_hash}, sort_keys=True), flush=True)
        actual_keys = {(run["profile_mode"], run["fixture_id"], run["budget_gib"]) for run in runs}
        status = "complete" if required_keys <= actual_keys else "partial"
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
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--test-executable", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path(r"D:\tmp\colibri-lite-runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-path", type=Path, default=Path("models/qwen3-30b-a3b/m5.3-03-compute-profile-results-v1.json"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("models/qwen3-30b-a3b/m5.3-03-profile-evidence"))
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
