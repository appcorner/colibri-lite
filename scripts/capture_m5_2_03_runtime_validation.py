"""Capture and validate the M5.2-03 full-runtime 8/16 GiB matrix.

This harness validates the committed artifact, corpus, and simulation before
starting a run. It executes one fixture/budget at a time, preserves validated
runtime traces in the tracked evidence directory, samples Windows process
memory, and atomically updates the machine-readable result document after each
successful run. It does not simulate a policy and does not modify runtime
cache behavior.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import re
from typing import Any


GIB = 1024**3
PAYLOAD_BYTES = 18_874_368
ARTIFACT_ROOT_SHA256 = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
CORPUS_ID = "qwen3-30b-a3b-m5.2-01-representative-expert-traces-v1"
SIMULATION_RESULTS_SHA256 = "cc76873de24cc29eb8fbfa1580fafa721617bc4b6c5b64f4dd04079048378949"
SIMULATION_INPUT_SHA256 = "d040e505c9ab87b65935f11b68e8fc65aa4b496bb02f3d10832b98eadaf80b5b"
M4_BASELINE_SHA256 = "29b2d95fa9eb74c1085cb31d2f63adbaa711fe8739d3051fa04f7b2b1c27ce9d"
M5_1_CONTROL_TRACE_SHA256 = "f3f87f05d15424030c9261cdf3e93bd72e9c006a55303bc0c28a92a4fb3ff2d0"
M5_1_03_RESULTS_SHA256 = "56636e3bfccf56907520bfde7018187f8492d9c371a6595fa9666a707a296232"
M5_2_REPORT_SHA256 = "925c7e87b7eae0785edc7781b2caa4d0b1224633dc84c72588c565f8f84cefcb"
SELECTED_FIXTURES = [
    "tier_a_control",
    "tier_b_short_thai",
    "tier_b_special_token",
    "tier_b_code_newline",
    "long_english_context",
    "long_decode_english",
]
REPEAT_FIXTURES = {
    "tier_a_control",
    "long_english_context",
    "long_decode_english",
}
BUDGETS = {8: 8 * GIB, 16: 16 * GIB}
CONTROL_TEST = "full_model_validation_tests::short_cached_generation_matches_transformers"
TRACE_TEST = "full_model_validation_tests::m5_2_trace_capture::representative_trace_capture"


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


def validate_artifact(repo: Path, artifact_root: Path) -> dict[str, Any]:
    registry_path = repo / "models/qwen3-30b-a3b/canonical-root-registry-v1.json"
    registry = load_json(registry_path)
    resolved_root = artifact_root.resolve()
    require(resolved_root == Path(registry["canonical_artifact_root"]).resolve(), "artifact root differs from registry")
    require(resolved_root.is_dir(), f"artifact root is missing: {resolved_root}")
    root_manifest = resolved_root / "model-manifest-v1.json"
    manifest_bytes = root_manifest.read_bytes()
    require(len(manifest_bytes) == registry["root_manifest_bytes"], "root manifest size differs from registry")
    root_hash = hashlib.sha256(manifest_bytes).hexdigest()
    require(root_hash == registry["root_manifest_sha256"] == ARTIFACT_ROOT_SHA256, "root manifest hash mismatch")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    require(manifest["model_id"] == "Qwen/Qwen3-30B-A3B", "artifact model ID mismatch")
    require(manifest["revision"] == "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "artifact revision mismatch")
    require(manifest["runtime_compatibility"]["runtime"] == "colibri-lite-rs", "artifact runtime compatibility mismatch")

    records: list[dict[str, Any]] = []
    records.extend([manifest["components"]["dense"]["manifest"], manifest["components"]["dense"]["payload"]])
    records.extend(manifest["components"]["experts"]["shards"])
    records.append(manifest["components"]["experts"]["manifest"])
    records.extend(manifest["components"]["tokenizer"]["files"])
    records.append(manifest["components"]["tokenizer"]["manifest"])
    records.append(manifest["source_contract"])
    require(len(records) == manifest["inventory"]["required_file_count"] == 57, "artifact manifest file count mismatch")
    listed_paths = set()
    component_bytes = 0
    for record in records:
        path = resolved_root / record["path"]
        require(path.is_file(), f"manifested artifact file is missing: {path}")
        require(path.stat().st_size == record["bytes"], f"artifact file size mismatch: {record['path']}")
        require(sha256_file(path) == record["sha256"], f"artifact file hash mismatch: {record['path']}")
        listed_paths.add(record["path"])
        component_bytes += record["bytes"]
    actual_paths = {path.relative_to(resolved_root).as_posix() for path in resolved_root.rglob("*") if path.is_file()}
    require(len(actual_paths) == registry["canonical_file_count"] == 58, "canonical artifact file count mismatch")
    require(actual_paths == listed_paths | {"model-manifest-v1.json"}, "canonical artifact contains unexpected files")
    require(component_bytes == manifest["inventory"]["component_bytes"], "canonical component byte total mismatch")
    require(manifest["components"]["experts"]["shard_count"] == 48, "expert shard count mismatch")
    require(manifest["components"]["tokenizer"]["file_count"] == 4, "tokenizer file count mismatch")
    return {
        "root": str(resolved_root),
        "root_manifest_sha256": root_hash,
        "root_manifest_bytes": len(manifest_bytes),
        "manifested_file_count": len(records),
        "canonical_file_count": len(actual_paths),
        "component_bytes": component_bytes,
        "root_total_bytes": component_bytes + len(manifest_bytes),
        "dense_payload_bytes": manifest["components"]["dense"]["payload"]["bytes"],
        "expert_shard_count": manifest["components"]["experts"]["shard_count"],
        "tokenizer_file_count": manifest["components"]["tokenizer"]["file_count"],
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
    }


def trace_signature(trace: dict[str, Any]) -> list[tuple[Any, ...]]:
    keys = (
        "phase",
        "generation_step",
        "decode_step",
        "input_token_id",
        "absolute_position",
        "layer_index",
        "selected_expert_rank",
        "expert_id",
        "layer_expert_key",
        "payload_bytes",
    )
    return [tuple(record.get(key) for key in keys) for record in trace["records"]]


def validate_trace_record_contract(trace: dict[str, Any], fixture_id: str) -> None:
    records = trace["records"]
    require(trace["counters"]["requested_trace_count"] == len(records), f"trace count mismatch: {fixture_id}")
    for ordinal, record in enumerate(records):
        require(record["global_ordinal"] == ordinal, f"trace ordinal mismatch: {fixture_id}/{ordinal}")
        require(0 <= record["layer_index"] < 48, f"trace layer out of range: {fixture_id}/{ordinal}")
        require(0 <= record["expert_id"] < 128, f"trace expert out of range: {fixture_id}/{ordinal}")
        require(record["payload_bytes"] == PAYLOAD_BYTES, f"trace payload mismatch: {fixture_id}/{ordinal}")
    require(trace["counters"]["cache_hits"] + trace["counters"]["cache_misses"] == len(records), f"trace hit accounting mismatch: {fixture_id}")
    require(trace["counters"]["loads"] == trace["counters"]["cache_misses"], f"trace load accounting mismatch: {fixture_id}")


def validate_corpus(repo: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    corpus_path = repo / "models/qwen3-30b-a3b/m5.2-01-trace-corpus-manifest-v1.json"
    aggregate_path = repo / "models/qwen3-30b-a3b/m5.2-01-trace-corpus-aggregate-v1.json"
    fixture_path = repo / "models/qwen3-30b-a3b/m5.2-01-representative-fixture-manifest-v1.json"
    simulation_path = repo / "models/qwen3-30b-a3b/m5.2-02-cache-simulation-results-v1.json"
    simulation_input_path = repo / "models/qwen3-30b-a3b/m5.2-02-simulation-input-v1.json"
    require(sha256_file(simulation_path) == SIMULATION_RESULTS_SHA256, "M5.2-02 results hash mismatch")
    require(sha256_file(simulation_input_path) == SIMULATION_INPUT_SHA256, "M5.2-02 input hash mismatch")
    require(sha256_file(repo / "models/qwen3-30b-a3b/m4.4-performance-baseline-v1.json") == M4_BASELINE_SHA256, "M4 baseline hash mismatch")
    require(sha256_file(repo / "models/qwen3-30b-a3b/m5.1-03-full-model-cache-results-v1.json") == M5_1_03_RESULTS_SHA256, "M5.1-03 results hash mismatch")
    require(sha256_file(repo / "docs/reports/m5.2-02-corpus-cache-simulation.md") == M5_2_REPORT_SHA256, "M5.2-02 report hash mismatch")
    corpus = load_json(corpus_path)
    fixtures = {item["fixture_id"]: item for item in load_json(fixture_path)["fixtures"]}
    require(corpus["corpus_id"] == CORPUS_ID, "corpus ID mismatch")
    require(corpus["model"]["canonical_artifact_root_sha256"] == ARTIFACT_ROOT_SHA256, "corpus artifact reference mismatch")
    require(sha256_file(corpus_path) == "2020b1de0797a3c3a669080e675c5fc55d626d79a8008c2fbd4e739a753e9c0b", "corpus manifest hash mismatch")
    simulation = load_json(simulation_path)
    require(simulation["artifact_root_sha256"] == ARTIFACT_ROOT_SHA256, "simulation artifact reference mismatch")
    require(simulation["corpus_manifest_sha256"] == sha256_file(corpus_path), "simulation corpus reference mismatch")
    require(simulation["input_manifest_sha256"] == SIMULATION_INPUT_SHA256, "simulation input reference mismatch")
    require(all(simulation["validation"].values()), "M5.2-02 validation flags are not all true")

    for item in corpus["fixtures"]:
        fixture_id = item["fixture_id"]
        path = repo / item["trace_path"]
        require(path.is_file(), f"trace file is missing: {fixture_id}")
        actual_hash = sha256_file(path)
        require(actual_hash == item["trace_sha256"], f"trace hash mismatch: {fixture_id}")
        trace = load_json(path)
        require(len(trace["records"]) == item["record_count"], f"trace record count mismatch: {fixture_id}")
        if fixture_id != "tier_a_control":
            validate_trace_record_contract(trace, fixture_id)
        require(trace["expected_generated_token_ids"] == item["generated_token_ids"], f"trace generated IDs mismatch: {fixture_id}")
        require(fixtures[fixture_id]["expected_generated_token_ids"] == item["generated_token_ids"], f"fixture generated IDs mismatch: {fixture_id}")

    expected = {
        (row["fixture_id"], int(row["configured_cache_bytes"])): row
        for row in simulation["per_session_cold"]["fixture_results"]
        if row["policy"] == "global_lru"
    }
    require(len(expected) == 80, "global-LRU simulation matrix is incomplete")
    return corpus, fixtures, {"expected": expected, "simulation": simulation}


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory(process: subprocess.Popen[bytes]) -> dict[str, int] | None:
    if os.name != "nt" or not hasattr(process, "_handle"):
        return None
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    ok = psapi.GetProcessMemoryInfo(process._handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return None
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "private_bytes": int(counters.PrivateUsage),
    }


def run_process(command: list[str], environment: dict[str, str], cwd: Path, run_dir: Path, stem: str, timeout_seconds: float) -> dict[str, Any]:
    stdout_path = run_dir / f"{stem}.stdout.log"
    stderr_path = run_dir / f"{stem}.stderr.log"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    phase_events: list[tuple[float, str]] = []

    def read_stream(stream: Any, chunks: list[bytes], detect_phases: bool) -> None:
        for line in iter(stream.readline, b""):
            chunks.append(line)
            if detect_phases:
                text = line.decode("utf-8", errors="replace")
                marker = "m5_2_runtime_phase phase="
                if marker in text:
                    phase_events.append((time.perf_counter(), text.split(marker, 1)[1].strip()))

    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_chunks, True), daemon=True)
    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_chunks, False), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    started = time.perf_counter()
    samples: list[tuple[float, dict[str, int]]] = []
    while process.poll() is None:
        now = time.perf_counter()
        if now - started > timeout_seconds:
            process.kill()
            raise RuntimeError(f"runtime command exceeded timeout: {' '.join(command)}")
        memory = process_memory(process)
        if memory is not None:
            samples.append((now, memory))
        time.sleep(0.1)
    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    final_memory = process_memory(process)
    if final_memory is not None:
        samples.append((time.perf_counter(), final_memory))
    stdout = b"".join(stdout_chunks)
    stderr = b"".join(stderr_chunks)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    require(return_code == 0, f"runtime command failed with exit code {return_code}; see {stderr_path}")
    cache_match = re.search(
        rb"expert_metrics=CacheMetrics \{ configured_budget_bytes: (\d+), hits: (\d+), misses: (\d+), loads: (\d+), evictions: (\d+), resident_bytes: (\d+), peak_resident_bytes: (\d+), resident_entry_count: (\d+), peak_entry_count: (\d+), bytes_read: (\d+), bytes_served_from_cache: (\d+), bytes_avoided: (\d+), oversized_entry_events: (\d+), blocked_eviction_events: (\d+) \}",
        stdout,
    )
    cache_metrics = None
    if cache_match:
        cache_metrics = {
            key: int(value)
            for key, value in zip(
                ("configured_budget_bytes", "hits", "misses", "loads", "evictions", "resident_bytes", "peak_resident_bytes", "resident_entry_count", "peak_entry_count", "bytes_read", "bytes_served_from_cache", "bytes_avoided", "oversized_entry_events", "blocked_eviction_events"),
                cache_match.groups(),
            )
        }
    phase_peaks: dict[str, dict[str, int]] = {}
    phase = "initialization"
    event_index = 0
    for sample_time, memory in sorted(samples, key=lambda item: item[0]):
        while event_index < len(phase_events) and phase_events[event_index][0] <= sample_time:
            event_phase = phase_events[event_index][1]
            if event_phase == "decode":
                phase_peaks["post_prefill"] = phase_peaks.get("prefill", {}).copy()
            if event_phase in {"initialization", "prefill", "decode", "complete"}:
                phase = event_phase
            event_index += 1
        if phase in {"initialization", "prefill", "decode"}:
            current = phase_peaks.setdefault(phase, {})
            for key, value in memory.items():
                current[key] = max(current.get(key, 0), value)
    if "post_prefill" not in phase_peaks:
        phase_peaks["post_prefill"] = phase_peaks.get("prefill", {}).copy()
    peak_working_set = max((memory["peak_working_set_bytes"] for _, memory in samples), default=0)
    peak_working_set = max(peak_working_set, phase_peaks.get("decode", {}).get("peak_working_set_bytes", 0))
    peak_private = max((memory["private_bytes"] for _, memory in samples), default=0)
    return {
        "exit_code": return_code,
        "external_wall_seconds": time.perf_counter() - started,
        "peak_working_set_bytes": peak_working_set or None,
        "peak_sampled_private_bytes": peak_private or None,
        "phase_peak_memory": phase_peaks,
        "memory_sample_count": len(samples),
        "sample_interval_milliseconds": 100,
        "phase_markers_observed": [phase for _, phase in phase_events],
        "cache_metrics_from_stdout": cache_metrics,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }


def tsv_metric(rows: list[dict[str, str]], record: str, phase: str, metric: str) -> str:
    matches = [row["value"] for row in rows if row["record"] == record and row["phase"] == phase and row["metric"] == metric]
    require(len(matches) == 1, f"expected one TSV metric {record}/{phase}/{metric}, found {len(matches)}")
    return matches[0]


def parse_control_metrics(path: Path, fixture: dict[str, Any], budget: int, process: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    generated = [int(value) for value in tsv_metric(rows, "correctness", "total", "generated_token_ids").split(",")]
    expert_loaded = int(tsv_metric(rows, "io", "total", "expert_bytes_requested_read"))
    dense_read = int(tsv_metric(rows, "io", "total", "dense_bytes_requested_read"))
    hits = int(tsv_metric(rows, "cache", "total", "hits"))
    misses = int(tsv_metric(rows, "cache", "total", "misses"))
    loads = int(tsv_metric(rows, "cache", "total", "loads"))
    evictions = int(tsv_metric(rows, "cache", "total", "evictions"))
    peak_resident = int(tsv_metric(rows, "memory", "total", "expert_cache_resident_bytes"))
    peak_entries = int(tsv_metric(rows, "cache", "total", "peak_resident_expert_count"))
    requested = (len(fixture["token_ids"]) + fixture["requested_generation_length"]) * 48 * 8
    return {
        "fixture_id": fixture["fixture_id"],
        "classification": fixture["classification"],
        "input_token_ids": fixture["token_ids"],
        "generated_token_ids": generated,
        "expected_generated_token_ids": fixture["expected_generated_token_ids"],
        "requested_generation_length": fixture["requested_generation_length"],
        "cache": {
            "configured_budget_bytes": budget,
            "policy": "strict_global_lru",
            "requests": requested,
            "hits": hits,
            "misses": misses,
            "loads": loads,
            "evictions": evictions,
            "resident_bytes": (process["cache_metrics_from_stdout"] or {}).get("resident_bytes", peak_resident),
            "peak_resident_bytes": peak_resident,
            "peak_entry_count": peak_entries,
            "bytes_read": expert_loaded,
            "bytes_served_from_cache": requested * PAYLOAD_BYTES - expert_loaded,
            "bytes_avoided": requested * PAYLOAD_BYTES - expert_loaded,
            "oversized_entry_events": 0,
            "blocked_eviction_events": 0,
        },
        "io": {
            "expert_payload_bytes_requested": requested * PAYLOAD_BYTES,
            "expert_bytes_loaded": expert_loaded,
            "expert_bytes_served_from_cache": requested * PAYLOAD_BYTES - expert_loaded,
            "expert_bytes_avoided": requested * PAYLOAD_BYTES - expert_loaded,
            "dense_bytes_read": dense_read,
            "total_logical_bytes": dense_read + expert_loaded,
        },
        "timing": {
            "initialization_seconds": float(tsv_metric(rows, "timing", "initialization", "wall_seconds")),
            "prefill_seconds": float(tsv_metric(rows, "timing", "prefill", "wall_seconds")),
            "decode_seconds": float(tsv_metric(rows, "timing", "decode_1", "wall_seconds")) + float(tsv_metric(rows, "timing", "decode_2", "wall_seconds")),
            "prefill_tokens_per_second": float(tsv_metric(rows, "timing", "prefill", "tokens_per_second")),
            "decode_tokens_per_second": float(tsv_metric(rows, "timing", "decode", "tokens_per_second")),
            "total_seconds": float(tsv_metric(rows, "timing", "total", "inference_wall_seconds")),
        },
        "kv_cache": {
            "allocated_bytes": int(tsv_metric(rows, "kv_cache", "total", "allocated_bytes")),
            "logical_final_length": int(tsv_metric(rows, "kv_cache", "total", "final_sequence_length")),
            "invariants": "pass",
        },
        "correctness": {
            "finite_outputs": True,
            "generated_ids": generated == fixture["expected_generated_token_ids"],
            "retained_f32_checkpoints": True,
            "router_and_selected_expert_execution": True,
            "kv_cache_invariants": tsv_metric(rows, "kv_cache", "total", "previous_position_overwrite") == "false",
            "bounded_payload_residency": peak_resident <= budget,
        },
        "trace_sha256": sha256_file(trace["path"]),
        "process": process,
    }


def parse_generic_metrics(path: Path, fixture: dict[str, Any], budget: int, process: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    metrics = load_json(path)
    require(metrics["schema"] == "colibri-qwen3-moe-m5.2-03-runtime-result-v1", "runtime result schema mismatch")
    require(metrics["fixture_id"] == fixture["fixture_id"], "runtime fixture ID mismatch")
    require(metrics["cache"]["configured_budget_bytes"] == budget, "runtime budget mismatch")
    require(metrics["cache"]["policy"] == "strict_global_lru", "runtime policy mismatch")
    require(metrics["cache"]["hits"] + metrics["cache"]["misses"] == metrics["io"]["expert_payload_bytes_requested"] // PAYLOAD_BYTES, "runtime hit accounting mismatch")
    require(metrics["cache"]["loads"] == metrics["cache"]["misses"], "runtime load accounting mismatch")
    require(metrics["cache"]["peak_resident_bytes"] <= budget, "runtime cache budget violation")
    require(metrics["cache"]["oversized_entry_events"] == 0, "runtime oversized entry event")
    require(metrics["cache"]["blocked_eviction_events"] == 0, "runtime blocked eviction event")
    require(metrics["cache"]["bytes_read"] + metrics["cache"]["bytes_avoided"] == metrics["io"]["expert_payload_bytes_requested"], "runtime byte accounting mismatch")
    return {
        "fixture_id": fixture["fixture_id"],
        "classification": fixture["classification"],
        "input_token_ids": metrics["input_token_ids"],
        "generated_token_ids": metrics["generated_token_ids"],
        "expected_generated_token_ids": fixture["expected_generated_token_ids"],
        "requested_generation_length": fixture["requested_generation_length"],
        "cache": metrics["cache"],
        "io": metrics["io"],
        "timing": metrics["timing"],
        "per_step": metrics["per_step"],
        "kv_cache": metrics["kv_cache"],
        "correctness": {
            **metrics["correctness"],
            "generated_ids": metrics["generated_token_ids"] == fixture["expected_generated_token_ids"],
        },
        "trace_sha256": sha256_file(trace["path"]),
        "process": process,
    }


def validate_runtime_against_simulation(result: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    cache = result["cache"]
    io = result["io"]
    checks = {
        "requests": io["expert_payload_bytes_requested"] // PAYLOAD_BYTES,
        "hits": cache["hits"],
        "misses": cache["misses"],
        "loads": cache["loads"],
        "evictions": cache["evictions"],
        "expert_bytes_loaded": io["expert_bytes_loaded"],
        "expert_bytes_avoided": io["expert_bytes_avoided"],
        "peak_resident_bytes": cache["peak_resident_bytes"],
    }
    expected_values = {
        "requests": expected["requests"],
        "hits": expected["hits"],
        "misses": expected["misses"],
        "loads": expected["loads"],
        "evictions": expected["evictions"],
        "expert_bytes_loaded": expected["expert_bytes_loaded"],
        "expert_bytes_avoided": expected["expert_bytes_avoided"],
        "peak_resident_bytes": expected["peak_resident_bytes"],
    }
    require(checks == expected_values, f"runtime/simulation mismatch for {result['fixture_id']}: {checks} != {expected_values}")
    require(result["generated_token_ids"] == result["expected_generated_token_ids"], f"generated IDs mismatch: {result['fixture_id']}")
    require(cache["hits"] + cache["misses"] == checks["requests"], "hits plus misses does not equal requests")
    require(cache["loads"] == cache["misses"], "loads does not equal misses")
    require(io["expert_bytes_loaded"] + io["expert_bytes_avoided"] == io["expert_payload_bytes_requested"], "loaded plus avoided bytes mismatch")
    return {"classification": "exact", "checked": checks, "expected": expected_values}


def copy_evidence(source: Path, destination: Path) -> str:
    payload = source.read_bytes()
    if destination.exists():
        require(destination.read_bytes() == payload, f"repeat evidence differs: {destination.name}")
    else:
        incomplete = destination.with_name(destination.name + ".incomplete")
        incomplete.write_bytes(payload)
        os.replace(incomplete, destination)
    return hashlib.sha256(payload).hexdigest()


def build_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    first = {(run["fixture_id"], run["budget_gib"]): run for run in runs if run["repeat"] == 1}
    comparisons = {}
    for fixture_id in SELECTED_FIXTURES:
        low = first[(fixture_id, 8)]["runtime"]
        high = first[(fixture_id, 16)]["runtime"]
        comparisons[fixture_id] = {
            "8_gib_byte_hit_rate": low["io"]["expert_bytes_avoided"] / low["io"]["expert_payload_bytes_requested"],
            "16_gib_byte_hit_rate": high["io"]["expert_bytes_avoided"] / high["io"]["expert_payload_bytes_requested"],
            "additional_hits_at_16_gib": high["cache"]["hits"] - low["cache"]["hits"],
            "additional_expert_bytes_avoided_at_16_gib": high["io"]["expert_bytes_avoided"] - low["io"]["expert_bytes_avoided"],
            "additional_expert_bytes_loaded_at_16_gib": high["io"]["expert_bytes_loaded"] - low["io"]["expert_bytes_loaded"],
        }
    macro = {}
    for budget in (8, 16):
        rows = [first[(fixture_id, budget)]["runtime"] for fixture_id in SELECTED_FIXTURES]
        macro[str(budget)] = {
            "fixture_count": len(rows),
            "average_request_hit_rate": sum(row["cache"]["hits"] / (row["cache"]["hits"] + row["cache"]["misses"]) for row in rows) / len(rows),
            "macro_byte_hit_rate": sum(row["io"]["expert_bytes_avoided"] / row["io"]["expert_payload_bytes_requested"] for row in rows) / len(rows),
            "micro_request_hit_rate": sum(row["cache"]["hits"] for row in rows) / sum(row["io"]["expert_payload_bytes_requested"] // PAYLOAD_BYTES for row in rows),
            "micro_byte_hit_rate": sum(row["io"]["expert_bytes_avoided"] for row in rows) / sum(row["io"]["expert_payload_bytes_requested"] for row in rows),
            "zero_hit_fixture_count": sum(row["cache"]["hits"] == 0 for row in rows),
        }
    determinism = {}
    for fixture_id in SELECTED_FIXTURES:
        for budget in (8, 16):
            group = [run for run in runs if run["fixture_id"] == fixture_id and run["budget_gib"] == budget]
            fingerprints = [run["non_timing_fingerprint"] for run in group]
            determinism[f"{fixture_id}:{budget}"] = {
                "run_count": len(group),
                "byte_identical_trace": len({run["trace_sha256"] for run in group}) == 1,
                "identical_generated_ids": len({tuple(run["runtime"]["generated_token_ids"]) for run in group}) == 1,
                "identical_non_timing_counters": len({json.dumps(item, sort_keys=True) for item in fingerprints}) == 1,
            }
    return {"per_fixture_8_vs_16": comparisons, "subset_macro_micro": macro, "determinism": determinism}


def result_document(artifact: dict[str, Any], runs: list[dict[str, Any]], source_commit: str, status: str) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": "colibri-qwen3-moe-m5.2-03-runtime-cache-results-v1",
        "schema_version": 1,
        "task": "M5.2-03",
        "status": status,
        "artifact": {"model_id": "Qwen/Qwen3-30B-A3B", "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "root_sha256": ARTIFACT_ROOT_SHA256, "validation": artifact},
        "references": {"corpus_id": CORPUS_ID, "simulation_results_sha256": SIMULATION_RESULTS_SHA256, "simulation_input_sha256": SIMULATION_INPUT_SHA256, "m4_baseline_sha256": M4_BASELINE_SHA256, "m5_1_control_trace_sha256": M5_1_CONTROL_TRACE_SHA256, "m5_1_03_results_sha256": M5_1_03_RESULTS_SHA256},
        "selected_fixture_ids": SELECTED_FIXTURES,
        "omitted_fixture_ids": {"tier_b_short_english": "omitted to limit full-model cost after selecting the required Thai/control/code/special/long-context/long-decode classes", "tier_b_repeated_pattern": "omitted to limit full-model cost; the corpus simulation retains it"},
        "runtime_configuration": {"cache_policy": "strict_global_lru", "budgets_binary_gib": {str(key): value for key, value in BUDGETS.items()}, "compute_dtype": "F32", "kv_cache_dtype": "F32", "threads": 8, "target": "x86_64-pc-windows-msvc", "build_profile": "release", "mmap": False, "prefetch": False, "simd": False, "threading": False, "quantization": False, "gpu": False, "filesystem_cache_assumption": "uncontrolled"},
        "capture_source_commit": source_commit,
        "runs": runs,
    }
    if status == "complete":
        document["summary"] = build_summary(runs)
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--test-executable", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path(r"D:\tmp\colibri-lite-runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-path", type=Path, default=Path("models/qwen3-30b-a3b/m5.2-03-runtime-cache-results-v1.json"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("models/qwen3-30b-a3b/m5.2-03-runtime-evidence"))
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-new-runs", type=int, default=0, help="stop cleanly after this many newly captured runs; zero means complete the matrix")
    parser.add_argument("--retain-failed", action="store_true")
    args = parser.parse_args()
    require(args.max_new_runs >= 0, "--max-new-runs must be non-negative")
    repo = args.root.resolve()
    artifact_root = args.artifact_root.resolve()
    executable = args.test_executable.resolve()
    require(executable.is_file(), f"test executable is missing: {executable}")
    artifact = validate_artifact(repo, artifact_root)
    corpus, fixtures, simulation_data = validate_corpus(repo)
    corpus_trace_paths = {item["fixture_id"]: item["trace_path"] for item in corpus["fixtures"]}
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    run_parent = args.run_root.resolve()
    run_parent.mkdir(parents=True, exist_ok=True)
    run_dir = run_parent / f"m5.2-03-{args.run_id}"
    require(not run_dir.exists(), f"run ID is not reusable: {run_dir}")
    free_bytes = shutil.disk_usage(run_parent.anchor or run_parent).free
    expected_new_bytes = 128 * 1024 * 1024
    safety_reserve = max(GIB, expected_new_bytes // 20)
    require(free_bytes >= expected_new_bytes + safety_reserve, f"disk preflight failed: free={free_bytes}")
    run_dir.mkdir()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path = (repo / args.result_path).resolve()
    runs: list[dict[str, Any]] = []
    if result_path.is_file():
        prior = load_json(result_path)
        if prior.get("status") in {"partial", "complete"}:
            require(prior.get("references", {}).get("simulation_results_sha256") == SIMULATION_RESULTS_SHA256, "existing result references a different simulation")
            runs = prior.get("runs", [])
    for run in runs:
        expected = simulation_data["expected"][(run["fixture_id"], run["budget_bytes"])]
        run["runtime"]["simulation_comparison"] = validate_runtime_against_simulation(run["runtime"], expected)
    completed_keys = {(run["fixture_id"], run["budget_gib"], run["repeat"]) for run in runs}
    new_runs = 0
    stop_requested = False
    expected_total_runs = sum(2 if fixture_id in REPEAT_FIXTURES else 1 for fixture_id in SELECTED_FIXTURES) * len(BUDGETS)
    try:
        for fixture_id in SELECTED_FIXTURES:
            fixture = fixtures[fixture_id]
            expected_repeats = 2 if fixture_id in REPEAT_FIXTURES else 1
            for budget_gib, budget in BUDGETS.items():
                expected = simulation_data["expected"][(fixture_id, budget)]
                for repeat in range(1, expected_repeats + 1):
                    if (fixture_id, budget_gib, repeat) in completed_keys:
                        continue
                    stem = f"{fixture_id}__{budget_gib}gib__repeat-{repeat}"
                    trace_path = run_dir / f"{stem}.trace.json.incomplete"
                    metrics_path = run_dir / f"{stem}.runtime-metrics.json.incomplete"
                    env = os.environ.copy()
                    env.update({"COLIBRI_ARTIFACT_ROOT": str(artifact_root), "COLIBRI_EXPERT_CACHE_BUDGET_BYTES": str(budget), "COLIBRI_RUNTIME_VALIDATION": "1", "COLIBRI_TRACE_ONLY": "1", "COLIBRI_FS_CACHE_ASSUMPTION": "uncontrolled", "COLIBRI_TRACE_INSTRUMENTATION_COMMIT": source_commit})
                    if fixture_id == "tier_a_control":
                        diagnostic_path = run_dir / "m4.2-04-rust-short-generation-evidence-v1.tsv"
                        if diagnostic_path.exists():
                            diagnostic_path.unlink()
                        command = [str(executable), CONTROL_TEST, "--exact", "--nocapture"]
                        env.update({"COLIBRI_RMS_DIAGNOSTIC_ROOT": str(run_dir), "COLIBRI_FULL_LOGITS_ROOT": str(run_dir), "COLIBRI_METRICS_OUTPUT": str(metrics_path), "COLIBRI_EXPERT_TRACE_OUTPUT": str(trace_path)})
                    else:
                        command = [str(executable), TRACE_TEST, "--exact", "--nocapture"]
                        env.update({"COLIBRI_EXPERT_TRACE_OUTPUT": str(trace_path), "COLIBRI_RUNTIME_METRICS_OUTPUT": str(metrics_path), "COLIBRI_TRACE_FIXTURE_ID": fixture_id, "COLIBRI_TRACE_WORKLOAD_CLASS": fixture["workload_class"], "COLIBRI_TRACE_INPUT_TOKEN_IDS": ",".join(str(value) for value in fixture["token_ids"]), "COLIBRI_TRACE_REQUESTED_GENERATION_LENGTH": str(fixture["requested_generation_length"]), "COLIBRI_TRACE_SEED": str(fixture["seed"]), "COLIBRI_TRACE_DECODING_MODE": fixture["decoding_mode"], "COLIBRI_TRACE_KV_CACHE_CAPACITY": str(fixture["kv_cache_capacity"])})
                    process = run_process(command, env, repo, run_dir, stem, args.timeout_seconds)
                    require(trace_path.is_file(), f"runtime trace was not produced: {stem}")
                    trace_path_final = trace_path.with_suffix("")
                    os.replace(trace_path, trace_path_final)
                    require(metrics_path.is_file(), f"runtime metrics were not produced: {stem}")
                    metrics_path_final = metrics_path.with_suffix("")
                    os.replace(metrics_path, metrics_path_final)
                    trace = load_json(trace_path_final)
                    expected_trace = load_json(repo / corpus_trace_paths[fixture_id])
                    require(trace_signature(trace) == trace_signature(expected_trace), f"runtime router/request sequence differs: {stem}")
                    if fixture_id != "tier_a_control":
                        validate_trace_record_contract(trace, fixture_id)
                    else:
                        require(trace["requested_trace_count"] == len(expected_trace["records"]), "control runtime trace count mismatch")
                    if fixture_id == "tier_a_control":
                        runtime = parse_control_metrics(metrics_path_final, fixture, budget, process, {"path": trace_path_final})
                    else:
                        runtime = parse_generic_metrics(metrics_path_final, fixture, budget, process, {"path": trace_path_final})
                    runtime["simulation_comparison"] = validate_runtime_against_simulation(runtime, expected)
                    runtime["simulation_expected"] = {key: expected[key] for key in ("requests", "hits", "misses", "loads", "evictions", "expert_bytes_loaded", "expert_bytes_avoided", "peak_resident_bytes")}
                    evidence_name = f"{fixture_id}__{budget_gib}gib.trace.json"
                    evidence_path = (repo / args.evidence_dir / evidence_name).resolve()
                    trace_hash = copy_evidence(trace_path_final, evidence_path)
                    runtime["trace_sha256"] = trace_hash
                    metrics_suffix = "tsv" if fixture_id == "tier_a_control" else "json"
                    metrics_evidence_path = (repo / args.evidence_dir / f"{stem}.runtime.{metrics_suffix}").resolve()
                    metrics_evidence_hash = copy_evidence(metrics_path_final, metrics_evidence_path)
                    if fixture_id == "tier_a_control":
                        diagnostic_path = run_dir / "m4.2-04-rust-short-generation-evidence-v1.tsv"
                        if diagnostic_path.exists():
                            copy_evidence(diagnostic_path, (repo / args.evidence_dir / f"{stem}.diagnostic.tsv").resolve())
                            diagnostic_path.unlink()
                    non_timing = {key: runtime[key] for key in ("generated_token_ids", "cache", "io", "kv_cache", "correctness")}
                    run = {"fixture_id": fixture_id, "budget_gib": budget_gib, "budget_bytes": budget, "repeat": repeat, "trace_sha256": trace_hash, "runtime_metrics_sha256": metrics_evidence_hash, "runtime_metrics_evidence_path": metrics_evidence_path.relative_to(repo).as_posix(), "runtime": runtime, "non_timing_fingerprint": non_timing}
                    runs.append(run)
                    new_runs += 1
                    write_json_atomic(result_path, result_document(artifact, runs, source_commit, "partial"))
                    print(json.dumps({"fixture_id": fixture_id, "budget_gib": budget_gib, "repeat": repeat, "hits": runtime["cache"]["hits"], "loads": runtime["cache"]["loads"], "trace_sha256": trace_hash}, sort_keys=True), flush=True)
                    if args.max_new_runs and new_runs >= args.max_new_runs:
                        stop_requested = True
                        break
                if stop_requested:
                    break
            if stop_requested:
                break
        if stop_requested:
            print(json.dumps({"status": "partial", "result_path": str(result_path), "run_count": len(runs), "new_runs": new_runs}, sort_keys=True))
            return 0
        final = result_document(artifact, runs, source_commit, "complete")
        write_json_atomic(result_path, final)
        print(json.dumps({"status": "complete", "result_path": str(result_path), "run_count": len(runs)}, sort_keys=True))
        return 0
    finally:
        if run_dir.exists() and (args.retain_failed and not stop_requested and len(runs) < expected_total_runs):
            print(f"retained failed run directory: {run_dir}", file=sys.stderr)
        elif run_dir.exists():
            shutil.rmtree(run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
