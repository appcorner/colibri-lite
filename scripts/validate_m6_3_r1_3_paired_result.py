"""Validate and summarize M6.3-R1.3 paired measurement evidence."""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

CONTRACT_SHA256 = "57a02d697231350977a4391a3625d2808b97cb394078290db398e2996195584f"
BINARY_SHA256 = "69109985e658918e8b0accf3dacb737daba53b6d6f05fb6535b256379805d541"
CANDIDATE_SHA256 = "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2"
RAM_BYTES = 17_179_869_184
F32_CACHE_BYTES = 18_874_368
CANDIDATE_PEAK_BYTES = 5_308_416
CANDIDATE_VERIFY_BYTES = 679_477_248
VERIFY_BUFFER_BYTES = 1_048_576
CONDITIONS = ["runtime_cache_cold", "runtime_cache_warm"]
PAIR_ORDER = [
    ["candidate", "reference"], ["reference", "candidate"],
    ["candidate", "reference"], ["reference", "candidate"],
    ["candidate", "reference"],
]
METRICS = {
    "process_wall_seconds": "lower",
    "setup_seconds": "lower",
    "timed_wall_seconds": "lower",
    "ttft_seconds": "lower",
    "prefill_tokens_per_second": "higher",
    "decode_tokens_per_second": "higher",
    "working_set_peak_bytes": "lower",
    "private_bytes_peak": "lower",
    "total_logical_bytes": "lower",
    "physical_read_bytes": "lower",
}

def finite_nonnegative(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0


def validate(document: dict) -> list[str]:
    errors: list[str] = []
    if document.get("schema") != "m6.3-r1.3-paired-samples-v1": errors.append("schema mismatch")
    if document.get("contract_sha256") != CONTRACT_SHA256: errors.append("contract hash mismatch")
    if document.get("binary_sha256") != BINARY_SHA256: errors.append("binary hash mismatch")
    if document.get("candidate_sha256") != CANDIDATE_SHA256: errors.append("candidate hash mismatch")
    samples = document.get("samples")
    if not isinstance(samples, list) or len(samples) != 20:
        errors.append("exactly 20 samples required")
        return errors
    expected = []
    for condition in CONDITIONS:
        for pair, modes in enumerate(PAIR_ORDER, start=1):
            for order_index, mode in enumerate(modes, start=1):
                expected.append((condition, pair, order_index, mode))
    actual = [(s.get("condition"), s.get("pair"), s.get("order_index"), s.get("mode")) for s in samples]
    if actual != expected: errors.append("sample order mismatch")
    required = [
        "process_wall_seconds", "setup_seconds", "timed_wall_seconds", "ttft_seconds",
        "prefill_tokens_per_second", "decode_tokens_per_second", "working_set_peak_bytes",
        "private_bytes_peak", "dense_logical_bytes", "f32_expert_logical_bytes",
        "candidate_logical_bytes", "total_logical_bytes", "logical_bytes_per_token",
        "physical_read_bytes", "physical_reads_per_token", "cache_hits", "cache_misses",
        "cache_loads", "cache_evictions", "f32_cache_peak_resident_bytes",
        "candidate_peak_packed_expert_bytes", "candidate_verification_bytes", "kv_cache_bytes", "vram_bytes",
    ]

    for index, sample in enumerate(samples):
        prefix = f"sample[{index}]"
        if sample.get("exit_code") != 0: errors.append(f"{prefix} exit code")
        if not isinstance(sample.get("memory_samples"), int) or sample["memory_samples"] < 1: errors.append(f"{prefix} memory samples")
        if sample.get("etw_status") != "correlated": errors.append(f"{prefix} ETW status")
        if sample.get("final_argmax") != 0: errors.append(f"{prefix} argmax")
        for name in required:
            if not finite_nonnegative(sample.get(name)): errors.append(f"{prefix} invalid {name}")
        ws = sample.get("working_set_peak_bytes", RAM_BYTES + 1)
        if ws > RAM_BYTES: errors.append(f"{prefix} RAM ceiling")
        if sample.get("f32_cache_peak_resident_bytes", F32_CACHE_BYTES + 1) > F32_CACHE_BYTES:
            errors.append(f"{prefix} F32 cache ceiling")
        total = sample.get("dense_logical_bytes", 0) + sample.get("f32_expert_logical_bytes", 0) + sample.get("candidate_logical_bytes", 0)
        if sample.get("total_logical_bytes") != total: errors.append(f"{prefix} logical byte reconciliation")
        if not math.isclose(float(sample.get("logical_bytes_per_token", -1)), total / 2.0, rel_tol=0, abs_tol=1e-6):
            errors.append(f"{prefix} logical bytes/token")
        if not math.isclose(float(sample.get("physical_reads_per_token", -1)), sample.get("physical_read_bytes", 0) / 2.0, rel_tol=0, abs_tol=1e-6):
            errors.append(f"{prefix} physical bytes/token")
        if sample.get("mode") == "candidate":
            if sample.get("candidate_logical_bytes", 0) <= 0: errors.append(f"{prefix} candidate payload not read")
            if sample.get("candidate_peak_packed_expert_bytes") != CANDIDATE_PEAK_BYTES: errors.append(f"{prefix} candidate peak")
            if sample.get("candidate_verification_bytes") != CANDIDATE_VERIFY_BYTES: errors.append(f"{prefix} candidate verification")
        else:
            if sample.get("candidate_logical_bytes") != 0 or sample.get("candidate_peak_packed_expert_bytes") != 0 or sample.get("candidate_verification_bytes") != 0:
                errors.append(f"{prefix} reference candidate accounting")

    for condition in CONDITIONS:
        for pair in range(1, 6):
            pair_samples = [s for s in samples if s["condition"] == condition and s["pair"] == pair]
            candidate = next((s for s in pair_samples if s["mode"] == "candidate"), None)
            reference = next((s for s in pair_samples if s["mode"] == "reference"), None)
            if candidate is None or reference is None:
                errors.append(f"{condition} pair {pair} missing mode")
                continue
            allowance = reference["working_set_peak_bytes"] * 1.10 + candidate["candidate_peak_packed_expert_bytes"] + VERIFY_BUFFER_BYTES
            if candidate["working_set_peak_bytes"] > allowance:
                errors.append(f"{condition} pair {pair} candidate memory accounting")
    return errors


def summarize(document: dict) -> dict:
    samples = document["samples"]
    output: dict[str, dict] = {}
    for condition in CONDITIONS:
        condition_summary: dict[str, dict] = {}
        for metric, preferred in METRICS.items():
            deltas = []
            for pair in range(1, 6):
                pair_samples = [s for s in samples if s["condition"] == condition and s["pair"] == pair]
                candidate = next(s for s in pair_samples if s["mode"] == "candidate")[metric]
                reference = next(s for s in pair_samples if s["mode"] == "reference")[metric]
                deltas.append(None if reference == 0 else 100.0 * (candidate - reference) / reference)
            numeric = [value for value in deltas if value is not None]
            median = statistics.median(numeric) if len(numeric) == 5 else None
            wanted = (lambda x: x < 0) if preferred == "lower" else (lambda x: x > 0)
            directional_win = median is not None and wanted(median) and all(wanted(value) for value in numeric)
            condition_summary[metric] = {
                "preferred": preferred,
                "paired_percent_deltas": deltas,
                "median_percent_delta": median,
                "min_percent_delta": min(numeric) if numeric else None,
                "max_percent_delta": max(numeric) if numeric else None,
                "directional_win": directional_win,
            }
        output[condition] = condition_summary
    return output

def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: validate_m6_3_r1_3_paired_result.py SAMPLES.json [RESULT.json]", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    document = json.loads(source.read_text(encoding="utf-8"))
    errors = validate(document)
    if errors:
        for error in errors: print(error, file=sys.stderr)
        return 1
    result = {
        "schema": "m6.3-r1.3-paired-result-v1",
        "status": "valid_measurement_set",
        "contract_sha256": CONTRACT_SHA256,
        "sample_count": 20,
        "summaries": summarize(document),
        "authorizes_r1_4": True,
        "authorizes_m6_4": False,
    }
    if len(sys.argv) == 3:
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        Path(sys.argv[2]).write_bytes(payload.encode("utf-8"))
    print(json.dumps({"status": result["status"], "sample_count": 20, "authorizes_r1_4": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
