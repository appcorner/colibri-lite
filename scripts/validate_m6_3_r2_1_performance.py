#!/usr/bin/env python3
"""Validate M6.3-R2.1c performance samples against the frozen gates."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

CONTRACT_SHA256 = "76bfc4a850a8d89fb5901eda338568b7226b36acb7207324575568ea21f6cc2a"
FIXTURES = ("short_english", "short_thai")
VIEWS = ("compute_only_preloaded", "load_plus_compute")
PATHS = ("native_avx2_fma_group32", "scalar_group32", "reference_f32")

class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def grouped_samples(document: dict) -> dict[tuple[str, str, int], dict[str, dict]]:
    groups: dict[tuple[str, str, int], dict[str, dict]] = {}
    for sample in document["samples"]:
        key = (sample["fixture"], sample["view"], sample["triplet"])
        groups.setdefault(key, {})[sample["path"]] = sample
    return groups

def validate_structure(document: dict) -> dict[tuple[str, str, int], dict[str, dict]]:
    require(document.get("schema") == "m6.3-r2.1-performance-samples-v1", "document schema")
    require(document.get("contract_sha256") == CONTRACT_SHA256, "contract binding")
    require(document.get("sample_count") == 72, "sample count")
    require(document.get("automatic_retry") is False, "automatic retry policy")
    samples = document.get("samples")
    require(isinstance(samples, list) and len(samples) == 72, "sample list coverage")
    for sample in samples:
        require(sample.get("schema") == "m6.3-r2.1-performance-sample-v1", "sample schema")
        require(sample.get("contract_sha256") == CONTRACT_SHA256, "sample contract")
        require(sample.get("fixture") in FIXTURES, "sample fixture")
        require(sample.get("view") in VIEWS, "sample view")
        require(sample.get("path") in PATHS, "sample path")
        require(sample.get("attempt_ordinal") == 1, "sample retry detected")
        expected_iterations = 25 if sample["view"] == "compute_only_preloaded" else 10
        require(sample.get("timed_iterations") == expected_iterations, "timed iteration count")
        require(sample.get("timed_total_nanos", 0) > 0, "non-positive timing")
        require(sample.get("logical_expert_bytes") == sample.get("expected_logical_expert_bytes"), "logical byte accounting")
    groups = grouped_samples(document)
    require(len(groups) == 24, "triplet group coverage")
    for fixture in FIXTURES:
        for view in VIEWS:
            for triplet in range(1, 7):
                group = groups.get((fixture, view, triplet))
                require(group is not None and set(group) == set(PATHS), "triplet path coverage")
    return groups

def evaluate_fixture(groups: dict, fixture: str, view: str) -> dict:
    scalar_over_native: list[float] = []
    native_over_f32: list[float] = []
    native_faster_scalar = 0
    native_faster_f32 = 0
    native_within_110 = 0
    for triplet in range(1, 7):
        group = groups[(fixture, view, triplet)]
        native = float(group["native_avx2_fma_group32"]["timed_total_nanos"])
        scalar = float(group["scalar_group32"]["timed_total_nanos"])
        f32 = float(group["reference_f32"]["timed_total_nanos"])
        scalar_over_native.append(scalar / native)
        native_over_f32.append(native / f32)
        native_faster_scalar += int(native < scalar)
        native_faster_f32 += int(native < f32)
        native_within_110 += int(native / f32 <= 1.10)
    summary = {
        "native_faster_than_scalar_triplets": native_faster_scalar,
        "median_scalar_over_native_speedup": median(scalar_over_native),
        "native_faster_than_f32_triplets": native_faster_f32,
        "median_native_over_f32_ratio": median(native_over_f32),
        "native_over_f32_within_1_10_triplets": native_within_110,
        "scalar_over_native_ratios": scalar_over_native,
        "native_over_f32_ratios": native_over_f32,
    }
    if view == "compute_only_preloaded":
        summary["passed"] = (
            native_faster_scalar >= 5
            and summary["median_scalar_over_native_speedup"] >= 1.50
            and summary["median_native_over_f32_ratio"] <= 1.05
            and native_within_110 >= 5
        )
    else:
        summary["passed"] = (
            native_faster_scalar >= 5
            and summary["median_scalar_over_native_speedup"] >= 1.25
            and native_faster_f32 == 6
            and summary["median_native_over_f32_ratio"] <= 0.50
        )
    return summary

def resource_gate(document: dict) -> dict:
    candidate_rows = [row for row in document["samples"] if row["path"] != "reference_f32"]
    reference_rows = [row for row in document["samples"] if row["path"] == "reference_f32"]
    candidate_payload_ok = all(
        row["timed_candidate_payload_bytes"] == row["expected_logical_expert_bytes"]
        for row in candidate_rows if row["view"] == "load_plus_compute"
    )
    candidate_compute_zero = all(
        row["timed_candidate_payload_bytes"] == 0
        for row in candidate_rows if row["view"] == "compute_only_preloaded"
    )
    reference_payload_ok = all(
        row["timed_f32_expert_load_bytes"] == row["expected_logical_expert_bytes"]
        for row in reference_rows if row["view"] == "load_plus_compute"
    )
    reference_compute_zero = all(
        row["timed_f32_expert_load_bytes"] == 0
        for row in reference_rows if row["view"] == "compute_only_preloaded"
    )
    return {
        "candidate_payload_accounting": candidate_payload_ok,
        "candidate_compute_preloaded_zero_load_bytes": candidate_compute_zero,
        "reference_payload_accounting": reference_payload_ok,
        "reference_compute_preloaded_zero_load_bytes": reference_compute_zero,
        "complete_f32_weight_materializations": 0,
        "persistent_native_prepack_bytes": 0,
        "vram_bytes": 0,
        "passed": candidate_payload_ok and candidate_compute_zero and reference_payload_ok and reference_compute_zero,
    }


def validate(document: dict) -> dict:
    groups = validate_structure(document)
    summaries = {
        f"{fixture}:{view}": evaluate_fixture(groups, fixture, view)
        for fixture in FIXTURES for view in VIEWS
    }
    resources = resource_gate(document)
    passed = resources["passed"] and all(item["passed"] for item in summaries.values())
    return {
        "schema": "m6.3-r2.1-performance-result-v1",
        "contract_sha256": CONTRACT_SHA256,
        "status": "passed" if passed else "failed",
        "summaries": summaries,
        "resource_gate": resources,
        "decision": "authorize_r2_2_design_only" if passed else "r2_1_no_go_or_new_compute_hypothesis",
        "authorizes_r2_2_implementation": False,
        "authorizes_m6_4": False,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_m6_3_r2_1_performance.py SAMPLES.json", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    try:
        result = validate(read_json(path))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        print(f"R2.1 validator: FAIL: {error}", file=sys.stderr)
        return 1
    result_path = path.with_name("m6.3-r2-1-performance-result-v1.json")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"R2.1 performance validation: {result['status'].upper()}")
    for key, summary in result["summaries"].items():
        print(
            f"{key}: scalar/native={summary['median_scalar_over_native_speedup']:.4f}x "
            f"native/f32={summary['median_native_over_f32_ratio']:.4f} pass={summary['passed']}"
        )
    print(f"result={result_path}")
    # A structurally valid measurement is a successful validation run even when
    # the frozen performance gates produce a legitimate NO-GO decision.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
