"""Validate and describe the M5.2-01 representative trace corpus.

This module performs schema, identity, ordering, repeatability, and descriptive
statistics checks. It intentionally does not run a cache-policy simulation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_ARTIFACT_ROOT = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
EXPECTED_TRACE_SCHEMA = "colibri-qwen3-moe-m5.2-01-ordered-expert-trace-v2"
CONTROL_SCHEMA = "colibri-qwen3-moe-m5.1-00-ordered-expert-trace-v1"
PAYLOAD_BYTES = 18_874_368
LAYER_COUNT = 48
EXPERT_COUNT = 128
CONTROL_TRACE = Path("models/qwen3-30b-a3b/m5.1-00-ordered-expert-trace-v1.json")
CONTROL_MANIFEST = Path("models/qwen3-30b-a3b/m5.1-00-trace-manifest-v1.json")
M4_BASELINE = Path("models/qwen3-30b-a3b/m4.4-performance-baseline-v1.json")
M5_1_03_RESULTS = Path("models/qwen3-30b-a3b/m5.1-03-full-model-cache-results-v1.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def lower_median(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def hot_share(frequencies: Counter[str], fraction: float, total: int) -> dict[str, Any]:
    if not frequencies or total == 0:
        return {"key_count": 0, "request_count": 0, "percentage": 0.0}
    key_count = max(1, math.ceil(len(frequencies) * fraction))
    request_count = sum(count for _, count in frequencies.most_common(key_count))
    return {
        "key_count": key_count,
        "request_count": request_count,
        "percentage": request_count / total,
    }


def validate_records(trace: dict[str, Any], fixture: dict[str, Any], is_control: bool) -> None:
    expected_schema = CONTROL_SCHEMA if is_control else EXPECTED_TRACE_SCHEMA
    assert trace["schema"] == expected_schema
    assert trace["canonical_artifact_root_sha256"] == EXPECTED_ARTIFACT_ROOT
    assert trace["input_token_ids"] == fixture["token_ids"]
    if is_control:
        assert trace["expected_generated_token_ids"] == fixture["expected_generated_token_ids"]
    else:
        assert trace["fixture_id"] == fixture["fixture_id"]
        assert trace["requested_generation_length"] == fixture["requested_generation_length"]
        assert trace["cache_configuration"]["budget_bytes"] == 18_874_368
        assert trace["cache_configuration"]["policy"] == "strict_global_lru"
        assert trace["runtime_configuration"]["compute_dtype"] == "F32"
        assert trace["kv_cache"]["capacity"] == fixture["kv_cache_capacity"]
        assert trace["expected_generated_token_ids"] == fixture["expected_generated_token_ids"]
        expected_steps = len(fixture["token_ids"]) + fixture["requested_generation_length"] - 1
        assert trace["kv_cache"]["final_sequence_length"] == expected_steps
        assert len(trace["records"]) == expected_steps * LAYER_COUNT * 8
    records = trace["records"]
    assert len(records) == trace.get("requested_trace_count", trace.get("counters", {}).get("requested_trace_count"))
    keys: list[str] = []
    for ordinal, record in enumerate(records):
        assert record["global_ordinal"] == ordinal
        fixture_id = fixture["fixture_id"]
        if not is_control:
            assert record["fixture_id"] == fixture_id
        assert record["layer_expert_key"] == f"layer.{record['layer_index']}.expert.{record['expert_id']}"
        assert 0 <= record["layer_index"] < LAYER_COUNT
        assert 0 <= record["expert_id"] < EXPERT_COUNT
        assert record["payload_bytes"] == PAYLOAD_BYTES
        assert record["phase"] in ("prefill", "decode")
        assert record["input_token_id"] < 151_936
        keys.append(record["layer_expert_key"])
    assert len(keys) == len(records)


def statistics(trace: dict[str, Any], fixture_id: str) -> dict[str, Any]:
    records = trace["records"]
    frequencies = Counter(record["layer_expert_key"] for record in records)
    per_layer = [0] * LAYER_COUNT
    phase_counts = {"prefill": 0, "decode": 0}
    last_seen: dict[str, tuple[int, int, str]] = {}
    distances: list[int] = []
    first_reuse_distance: int | None = None
    repeated_occurrences = 0
    cross_token_reuse = 0
    cross_step_reuse = 0
    cross_phase_reuse = 0
    for ordinal, record in enumerate(records):
        key = record["layer_expert_key"]
        per_layer[record["layer_index"]] += 1
        phase_counts[record["phase"]] += 1
        previous = last_seen.get(key)
        if previous is not None:
            previous_ordinal, previous_position, previous_phase = previous
            distance = ordinal - previous_ordinal
            distances.append(distance)
            repeated_occurrences += 1
            if first_reuse_distance is None:
                first_reuse_distance = distance
            if record["absolute_position"] != previous_position:
                cross_token_reuse += 1
            if record["generation_step"] != previous_position:
                cross_step_reuse += 1
            if record["phase"] != previous_phase:
                cross_phase_reuse += 1
        last_seen[key] = (ordinal, record["absolute_position"], record["phase"])
    total = len(records)
    payload_requested = sum(record["payload_bytes"] for record in records)
    repeated_payload = repeated_occurrences * PAYLOAD_BYTES
    unique_payload = len(frequencies) * PAYLOAD_BYTES
    histogram = Counter(frequencies.values())
    top_keys = [
        {"layer_expert_key": key, "request_count": count}
        for key, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:20]
    ]
    return {
        "fixture_id": fixture_id,
        "total_expert_occurrences": total,
        "unique_layer_expert_keys": len(frequencies),
        "unique_key_ratio": len(frequencies) / total if total else 0.0,
        "request_frequency_distribution": {
            "key_frequency_to_key_count": {str(key): histogram[key] for key in sorted(histogram)},
            "minimum_requests_per_key": min(frequencies.values()) if frequencies else 0,
            "median_requests_per_key": lower_median(list(frequencies.values())),
            "maximum_requests_per_key": max(frequencies.values()) if frequencies else 0,
            "top_keys": top_keys,
        },
        "reuse_distance": {
            "first_reuse_distance": first_reuse_distance,
            "minimum": min(distances) if distances else None,
            "median": lower_median(distances),
            "maximum": max(distances) if distances else None,
            "repeated_occurrences": repeated_occurrences,
        },
        "per_layer_request_counts": per_layer,
        "repeated_key_percentage": repeated_occurrences / total if total else 0.0,
        "hot_key_concentration": {
            "top_1_percent": hot_share(frequencies, 0.01, total),
            "top_5_percent": hot_share(frequencies, 0.05, total),
            "top_10_percent": hot_share(frequencies, 0.10, total),
        },
        "working_set": {
            "unique_layer_expert_keys": len(frequencies),
            "unique_payload_bytes": unique_payload,
            "maximum_observed_unique_keys": len(frequencies),
        },
        "prefill_decode_occurrences": phase_counts,
        "reuse_scope": {
            "cross_token_reuse_occurrences": cross_token_reuse,
            "cross_token_reuse_percentage": cross_token_reuse / total if total else 0.0,
            "cross_step_reuse_occurrences": cross_step_reuse,
            "cross_step_reuse_percentage": cross_step_reuse / total if total else 0.0,
            "cross_phase_reuse_occurrences": cross_phase_reuse,
        },
        "payload_bytes_requested": payload_requested,
        "potential_cacheability": {
            "repeated_payload_bytes": repeated_payload,
            "repeated_payload_percentage": repeated_payload / payload_requested if payload_requested else 0.0,
            "compulsory_unique_payload_bytes": unique_payload,
            "descriptive_basis": "repeated logical requests are cacheable in principle; no policy or budget replay is performed here",
        },
    }


def compare_with_control(control: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    control_reuse = control["reuse_distance"]
    current_reuse = current["reuse_distance"]
    control_hot = control["hot_key_concentration"]
    current_hot = current["hot_key_concentration"]
    control_scope = control["reuse_scope"]
    current_scope = current["reuse_scope"]
    control_cache = control["potential_cacheability"]
    current_cache = current["potential_cacheability"]

    def delta(current_value: Any, control_value: Any) -> Any:
        if current_value is None or control_value is None:
            return None
        return current_value - control_value

    return {
        "occurrence_count": {
            "control": control["total_expert_occurrences"],
            "workload": current["total_expert_occurrences"],
            "difference": current["total_expert_occurrences"] - control["total_expert_occurrences"],
        },
        "unique_key_ratio": {
            "control": control["unique_key_ratio"],
            "workload": current["unique_key_ratio"],
            "difference": current["unique_key_ratio"] - control["unique_key_ratio"],
        },
        "first_reuse_distance": {
            "control": control_reuse["first_reuse_distance"],
            "workload": current_reuse["first_reuse_distance"],
            "difference": delta(current_reuse["first_reuse_distance"], control_reuse["first_reuse_distance"]),
        },
        "median_reuse_distance": {
            "control": control_reuse["median"],
            "workload": current_reuse["median"],
            "difference": delta(current_reuse["median"], control_reuse["median"]),
        },
        "maximum_reuse_distance": {
            "control": control_reuse["maximum"],
            "workload": current_reuse["maximum"],
            "difference": delta(current_reuse["maximum"], control_reuse["maximum"]),
        },
        "hot_set_concentration": {
            name: {
                "control": control_hot[name]["percentage"],
                "workload": current_hot[name]["percentage"],
                "difference": current_hot[name]["percentage"] - control_hot[name]["percentage"],
            }
            for name in ("top_1_percent", "top_5_percent", "top_10_percent")
        },
        "prefill_decode_reuse": {
            "control_prefill_occurrences": control["prefill_decode_occurrences"]["prefill"],
            "workload_prefill_occurrences": current["prefill_decode_occurrences"]["prefill"],
            "control_decode_occurrences": control["prefill_decode_occurrences"]["decode"],
            "workload_decode_occurrences": current["prefill_decode_occurrences"]["decode"],
            "control_cross_token_reuse_percentage": control_scope["cross_token_reuse_percentage"],
            "workload_cross_token_reuse_percentage": current_scope["cross_token_reuse_percentage"],
            "control_cross_step_reuse_percentage": control_scope["cross_step_reuse_percentage"],
            "workload_cross_step_reuse_percentage": current_scope["cross_step_reuse_percentage"],
        },
        "potential_cacheability": {
            "control_repeated_key_percentage": control["repeated_key_percentage"],
            "workload_repeated_key_percentage": current["repeated_key_percentage"],
            "control_repeated_payload_percentage": control_cache["repeated_payload_percentage"],
            "workload_repeated_payload_percentage": current_cache["repeated_payload_percentage"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fixture-manifest", type=Path, default=Path("models/qwen3-30b-a3b/m5.2-01-representative-fixture-manifest-v1.json"))
    parser.add_argument("--repeat-report", type=Path, default=Path("models/qwen3-30b-a3b/m5.2-01-repeatability-v1.json"))
    parser.add_argument("--trace-root", type=Path, default=Path("models/qwen3-30b-a3b/m5.2-01-traces"))
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    fixture_manifest = load_json(root / args.fixture_manifest)
    repeat_report = load_json(root / args.repeat_report)
    control_trace = load_json(root / CONTROL_TRACE)
    control_manifest = load_json(root / CONTROL_MANIFEST)
    assert sha256(root / CONTROL_TRACE) == control_manifest["trace"]["sha256"]
    assert control_manifest["repeat"]["byte_identical"] is True

    fixture_by_id = {fixture["fixture_id"]: fixture for fixture in fixture_manifest["fixtures"]}
    repeat_by_id = {item["fixture_id"]: item for item in repeat_report["fixtures"]}
    assert set(fixture_by_id) == set(repeat_by_id)
    trace_paths: dict[str, Path] = {"tier_a_control": root / CONTROL_TRACE}
    for fixture_id in fixture_by_id:
        if fixture_id != "tier_a_control":
            trace_paths[fixture_id] = root / args.trace_root / f"{fixture_id}.json"

    per_fixture: list[dict[str, Any]] = []
    for fixture_id, fixture in fixture_by_id.items():
        trace_path = trace_paths[fixture_id]
        trace = load_json(trace_path)
        validate_records(trace, fixture, fixture_id == "tier_a_control")
        trace_digest = sha256(trace_path)
        repeat = repeat_by_id[fixture_id]
        assert repeat["trace_sha256"] == trace_digest
        assert repeat["byte_identical_repeat"] is True
        assert repeat["repeat_count"] == 2
        assert len(set(repeat["repeat_sha256"])) == 1
        assert repeat["repeat_sha256"][0] == trace_digest
        stats = statistics(trace, fixture_id)
        per_fixture.append(
            {
                "fixture": fixture,
                "trace": {
                    "path": trace_path.relative_to(root).as_posix(),
                    "bytes": trace_path.stat().st_size,
                    "sha256": trace_digest,
                    "repeat_sha256": repeat["repeat_sha256"],
                    "record_count": len(trace["records"]),
                    "generated_token_ids": trace["expected_generated_token_ids"],
                },
                "statistics": stats,
            }
        )

    control_stats = next(item["statistics"] for item in per_fixture if item["fixture"]["fixture_id"] == "tier_a_control")
    comparisons = []
    for item in per_fixture:
        if item["fixture"]["fixture_id"] != "tier_a_control":
            comparisons.append(
                {
                    "fixture_id": item["fixture"]["fixture_id"],
                    "comparison": compare_with_control(control_stats, item["statistics"]),
                }
            )
    all_stats = statistics(
        {
            "records": [
                record
                for item in per_fixture
                for record in load_json(root / item["trace"]["path"])["records"]
            ]
        },
        "corpus",
    )
    corpus_manifest = {
        "schema": "colibri-qwen3-moe-m5.2-01-trace-corpus-manifest-v1",
        "schema_version": 1,
        "corpus_id": fixture_manifest["corpus_id"],
        "task": "M5.2-01",
        "baseline": {
            "baseline_id": "qwen3-30b-a3b-colibri-f32-windows-x64-v1",
            "release_id": "colibri-lite-rs-m4-qwen3-30b-a3b-f32-v1",
            "release_tag": "m4-full-qwen3-baseline-v1",
            "m4_performance_baseline": "models/qwen3-30b-a3b/m4.4-performance-baseline-v1.json",
            "m4_performance_baseline_sha256": sha256(root / M4_BASELINE),
            "m5_1_00_control_trace": CONTROL_TRACE.as_posix(),
            "m5_1_00_control_trace_sha256": sha256(root / CONTROL_TRACE),
            "m5_1_03_results": "models/qwen3-30b-a3b/m5.1-03-full-model-cache-results-v1.json",
            "m5_1_03_results_sha256": sha256(root / M5_1_03_RESULTS),
        },
        "model": fixture_manifest["model"],
        "trace_schema": {
            "path": "models/qwen3-30b-a3b/m5.2-01-ordered-expert-trace-schema-v2.json",
            "schema": EXPECTED_TRACE_SCHEMA,
        },
        "fixtures": [
            {
                "fixture_id": item["fixture"]["fixture_id"],
                "workload_class": item["fixture"]["workload_class"],
                "classification": item["fixture"]["classification"],
                "trace_path": item["trace"]["path"],
                "trace_sha256": item["trace"]["sha256"],
                "repeat_sha256": item["trace"]["repeat_sha256"],
                "record_count": item["trace"]["record_count"],
                "generated_token_ids": item["trace"]["generated_token_ids"],
            }
            for item in per_fixture
        ],
        "aggregate_counters": {
            "fixture_count": len(per_fixture),
            "trace_count": len(per_fixture),
            "repeat_capture_count": sum(item["trace"]["record_count"] * 2 for item in per_fixture),
            "total_expert_occurrences": all_stats["total_expert_occurrences"],
            "unique_layer_expert_keys_union": len(
                {
                    record["layer_expert_key"]
                    for item in per_fixture
                    for record in load_json(root / item["trace"]["path"])["records"]
                }
            ),
            "payload_bytes_requested": all_stats["payload_bytes_requested"],
        },
        "deterministic_serialization": "UTF-8 JSON, sorted keys, compact separators, trailing newline; traces retain fixed Rust field order",
    }
    aggregate = {
        "schema": "colibri-qwen3-moe-m5.2-01-trace-corpus-aggregate-v1",
        "schema_version": 1,
        "corpus_id": fixture_manifest["corpus_id"],
        "trace_schema": EXPECTED_TRACE_SCHEMA,
        "canonical_artifact_root_sha256": EXPECTED_ARTIFACT_ROOT,
        "per_fixture": per_fixture,
        "corpus_wide": all_stats,
        "comparison_with_frozen_tier_a": comparisons,
        "eight_gib_recommendation_classification": {
            "classification": "inconclusive",
            "basis": "The corpus contains materially shorter Tier-B prompts, a longer context, and a longer decode; descriptive trace diversity alone cannot establish the 8 GiB hit-rate generalization without the prohibited M5.2-02 replay.",
            "cache_policy_recommendation_made": False,
        },
        "simulation_executed": False,
    }
    for relative, payload in ((args.corpus_manifest, corpus_manifest), (args.aggregate, aggregate)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"corpus_manifest": corpus_manifest, "aggregate": aggregate}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
