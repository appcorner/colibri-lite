"""Aggregate and validate M5.3-03 compute-profile evidence.

This analysis is descriptive only. It does not simulate a cache policy, alter
the runtime, or infer throughput from profile timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "colibri-qwen3-moe-m5.3-03-compute-profile-results-v1"
AGGREGATE_SCHEMA = "colibri-qwen3-moe-m5.3-03-compute-profile-aggregate-v1"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def detailed_runs(document: dict[str, Any]) -> list[dict[str, Any]]:
    require(document.get("schema") == RESULT_SCHEMA, "profile result schema mismatch")
    require(document.get("status") == "complete", "profile result is not complete")
    runs = [run for run in document["runs"] if run["profile_mode"] == "detailed"]
    require(len(runs) == 8, "expected eight detailed profile runs")
    return runs


def aggregate_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for event in events:
        operation = event["operation"]
        item = totals.setdefault(
            operation,
            {
                "calls": 0,
                "total_nanos": 0,
                "exclusive_nanos": 0,
                "estimated_flops": 0,
                "input_bytes": 0,
                "output_bytes": 0,
            },
        )
        for key in item:
            item[key] += event[key]
    return totals


def phase_events(profile: dict[str, Any], phase_prefix: str) -> list[dict[str, Any]]:
    return [event for event in profile["events"] if event["phase"].startswith(phase_prefix)]


def high_level_phase_totals(profile: dict[str, Any], phase_prefix: str) -> dict[str, int]:
    events = phase_events(profile, phase_prefix)
    totals = defaultdict(int)
    for event in events:
        operation = event["operation"]
        if operation.startswith("decoder.layer."):
            totals["decoder_layers"] += event["total_nanos"]
        elif operation == "embedding.lookup":
            totals["embedding"] += event["total_nanos"]
        elif operation == "final_norm":
            totals["final_norm"] += event["total_nanos"]
        elif operation == "lm_head":
            totals["lm_head"] += event["total_nanos"]
        elif operation == "cache.append":
            totals["cache_append"] += event["total_nanos"]
    return dict(sorted(totals.items()))


def matrix_aggregates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shapes: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for event in events:
        for matrix in event["matrices"]:
            key = (event["operation"], matrix["rows"], matrix["outputs"], matrix["inputs"])
            item = shapes.setdefault(
                key,
                {
                    "operation": event["operation"],
                    "rows": matrix["rows"],
                    "outputs": matrix["outputs"],
                    "inputs": matrix["inputs"],
                    "calls": 0,
                    "estimated_flops": 0,
                    "input_bytes": 0,
                    "weight_bytes": 0,
                    "output_bytes": 0,
                    "total_nanos": 0,
                    "exclusive_nanos": 0,
                },
            )
            item["calls"] += matrix["calls"]
            item["estimated_flops"] += matrix["estimated_flops"]
            item["input_bytes"] += matrix["input_bytes"]
            item["weight_bytes"] += matrix["weight_bytes"]
            item["output_bytes"] += matrix["output_bytes"]
            item["total_nanos"] += event["total_nanos"]
            item["exclusive_nanos"] += event["exclusive_nanos"]
    return sorted(shapes.values(), key=lambda item: (-item["total_nanos"], item["operation"], item["rows"], item["outputs"], item["inputs"]))


def run_analysis(run: dict[str, Any]) -> dict[str, Any]:
    profile = run["profile"]
    events = profile["events"]
    event_totals = aggregate_events(events)
    runtime_total = run["runtime"]["timing"]["total_seconds"] * 1.0e9
    model_total = event_totals.get("model.total", {}).get("total_nanos", 0)
    direct = high_level_phase_totals(profile, "prefill")
    decode_phases = sorted({event["phase"] for event in events if event["phase"].startswith("decode_")})
    decode_totals = {phase: high_level_phase_totals(profile, phase) for phase in decode_phases}
    storage = run["runtime"]["storage_metrics"]
    storage_timing = storage["timing"]
    storage_nanos = storage_timing["path"]["total_nanos"]
    expert_compute_nanos = event_totals.get("expert.mlp.total", {}).get("total_nanos", 0)
    runtime_timing = run["runtime"]["timing"]
    decode_seconds = runtime_timing["decode_seconds"]
    if decode_seconds == 0:
        decode_seconds = 0.0
    return {
        "profile_mode": run["profile_mode"],
        "fixture_id": run["fixture_id"],
        "budget_gib": run["budget_gib"],
        "runtime_total_nanos": runtime_total,
        "profile_model_total_nanos": model_total,
        "profile_to_runtime_ratio": model_total / runtime_total if runtime_total else None,
        "scope_count": profile["scope_count"],
        "event_count": len(events),
        "runtime_timing": {**runtime_timing, "decode_seconds": decode_seconds},
        "cache": run["runtime"]["cache"],
        "storage": {
            "path_total_nanos": storage_nanos,
            "expert_load_nanos": storage_timing["path"]["expert_load_nanos"],
            "cache_lookup_nanos": storage_timing["path"]["cache_lookup_nanos"],
            "reader_read_nanos": storage_timing["reader"]["read_nanos"],
            "reader_hash_nanos": storage_timing["reader"]["hash_nanos"],
        },
        "expert_compute_nanos": expert_compute_nanos,
        "high_level_prefill_nanos": direct,
        "high_level_decode_nanos": decode_totals,
        "operation_totals": event_totals,
        "matrix_shapes": matrix_aggregates(events),
        "top_operations_by_total": sorted(
            ({"operation": operation, **values} for operation, values in event_totals.items()),
            key=lambda item: (-item["total_nanos"], item["operation"]),
        )[:20],
        "top_operations_by_exclusive": sorted(
            ({"operation": operation, **values} for operation, values in event_totals.items()),
            key=lambda item: (-item["exclusive_nanos"], item["operation"]),
        )[:20],
    }


def summarize_modes(document: dict[str, Any]) -> dict[str, Any]:
    rows = {(run["profile_mode"], run["fixture_id"], run["budget_gib"]): run for run in document["runs"]}
    tier_a = {}
    for mode in ("disabled", "coarse", "detailed"):
        run = rows[(mode, "tier_a_control", 8)]
        tier_a[mode] = {
            "total_seconds": run["runtime"]["timing"]["total_seconds"],
            "scope_count": run["profile"]["scope_count"],
            "event_count": len(run["profile"]["events"]),
            "non_timing_fingerprint": run["non_timing_fingerprint"],
        }
    return {"tier_a_control_8gib": tier_a}


def build(document: dict[str, Any]) -> dict[str, Any]:
    runs = detailed_runs(document)
    analyses = [run_analysis(run) for run in runs]
    by_fixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for analysis in analyses:
        by_fixture[analysis["fixture_id"]].append(analysis)
    fixture_summary = {}
    for fixture_id, values in sorted(by_fixture.items()):
        fixture_summary[fixture_id] = {
            "budgets": sorted(value["budget_gib"] for value in values),
            "total_seconds": {str(value["budget_gib"]): value["runtime_timing"]["total_seconds"] for value in values},
            "prefill_seconds": {str(value["budget_gib"]): value["runtime_timing"]["prefill_seconds"] for value in values},
            "decode_seconds": {str(value["budget_gib"]): value["runtime_timing"]["decode_seconds"] for value in values},
            "cache_hits": {str(value["budget_gib"]): value["cache"]["hits"] for value in values},
            "cache_loads": {str(value["budget_gib"]): value["cache"]["loads"] for value in values},
            "expert_bytes_loaded": {str(value["budget_gib"]): value["cache"]["bytes_read"] for value in values},
            "storage_seconds": {str(value["budget_gib"]): value["storage"]["path_total_nanos"] / 1.0e9 for value in values},
            "expert_compute_seconds": {str(value["budget_gib"]): value["expert_compute_nanos"] / 1.0e9 for value in values},
            "top_operations_by_total": {
                str(value["budget_gib"]): value["top_operations_by_total"][:10] for value in values
            },
            "top_matrix_shapes": {
                str(value["budget_gib"]): value["matrix_shapes"][:10] for value in values
            },
        }
    return {
        "schema": AGGREGATE_SCHEMA,
        "schema_version": 1,
        "task": "M5.3-03",
        "result_sha256_source": None,
        "references": document["references"],
        "selected_fixture_ids": document["selected_fixture_ids"],
        "budgets_binary_gib": document["budgets_binary_gib"],
        "profile_modes": document["profile_modes"],
        "validation": {
            "result_status_complete": True,
            "detailed_run_count": len(runs),
            "all_simulation_comparisons_exact": document["summary"]["all_simulation_comparisons_exact"],
            "all_correctness_pass": document["summary"]["all_correctness_pass"],
            "mode_non_timing_identity_pass": all(item["non_timing_identical"] for item in document["summary"]["mode_comparisons"]),
        },
        "profiling_overhead_comparison": summarize_modes(document),
        "fixture_summary": fixture_summary,
        "runs": analyses,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("models/qwen3-30b-a3b/m5.3-03-compute-profile-results-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("models/qwen3-30b-a3b/m5.3-03-compute-profile-aggregate-v1.json"))
    args = parser.parse_args()
    document = load(args.result)
    output = build(document)
    output["result_sha256_source"] = hashlib.sha256(args.result.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
