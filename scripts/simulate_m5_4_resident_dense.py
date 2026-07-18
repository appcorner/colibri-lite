"""Simulation-only M5.4 resident-dense plus strict-global-LRU study.

This study replays the validated M5.2 representative corpus and does not
invoke Rust, load model payloads, or change production runtime behavior. It
models total-RAM budgets by reserving the dense artifact and fixed runtime
components before assigning the remaining bytes to the existing strict global
LRU expert cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# When invoked as ``python scripts/<tool>.py``, Python places ``scripts`` on
# sys.path rather than the repository root.  Add the root for the existing
# simulation modules while keeping their validation contracts centralized.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import simulate_m5_1_memory_hierarchy as m51
from scripts import simulate_m5_2_corpus_cache as m52


GIB = 1024**3
EXPECTED_ARTIFACT_ROOT = m52.EXPECTED_ARTIFACT_ROOT
EXPECTED_M52_INPUT_SHA = "d040e505c9ab87b65935f11b68e8fc65aa4b496bb02f3d10832b98eadaf80b5b"
EXPECTED_M52_RUNTIME_SHA = "0a0b964eaca9de55f3244f45b275b8d386b66b448a701a35377fbf85631ae870"
M52_INPUT_REL = "models/qwen3-30b-a3b/m5.2-02-simulation-input-v1.json"
M52_RUNTIME_REL = "models/qwen3-30b-a3b/m5.2-03-runtime-cache-results-v1.json"
BASELINE_REL = "models/qwen3-30b-a3b/m4.4-performance-baseline-v1.json"
M52_BUDGETS = (8, 12, 16, 24, 32, 48)
CONFIGURATIONS = ("streamed_dense", "resident_dense")
M52_RUNTIME_FIXTURES = (
    "tier_a_control",
    "tier_b_short_thai",
    "tier_b_special_token",
    "tier_b_code_newline",
    "long_english_context",
    "long_decode_english",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_bytes(components: dict[str, int], configuration: str) -> int:
    if configuration == "resident_dense":
        dense = components["dense_artifact_bytes"]
    elif configuration == "streamed_dense":
        dense = components["dense_stream_buffer_bytes"]
    else:
        raise ValueError(f"unknown dense configuration: {configuration}")
    return (
        dense
        + components["decoded_expert_buffer_bytes"]
        + components["runtime_structures_bytes"]
        + components["safety_reserve_bytes"]
    )


def validate_runtime_evidence(root: Path, validated: dict[str, Any]) -> dict[str, int]:
    runtime_path = root / M52_RUNTIME_REL
    if sha256(runtime_path) != EXPECTED_M52_RUNTIME_SHA:
        raise ValueError("M5.2 runtime result hash mismatch")
    runtime = load_json(runtime_path)
    if runtime["schema"] != "colibri-qwen3-moe-m5.2-03-runtime-cache-results-v1":
        raise ValueError("unsupported M5.2 runtime result schema")
    if runtime["artifact"]["root_sha256"] != EXPECTED_ARTIFACT_ROOT:
        raise ValueError("M5.2 runtime artifact identity mismatch")
    if tuple(runtime["selected_fixture_ids"]) != M52_RUNTIME_FIXTURES:
        raise ValueError("M5.2 runtime fixture set mismatch")

    dense_reads: dict[str, int] = {}
    for run in runtime["runs"]:
        fixture_id = run["fixture_id"]
        value = int(run["runtime"]["io"]["dense_bytes_read"])
        previous = dense_reads.get(fixture_id)
        if previous is None:
            dense_reads[fixture_id] = value
        elif previous != value:
            raise ValueError(f"dense read evidence varies by budget/repeat for {fixture_id}")
    if set(dense_reads) != set(M52_RUNTIME_FIXTURES):
        raise ValueError("M5.2 runtime dense-read evidence is incomplete")

    for fixture_id, trace in validated["traces"].items():
        if len(trace["records"]) == 0:
            raise ValueError(f"empty trace for {fixture_id}")
    return dense_reads


def validate_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    input_path = root / M52_INPUT_REL
    if sha256(input_path) != EXPECTED_M52_INPUT_SHA:
        raise ValueError("M5.2 simulation input hash mismatch")
    input_doc = load_json(input_path)
    validated = m52.validate_corpus(root, input_doc)
    baseline = validated["baseline"]
    components = m51.dense_components(baseline)
    dense_reads = validate_runtime_evidence(root, validated)
    return validated, {"baseline": baseline, "components": components}, dense_reads


def simulate_row(
    records: list[dict[str, Any]],
    fixture_id: str,
    budget_gib: int,
    configuration: str,
    dense_read_bytes: int,
    components: dict[str, int],
    weights: dict[int, int],
) -> dict[str, Any]:
    total_budget = budget_gib * GIB
    fixed = fixed_bytes(components, configuration)
    dense_resident = configuration == "resident_dense"
    if total_budget < fixed:
        return {
            "fixture_id": fixture_id,
            "budget_gib": budget_gib,
            "configuration": configuration,
            "feasible": False,
            "total_ram_budget_bytes": total_budget,
            "fixed_overhead_bytes": fixed,
            "reason": "fixed dense/runtime components exceed total budget",
        }

    cache_budget = total_budget - fixed
    cache, _ = m52.run_cache(records, cache_budget, "global_lru", weights)
    cache_result = cache.result(records, [False] * len(records))
    requested_expert_bytes = len(records) * m52.PAYLOAD_BYTES
    remaining_dense_bytes = 0 if dense_resident else dense_read_bytes
    total_reads = remaining_dense_bytes + cache.loaded_bytes
    baseline_reads = dense_read_bytes + requested_expert_bytes
    row = {
        "fixture_id": fixture_id,
        "budget_gib": budget_gib,
        "configuration": configuration,
        "feasible": True,
        "total_ram_budget_bytes": total_budget,
        "fixed_overhead_bytes": fixed,
        "dense_resident_bytes": components["dense_artifact_bytes"] if dense_resident else 0,
        "usable_expert_cache_bytes": cache_budget,
        "dense_bytes_read": remaining_dense_bytes,
        "baseline_dense_bytes_read": dense_read_bytes,
        "requested_expert_bytes": requested_expert_bytes,
        "expert_bytes_loaded": cache.loaded_bytes,
        "expert_bytes_avoided": cache.avoided_bytes,
        "total_modeled_logical_reads": total_reads,
        "baseline_modeled_logical_reads": baseline_reads,
        "total_logical_read_reduction_pct": 100.0 * (baseline_reads - total_reads) / baseline_reads,
        "dense_logical_read_reduction_pct": 100.0 if dense_resident else 0.0,
        **cache_result,
    }
    validate_row(row, components)
    return row


def validate_row(row: dict[str, Any], components: dict[str, int]) -> None:
    if not row["feasible"]:
        return
    if row["hits"] + row["misses"] != row["requests"]:
        raise ValueError("cache accounting: hits + misses != requests")
    if row["loads"] != row["misses"]:
        raise ValueError("cache accounting: loads != misses")
    if row["expert_bytes_loaded"] + row["expert_bytes_avoided"] != row["requested_expert_bytes"]:
        raise ValueError("expert accounting mismatch")
    if row["peak_resident_bytes"] > row["usable_expert_cache_bytes"]:
        raise ValueError("cache exceeds usable expert budget")
    if row["fixed_overhead_bytes"] + row["peak_resident_bytes"] > row["total_ram_budget_bytes"]:
        raise ValueError("total resident budget exceeded")
    if row["configuration"] == "resident_dense":
        if row["dense_bytes_read"] != 0:
            raise ValueError("resident dense still reports dense logical reads")
        if row["dense_resident_bytes"] != components["dense_artifact_bytes"]:
            raise ValueError("resident dense byte accounting mismatch")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        return {"feasible": False, "fixture_count": len(rows)}
    requests = sum(row["requests"] for row in feasible)
    expert_requested = sum(row["requested_expert_bytes"] for row in feasible)
    expert_avoided = sum(row["expert_bytes_avoided"] for row in feasible)
    baseline_reads = sum(row["baseline_modeled_logical_reads"] for row in feasible)
    total_reads = sum(row["total_modeled_logical_reads"] for row in feasible)
    return {
        "feasible": len(feasible) == len(rows),
        "fixture_count": len(rows),
        "requests": requests,
        "expert_bytes_requested": expert_requested,
        "expert_bytes_loaded": sum(row["expert_bytes_loaded"] for row in feasible),
        "expert_bytes_avoided": expert_avoided,
        "expert_byte_hit_rate": expert_avoided / expert_requested if expert_requested else 0.0,
        "baseline_modeled_logical_reads": baseline_reads,
        "total_modeled_logical_reads": total_reads,
        "total_logical_read_reduction_pct": 100.0 * (baseline_reads - total_reads) / baseline_reads if baseline_reads else 0.0,
        "peak_cache_bytes": max(row["peak_resident_bytes"] for row in feasible),
        "zero_hit_fixture_count": sum(row["hits"] == 0 for row in feasible),
    }


def run(root: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    validated, baseline_data, dense_reads = validate_inputs(root)
    baseline = baseline_data["baseline"]
    components = baseline_data["components"]
    traces = validated["traces"]
    weights = validated["layer_weights"]
    rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for budget_gib in M52_BUDGETS:
        for configuration in CONFIGURATIONS:
            group = [
                simulate_row(
                    traces[fixture_id]["records"],
                    fixture_id,
                    budget_gib,
                    configuration,
                    dense_reads[fixture_id],
                    components,
                    weights,
                )
                for fixture_id in M52_RUNTIME_FIXTURES
            ]
            rows.extend(group)
            summary = aggregate(group)
            summary.update({"budget_gib": budget_gib, "configuration": configuration})
            aggregates.append(summary)

    result = {
        "schema": "colibri-qwen3-moe-m5.4-01-resident-dense-simulation-v1",
        "schema_version": 1,
        "task": "M5.4-01",
        "simulation_only": True,
        "artifact_root_sha256": EXPECTED_ARTIFACT_ROOT,
        "baseline_id": baseline["baseline_id"],
        "model_provenance": {
            "model_id": baseline["model_identity"]["repository"],
            "model_revision": baseline["model_identity"]["revision"],
            "source_url": "https://huggingface.co/Qwen/Qwen3-30B-A3B",
            "license": "Apache-2.0",
            "artifact_format_version": baseline["model_identity"]["artifact_format_version"],
            "conversion_reference": "models/qwen3-30b-a3b/m4-release-provenance-v1.json",
            "tensor_inventory_reference": "models/qwen3-30b-a3b/model-manifest-v1.json",
            "tool": "scripts/simulate_m5_4_resident_dense.py",
            "tool_version": "m5.4-01-v1",
            "generation_date": "2026-07-18",
        },
        "references": {
            "m5.2_input_path": M52_INPUT_REL,
            "m5.2_input_sha256": EXPECTED_M52_INPUT_SHA,
            "m5.2_runtime_results_path": M52_RUNTIME_REL,
            "m5.2_runtime_results_sha256": EXPECTED_M52_RUNTIME_SHA,
            "baseline_path": BASELINE_REL,
            "baseline_sha256": sha256(root / BASELINE_REL),
        },
        "accounting": {
            "unit": "binary_gib",
            "cache_policy": "strict_global_lru",
            "budget_semantics": "total RAM budget; fixed components reserved before expert payload cache",
            "dense_read_source": "M5.2-03 full-runtime dense_bytes_read, per fixture",
            "physical_io_measured": False,
            "throughput_measured": False,
            "components": components,
        },
        "budgets_binary_gib": list(M52_BUDGETS),
        "configurations": list(CONFIGURATIONS),
        "selected_fixture_ids": list(M52_RUNTIME_FIXTURES),
        "omitted_corpus_fixture_ids": [
            fixture_id for fixture_id in m52.EXPECTED_M52_FIXTURES if fixture_id not in M52_RUNTIME_FIXTURES
        ],
        "per_fixture": rows,
        "aggregates": aggregates,
        "validation": {
            "corpus_validated_before_simulation": True,
            "runtime_dense_read_evidence_validated": True,
            "cache_accounting_invariants_validated": True,
            "total_budget_invariants_validated": True,
            "numerical_execution_changed": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    report_path.write_text(render_report(result), encoding="utf-8", newline="\n")
    return result


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# M5.4-01 Resident-Dense plus Strict Global-LRU Simulation",
        "",
        "This is a deterministic, simulation-only study over the six-workload M5.2 representative full-runtime subset. It does not invoke Rust, load model payloads, or change runtime behavior.",
        "",
        f"Canonical artifact SHA-256: `{result['artifact_root_sha256']}`.",
        f"M5.2 input SHA-256: `{result['references']['m5.2_input_sha256']}`.",
        f"M5.2 runtime evidence SHA-256: `{result['references']['m5.2_runtime_results_sha256']}`.",
        "",
        "## Accounting",
        "",
        "Total binary-GiB budgets reserve the dense artifact or streamed dense buffer, decoded expert buffer, runtime structures, and a 256 MiB safety reserve before assigning the remainder to strict global LRU expert payload capacity. Dense logical-read values come from the recorded M5.2 full-runtime evidence. Physical I/O and throughput are not measured.",
        "",
        "## Corpus aggregate",
        "",
        "| Budget | Configuration | Feasible | Expert cache | Expert byte hit | Total logical-read reduction | Zero-hit fixtures |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in result["aggregates"]:
        lines.append(
            f"| {row['budget_gib']} GiB | `{row['configuration']}` | {row['feasible']} | {next(item['usable_expert_cache_bytes'] for item in result['per_fixture'] if item['budget_gib'] == row['budget_gib'] and item['configuration'] == row['configuration'] and item['feasible']) / GIB:.3f} GiB | {row.get('expert_byte_hit_rate', 0.0):.2%} | {row.get('total_logical_read_reduction_pct', 0.0):.2f}% | {row.get('zero_hit_fixture_count', 0)} |"
        )
    lines += [
        "",
        "## Resident-dense decision evidence",
        "",
        "Resident dense removes the modeled dense logical-read component, but it reduces the expert-cache budget by the dense artifact size. The result is workload- and RAM-dependent; this simulation does not claim a latency improvement.",
        "",
        "| Budget | Streamed-dense read reduction | Resident-dense read reduction | Resident-dense feasible |",
        "|---:|---:|---:|---|",
    ]
    for budget in M52_BUDGETS:
        streamed = next(row for row in result["aggregates"] if row["budget_gib"] == budget and row["configuration"] == "streamed_dense")
        resident = next(row for row in result["aggregates"] if row["budget_gib"] == budget and row["configuration"] == "resident_dense")
        lines.append(
            f"| {budget} GiB | {streamed.get('total_logical_read_reduction_pct', 0.0):.2f}% | {resident.get('total_logical_read_reduction_pct', 0.0):.2f}% | {resident['feasible']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "Classification: `simulation_complete_candidate_review_required`.",
        "",
        "This evidence is sufficient to select a narrowly scoped resident-dense runtime prototype for separate review only if a measured end-to-end experiment is authorized. It does not authorize runtime changes, a production preset, or a throughput claim. The prototype must preserve frozen F32 correctness, strict total-budget accounting, and the reference reader unless a later reviewed task changes those decisions.",
        "",
        "## Limitations",
        "",
        "The dense-read values are inherited from M5.2 runtime evidence and are used as a modeled baseline for resident-dense elimination. The resident-dense path itself was not executed. The two M5.2 corpus traces omitted from full-runtime validation are not included here. Filesystem cache state, physical I/O, latency, throughput, allocator overhead, and concurrent behavior remain unmeasured.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    run(root, output, report)
    print(json.dumps({"output": str(output), "report": str(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
