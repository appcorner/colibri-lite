#!/usr/bin/env python3
"""Build the deterministic M4.4 performance-baseline index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "colibri-qwen3-moe-m4.4-performance-baseline-v1"
BASELINE_ID = "qwen3-30b-a3b-colibri-f32-windows-x64-v1"
RUNTIME_COMMIT = "80099f05246a4450ded6f42baf6b8db5a4b2e623"
RUNTIME_BRANCH = "milestone/m4-full-qwen3"

REFERENCE_ROLES = {
    "model_manifest": "models/qwen3-30b-a3b/model-manifest-v1.json",
    "source_manifest": "models/qwen3-30b-a3b/source-manifest-v1.json",
    "tolerance_registry": "models/qwen3-30b-a3b/m4.2-tolerance-contract-registry-v1.json",
    "f32_baseline_manifest": "models/qwen3-30b-a3b/m4.3-01-f32-baseline-manifest-v1.json",
    "f32_fixtures": "models/qwen3-30b-a3b/m4.3-01-fixtures-v1.json",
    "comparison_schema": "models/qwen3-30b-a3b/m4.3-01-comparison-schema-v1.json",
    "selected_intermediates_rust": "models/qwen3-30b-a3b/m4.2-03-rust-intermediate-evidence-v1.tsv",
    "selected_intermediates_transformers": "models/qwen3-30b-a3b/m4.2-03-transformers-f32-intermediate-evidence-v1.json",
    "tier_a_rust": "models/qwen3-30b-a3b/m4.2-04-rust-short-generation-evidence-v1.tsv",
    "tier_a_transformers_f32": "models/qwen3-30b-a3b/m4.2-04-transformers-f32-generation-evidence-v1.json",
    "tier_a_transformers_bf16": "models/qwen3-30b-a3b/m4.2-04-transformers-bf16-generation-evidence-v1.json",
    "tier_b_rust": "models/qwen3-30b-a3b/m4.3-01-rust-tier-b-evidence-v1.tsv",
    "tier_b_transformers_f32": "models/qwen3-30b-a3b/m4.3-01-tier-b-transformers-f32-v1.json",
    "tier_b_transformers_tsv": "models/qwen3-30b-a3b/m4.3-01-tier-b-transformers-f32-v1.tsv",
    "f64_diagnostics": "models/qwen3-30b-a3b/m4.3-01-f64-diagnostics-v1.json",
    "resource_summary": "models/qwen3-30b-a3b/m4.2-05-baseline-summary-v1.json",
    "resource_report": "docs/reports/m4.2-05-resource-baseline.md",
    "quantization_evidence": "models/qwen3-30b-a3b/m4.3-02-quantization-evidence-v1.json",
    "precision_registry": "models/qwen3-30b-a3b/m4.3-03-tensor-precision-registry-v1.json",
    "precision_evidence": "models/qwen3-30b-a3b/m4.3-03-precision-sensitivity-evidence-v1.json",
    "degradation_evidence": "models/qwen3-30b-a3b/m4.3-04-degradation-evidence-v1.json",
    "ik_environment": "models/qwen3-30b-a3b/m4.3-05-ik-llama-environment-v1.json",
    "ik_metrics": "models/qwen3-30b-a3b/m4.3-05-ik-llama-run-metrics-v1.json",
    "ik_report": "docs/reports/m4.3-05-ik-llama-comparison.md",
    "candidate_registry": "models/qwen3-30b-a3b/m4.3-06-candidate-status-registry-v1.json",
    "decision_summary": "models/qwen3-30b-a3b/m4.3-06-decision-summary-v1.json",
    "decision_report": "docs/reports/m4.3-06-candidate-decision.md",
    "phase_closure": "docs/reports/m4.3-phase-closure.md",
    "f32_invariants": "docs/m4.3-f32-baseline-invariants.md",
    "optimization_invariants": "docs/m4.2-optimization-invariants.md",
    "memory_roadmap": "docs/m4.3-next-phase-memory-hierarchy-roadmap.md",
}

EXPECTED_SCHEMA_VERSIONS = {
    "resource_summary": 1,
    "f32_baseline_manifest": 1,
    "f32_fixtures": 1,
    "comparison_schema": 1,
    "tolerance_registry": 1,
    "candidate_registry": "m4.3-06-candidate-status-registry-v1",
    "decision_summary": "m4.3-06-decision-summary-v1",
    "ik_environment": "m4.3-05-ik-llama-environment-v1",
    "ik_metrics": "m4.3-05-ik-llama-run-metrics-v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return value


def reference(root: Path, role: str, path_text: str) -> dict[str, Any]:
    path = root / path_text
    if not path.is_file():
        raise ValueError(f"missing evidence reference {role}: {path_text}")
    return {"role": role, "path": path_text.replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size}


def validate_evidence_schemas(manifests: dict[str, dict[str, Any]]) -> None:
    """Reject references whose version marker no longer matches the contract."""
    for role, expected in EXPECTED_SCHEMA_VERSIONS.items():
        document = manifests[role]
        actual = document.get("schema")
        if actual is None:
            actual = document.get("schema_version")
        if actual != expected:
            raise ValueError(f"evidence schema mismatch for {role}: expected {expected!r}, got {actual!r}")


def validate_unique_baseline_id(root: Path) -> None:
    """A baseline ID names one canonical index in the model directory."""
    matches: list[Path] = []
    for path in (root / "models" / "qwen3-30b-a3b").glob("m4.4-performance-baseline-v1.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and document.get("baseline_id") == BASELINE_ID:
            matches.append(path)
    if len(matches) > 1:
        names = ", ".join(str(path.relative_to(root)) for path in matches)
        raise ValueError(f"baseline ID {BASELINE_ID!r} is duplicated: {names}")


def build(root: Path) -> dict[str, Any]:
    refs = {role: reference(root, role, path) for role, path in REFERENCE_ROLES.items()}
    manifests = {role: read_json(root / item["path"]) for role, item in refs.items() if item["path"].endswith(".json")}
    model = manifests["model_manifest"]
    baseline = manifests["f32_baseline_manifest"]
    resources = manifests["resource_summary"]
    candidate = manifests["candidate_registry"]
    ik_env = manifests["ik_environment"]
    ik_metrics = manifests["ik_metrics"]

    validate_evidence_schemas(manifests)
    validate_unique_baseline_id(root)

    if model["model_id"] != "Qwen/Qwen3-30B-A3B" or model["revision"] != "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39":
        raise ValueError("canonical model identity does not match the frozen Qwen3 revision")
    if model["inventory"]["component_bytes"] != 122147666917 or model["inventory"]["required_file_count"] != 57:
        raise ValueError("canonical artifact inventory changed")
    if baseline["status"] != "authoritative_unquantized_f32_baseline_frozen":
        raise ValueError("F32 baseline is not authoritative")
    statuses = {entry["item"]: entry["status"] for entry in candidate["phase_verdict"]}
    if statuses.get("int8_per_output_channel") != "rejected_full_model_candidate":
        raise ValueError("provisional INT8 candidate is incorrectly accepted")
    if statuses.get("f32_baseline") != "authoritative_and_accepted":
        raise ValueError("candidate registry does not accept F32 baseline")
    for entry in candidate["phase_verdict"]:
        if entry["item"] != "f32_baseline" and entry["status"] in {
            "accepted_for_runtime_prototype",
            "authoritative_and_accepted",
        }:
            raise ValueError(f"non-F32 candidate is marked production-accepted: {entry['item']}")
    if ik_env["model"]["quantization"] != "Q4_K_M" or not ik_env["runtime"]["mmap"]:
        raise ValueError("external reference identity is inconsistent")
    if ik_metrics["fixture"]["prompt_token_ids"] != [9707, 11, 1879, 0] or ik_metrics["short_runs"][0]["exit_status"] != 0:
        raise ValueError("external reference fixture or run status is inconsistent")
    if resources["fixture"]["generated_token_ids"] != [1096, 374]:
        raise ValueError("performance baseline fixture changed")

    performance_reference = baseline["performance_reference"]
    total_phase = resources["phases"][-1]
    if total_phase["total_bytes"] != performance_reference["logical_total_bytes"]:
        raise ValueError("performance read totals disagree")
    if resources["expert_cache"]["hits"] != 0 or resources["expert_cache"]["misses"] != resources["expert_cache"]["loads"]:
        raise ValueError("expert cache counters do not reconcile")

    performance = {
        "source": refs["resource_summary"],
        "filesystem_cache_assumption": performance_reference["filesystem_cache_assumption"],
        "logical_application_reads": True,
        "physical_device_io_measured": False,
        "initialization_seconds": {"minimum": min(performance_reference["initialization_seconds"]), "maximum": max(performance_reference["initialization_seconds"])},
        "prefill_seconds": {"minimum": min(performance_reference["prefill_seconds"]), "maximum": max(performance_reference["prefill_seconds"])},
        "prefill_tokens_per_second": {"minimum": 4 / max(performance_reference["prefill_seconds"]), "maximum": 4 / min(performance_reference["prefill_seconds"])},
        "decode_seconds_per_token": {"minimum": min(value for pair in performance_reference["decode_seconds_by_token"] for value in pair), "maximum": max(value for pair in performance_reference["decode_seconds_by_token"] for value in pair)},
        "decode_tokens_per_second": {"minimum": 1 / max(value for pair in performance_reference["decode_seconds_by_token"] for value in pair), "maximum": 1 / min(value for pair in performance_reference["decode_seconds_by_token"] for value in pair)},
        "process_peak_working_set_bytes": performance_reference["process_peak_working_set_bytes_range"],
        "process_peak_private_bytes": performance_reference["process_peak_private_bytes_range"],
        "modeled_explicit_memory_bytes": performance_reference["modeled_explicit_memory_bytes"],
        "dense_logical_bytes": performance_reference["logical_dense_bytes"],
        "expert_logical_bytes": performance_reference["logical_expert_bytes"],
        "total_logical_bytes": performance_reference["logical_total_bytes"],
        "logical_bytes_per_cached_decode_token": performance_reference["logical_bytes_per_cached_decode_token"],
        "read_amplification": performance_reference["total_read_amplification"],
        "expert_cache": resources["expert_cache"],
        "modeled_memory_components": resources["modeled_memory"],
        "kv_cache": resources["kv_cache"],
        "reconciliation": resources["reconciliation"],
    }

    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "baseline_id": BASELINE_ID,
        "creation": {
            "tool": "python/reference/build_m4_4_baseline.py",
            "tool_version": "m4.4-01-baseline-generator-v1",
            "serialization": "UTF-8 JSON, sorted keys, compact separators, trailing newline, no timestamp",
            "deterministic_payload": True,
        },
        "compatibility": {
            "canonical_artifact_is_read_only": True,
            "runtime_dtype": "F32",
            "future_variants_must_use_comparison_schema": True,
            "logical_reads_are_not_physical_io": True,
        },
        "model_identity": {
            "repository": "Qwen/Qwen3-30B-A3B",
            "revision": model["revision"],
            "architecture": model["architecture"],
            "model_type": model["model_type"],
            "canonical_root_manifest_sha256": refs["model_manifest"]["sha256"],
            "artifact_format": model["artifact_format"],
            "artifact_format_version": model["format_version"],
            "artifact_file_count": model["inventory"]["required_file_count"],
            "artifact_total_bytes": model["inventory"]["component_bytes"],
            "dense": {"bytes": model["components"]["dense"]["bytes"], "manifest_sha256": model["components"]["dense"]["manifest"]["sha256"], "payload_sha256": model["components"]["dense"]["payload"]["sha256"]},
            "experts": {"bytes": model["components"]["experts"]["bytes"], "manifest_sha256": model["components"]["experts"]["manifest"]["sha256"], "ordered_shard_set_sha256": model["components"]["experts"]["ordered_shard_set_sha256"], "shard_count": model["components"]["experts"]["shard_count"], "logical_expert_count": model["components"]["experts"]["logical_expert_count"]},
            "tokenizer": {"class": model["components"]["tokenizer"]["tokenizer_class"], "bytes": model["components"]["tokenizer"]["bytes"], "manifest_sha256": model["components"]["tokenizer"]["manifest"]["sha256"], "base_vocabulary_size": model["components"]["tokenizer"]["base_vocabulary_size"]},
            "source_dtype": model["dtypes"]["source"],
            "storage_dtype": model["dtypes"]["storage"],
            "compute_dtype": model["dtypes"]["compute"],
        },
        "runtime_identity": {
            "repository_branch": RUNTIME_BRANCH,
            "runtime_commit": RUNTIME_COMMIT,
            "runtime": "colibri-lite-rs",
            "rust_toolchain": "1.96.1",
            "target": "x86_64-pc-windows-msvc",
            "operating_system": "Windows 11 Enterprise 10.0.26200",
            "build_profile": "release",
            "feature_flags": [],
            "arithmetic": "safe scalar ordered F32",
            "threads": 8,
            "expert_cache_budget_bytes": resources["expert_cache"]["configured_budget_bytes"],
            "runtime_configuration": {"kv_capacity": resources["kv_cache"]["capacity"], "kv_type": "F32", "mmap": False, "prefetch": False, "simd": False, "parallel_experts": False},
        },
        "correctness_baseline": {
            "oracle_policy": "Transformers F32 from pinned BF16-derived weights; ordered Rust F32 is the internal reference",
            "tolerance_registry": refs["tolerance_registry"],
            "f32_baseline_manifest": refs["f32_baseline_manifest"],
            "fixtures": refs["f32_fixtures"],
            "tier_a": {"input_token_ids": [9707, 11, 1879, 0], "generated_token_ids": [1096, 374], "evidence": [refs["tier_a_rust"], refs["tier_a_transformers_f32"], refs["tier_a_transformers_bf16"]]},
            "tier_b": {"fixture_set": "six complete-forward fixtures", "evidence": [refs["tier_b_rust"], refs["tier_b_transformers_f32"], refs["tier_b_transformers_tsv"]]},
            "tier_c": {"scope": ["embedding", "RMSNorm", "attention", "router", "selected expert MLP", "final norm", "LM head", "KV update"], "evidence": [refs["selected_intermediates_rust"], refs["selected_intermediates_transformers"], refs["f64_diagnostics"]]},
            "guard_layers": [0, 1, 8, 16, 24, 32, 40, 47],
            "selected_expert_intermediates": [refs["selected_intermediates_rust"], refs["selected_intermediates_transformers"]],
            "final_norm_and_lm_head": [refs["tier_a_rust"], refs["tier_b_rust"], refs["f64_diagnostics"]],
            "kv_cache_contract": resources["kv_cache"],
            "routing_policy": "higher router score first; lower expert ID first on exact ties",
            "finite_output_required": True,
            "bounded_memory_required": True,
        },
        "performance_baseline": performance,
        "external_optimized_reference": {
            "classification": "directional_not_format_controlled",
            "environment": refs["ik_environment"],
            "metrics": refs["ik_metrics"],
            "repository_commit": ik_env["repository"]["commit"],
            "model": {"file": ik_env["model"]["file"], "format": ik_env["model"]["quantization"], "bytes": ik_env["model"]["file_bytes"]},
            "cpu_only": True,
            "threads": ik_env["runtime"]["threads"],
            "capabilities": ["mmap", "fused_moe", "flash_attention", "OpenMP", "AVX", "AVX2", "FMA", "AVX-512"],
            "prompt_tokens_per_second": ik_metrics["llama_bench"]["prompt_4_tokens"],
            "long_decode_tokens_per_second": ik_metrics["long_decode"]["decode_tokens_per_second"],
            "short_decode_tokens_per_second": {"minimum": min(item["decode_tokens_per_second"] for item in ik_metrics["short_runs"]), "maximum": max(item["decode_tokens_per_second"] for item in ik_metrics["short_runs"])},
            "memory_working_set_bytes": {"minimum": min(item["peak_working_set_bytes"] for item in ik_metrics["short_runs"]), "maximum": ik_metrics["long_decode"]["peak_working_set_bytes"]},
            "quality_equivalence_to_f32": False,
            "evidence": [refs["ik_environment"], refs["ik_metrics"], refs["ik_report"]],
        },
        "quantization_decision_registry": {"source": refs["candidate_registry"], "decision_report": refs["decision_report"], "statuses": candidate["phase_verdict"]},
        "frozen_optimization_invariants": {"source": refs["f32_invariants"], "summary": ["canonical artifact identity", "ordered F32/RMSNorm contract", "deterministic router tie policy", "Tier-A generated IDs [1096, 374]", "guard-layer router IDs", "selected expert intermediates", "finite outputs", "KV-cache shape/allocation/no-overwrite", "bounded memory", "evidence schema compatibility"]},
        "memory_hierarchy_study_inputs": {"ram_budgets_binary_gib": [1,2,4,8,16,24,32], "dense_artifact_bytes": model["components"]["dense"]["bytes"], "expert_component_bytes": model["components"]["experts"]["bytes"], "expert_payload_bytes": resources["expert_cache"]["expert_payload_bytes"], "expert_request_occurrences": resources["expert_cache"]["occurrences"], "unique_layer_expert_keys": resources["expert_cache"]["unique_layer_expert_requests"], "reuse_distance": resources["expert_cache"]["reuse_distance"], "baseline_cache_policy": {"budget_bytes": resources["expert_cache"]["configured_budget_bytes"], "capacity": resources["expert_cache"]["theoretical_capacity"], "hits": resources["expert_cache"]["hits"]}, "cache_entry_overhead_assumption": "not separately measured; simulation must parameterize metadata overhead", "trace_references": [refs["resource_summary"], refs["tier_a_rust"], refs["selected_intermediates_rust"]], "filesystem_cache_assumption": "uncontrolled; no cold-cache claim"},
        "future_comparison_record_schema": {"source": refs["comparison_schema"], "required_fields": ["baseline_id", "variant_id", "correctness_classification", "generated_token_agreement", "router_id_agreement", "maximum_error_by_checkpoint", "memory_delta", "logical_read_delta", "cache_hit_delta", "prefill_throughput_delta", "decode_throughput_delta", "artifact_size_delta", "dependency_platform_delta", "invariant_pass_fail"]},
        "success_gates": {"throughput_phase_gates_tok_per_second": {"M5-A": 0.05, "M5-B": 0.2, "M5-C": 0.5, "M5-D": 1.0, "stretch": 2.0}, "memory_hierarchy": ["frozen F32 correctness preserved", "deterministic fixture unchanged", "cache hits greater than zero where reuse is predicted", "material logical-read reduction", "strict configured memory budget", "no artifact duplication", "bounded startup and runtime memory"], "speed_alone_is_insufficient": True},
        "references": sorted(refs.values(), key=lambda item: item["role"]),
    }


def canonical_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/qwen3-30b-a3b/m4.4-performance-baseline-v1.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    document = build(root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
