#!/usr/bin/env python3
"""Build the deterministic M4.3-01 frozen F32 baseline bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.reference.export_full_model_router_reference import atomic_bytes, canonical_json


MODEL_DIR = Path("models/qwen3-30b-a3b")
OUTPUT_NAMES = {
    "rust_evidence": "m4.3-01-rust-tier-b-evidence-v1.tsv",
    "fixtures": "m4.3-01-fixtures-v1.json",
    "f64": "m4.3-01-f64-diagnostics-v1.json",
    "comparison": "m4.3-01-comparison-schema-v1.json",
    "manifest": "m4.3-01-f32-baseline-manifest-v1.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fixture_budget(observed: float, guard: float) -> float:
    return f32(f32(3.0 * f32(observed)) + f32(guard))


def rust_scientific(value: float) -> str:
    return f"{value:.17e}".replace("e-0", "e-").replace("e+0", "e+")


def read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="ascii", newline="") as source:
        records = list(csv.reader(source, delimiter="\t"))
    return records[0], records[1:]


def final_rust_evidence(characterization_path: Path) -> bytes:
    header, records = read_tsv(characterization_path)
    expected_header = [
        "fixture",
        "token_ids",
        "final_norm_reference_sha256",
        "final_norm_rust_sha256",
        "maximum_fixed_final_norm_error",
        "final_norm_characterization_ceiling",
        "maximum_fixed_logit_error",
        "maximum_top20_logit_error",
        "logit_characterization_ceiling",
        "argmax",
        "top1_margin",
        "required_safe_margin",
        "classification",
        "top20_ids",
        "guard_layer0_ids",
        "guard_layer24_ids",
        "guard_layer47_ids",
        "kv_cache_bytes",
        "dense_bytes_read",
        "expert_bytes_read",
        "expert_loads",
        "expert_evictions",
    ]
    if header != expected_header:
        raise ValueError("unexpected Tier B characterization schema")
    header[5] = "final_norm_fixture_budget"
    header[8] = "logit_fixture_budget"
    for record in records:
        if len(record) != len(header):
            raise ValueError(f"invalid Tier B record {record[0]}")
        norm_observed = float(record[4])
        logit_observed = max(float(record[6]), float(record[7]))
        record[5] = rust_scientific(fixture_budget(norm_observed, 5.0e-7))
        record[8] = rust_scientific(fixture_budget(logit_observed, 2.0e-6))
    lines = ["\t".join(header), *("\t".join(record) for record in records)]
    return ("\n".join(lines) + "\n").encode("ascii")


def fixture_document(root: Path, rust_evidence: bytes) -> dict[str, Any]:
    reference = json.loads(
        (root / MODEL_DIR / "m4.3-01-tier-b-transformers-f32-v1.json").read_text(
            encoding="utf-8"
        )
    )
    rust_lines = list(csv.DictReader(rust_evidence.decode("ascii").splitlines(), delimiter="\t"))
    rust_by_name = {record["fixture"]: record for record in rust_lines}
    tier_b = []
    for fixture in reference["fixtures"]:
        rust = rust_by_name[fixture["name"]]
        tier_b.append(
            {
                "name": fixture["name"],
                "text": fixture["text"],
                "token_ids": fixture["token_ids"],
                "coverage": fixture["coverage"],
                "execution": "complete_f32_forward_with_genuine_kv_cache_no_generation",
                "final_position": fixture["final_position"],
                "expected_argmax": fixture["logits"]["argmax_token_id"],
                "reference_final_norm_sha256": fixture["final_norm"]["sha256_f32_le"],
                "rust_final_norm_sha256": rust["final_norm_rust_sha256"],
                "guard_router_ids": fixture["guard_router_ids"],
                "top20_token_ids": fixture["logits"]["top20_token_ids"],
                "top1_margin": fixture["logits"]["top1_margin"],
                "classification": rust["classification"],
                "kv_cache_bytes": int(rust["kv_cache_bytes"]),
            }
        )
    return {
        "schema_version": 1,
        "task": "M4.3-01",
        "status": "frozen",
        "tiers": {
            "A": {
                "purpose": "authoritative_end_to_end_generation_regression",
                "input_token_ids": [9707, 11, 1879, 0],
                "generated_token_ids": [1096, 374],
                "processed_positions": 6,
                "expected_cost": "394_to_438_process_seconds_on_recorded_host",
                "evidence": "models/qwen3-30b-a3b/m4.2-04-rust-short-generation-evidence-v1.tsv",
            },
            "B": {
                "purpose": "representative_complete_forward_coverage",
                "fixture_count": len(tier_b),
                "processed_positions": sum(len(item["token_ids"]) for item in tier_b),
                "fixtures": tier_b,
                "reference_evidence": "models/qwen3-30b-a3b/m4.3-01-tier-b-transformers-f32-v1.json",
                "rust_evidence": "models/qwen3-30b-a3b/m4.3-01-rust-tier-b-evidence-v1.tsv",
            },
            "C": {
                "purpose": "fast_focused_guard_regressions_without_unnecessary_full_model_execution",
                "fixtures": [
                    {
                        "operation": "embedding",
                        "evidence": "models/qwen3-30b-a3b/m4.2-02-transformers-f32-layer0-evidence-v1.json",
                    },
                    {
                        "operation": "rmsnorm",
                        "evidence": "models/qwen3-30b-a3b/m4.2-02-rms-scalar-diagnostics-v1.json",
                    },
                    {
                        "operation": "attention",
                        "evidence": "models/qwen3-30b-a3b/m4.2-02-rust-layer1-checkpoint-evidence-v1.tsv",
                    },
                    {
                        "operation": "router",
                        "evidence": "models/qwen3-30b-a3b/m4.2-02-rust-layer47-router-evidence-v1.tsv",
                    },
                    {
                        "operation": "selected_expert_mlp",
                        "evidence": "models/qwen3-30b-a3b/m4.2-03-rust-intermediate-evidence-v1.tsv",
                    },
                    {
                        "operation": "final_norm_and_lm_head",
                        "evidence": "models/qwen3-30b-a3b/m4.2-04-rust-short-generation-evidence-v1.tsv",
                    },
                    {
                        "operation": "kv_cache_update",
                        "evidence": "models/qwen3-30b-a3b/m4.2-05-baseline-summary-v1.json",
                    },
                ],
            },
        },
        "coverage": {
            "low_token_id": 0,
            "high_token_ids": [125451, 151643],
            "languages": ["English", "Thai"],
            "structures": ["one_token", "newline", "code_like", "repeated", "special_token"],
            "expert_ids": "low_and_high_ids_in_guard_and_selected_intermediate_evidence",
            "margins": "F32_safe_and_independent_BF16_ambiguous_cases_referenced_from_M4.2",
            "decode_modes": ["prefill", "cached_decode"],
        },
    }


def diagnostic_record(operation: str, index: str, f64_value: float, tf32: float, rust: float) -> dict[str, Any]:
    tf32_difference = abs(tf32 - f64_value)
    rust_difference = abs(rust - f64_value)
    return {
        "operation": operation,
        "selected_index": index,
        "f64": f64_value,
        "transformers_f32": tf32,
        "rust_f32": rust,
        "transformers_f32_absolute_difference": tf32_difference,
        "rust_f32_absolute_difference": rust_difference,
        "closer_f32_path": "transformers_f32" if tf32_difference < rust_difference else "rust_f32" if rust_difference < tf32_difference else "equal",
        "status": "diagnostic_only",
        "contract_impact": "none",
    }


def f64_document(root: Path) -> dict[str, Any]:
    lm_diagnostic = json.loads(
        (root / MODEL_DIR / "m4.3-01-single-low-token-lm-head-diagnostic-v1.json").read_text(
            encoding="utf-8"
        )
    )
    records = [
        diagnostic_record(
            "rmsnorm_weighted_output",
            "token=0,element=952",
            1.3951061001698273,
            1.3951061964035034,
            1.3951060771942139,
        ),
        diagnostic_record(
            "rmsnorm_weighted_output",
            "token=3,element=830",
            -1.7088856424047476,
            -1.7088855504989624,
            -1.708885908126831,
        ),
        diagnostic_record(
            "rmsnorm_sum_of_squares",
            "token=0",
            1.748016123184543,
            1.748016119003296,
            1.7480164766311646,
        ),
    ]
    for item in lm_diagnostic["f64_diagnostics"]:
        records.append(
            {
                "operation": item["operation"],
                "selected_index": f"vocabulary={item['vocabulary_index']}",
                "f64": item["f64"],
                "transformers_f32": item["transformers_f32"],
                "rust_f32": item["rust_f32"],
                "transformers_f32_absolute_difference": item[
                    "transformers_f32_absolute_difference"
                ],
                "rust_f32_absolute_difference": item["rust_f32_absolute_difference"],
                "closer_f32_path": item["closer_f32_path"],
                "status": "diagnostic_only",
                "contract_impact": "none",
            }
        )
    return {
        "schema_version": 1,
        "task": "M4.3-01",
        "status": "diagnostic_only_no_contract_change",
        "precision_hierarchy": [
            "selected_F64_diagnostics",
            "Transformers_F32_from_pinned_BF16_weights",
            "Rust_ordered_F32_runtime",
            "future_lower_precision_or_optimized_variants",
        ],
        "complete_model_f64_executed": False,
        "records": records,
        "supporting_evidence": [
            "models/qwen3-30b-a3b/m4.2-02-rms-scalar-diagnostics-v1.json",
            "models/qwen3-30b-a3b/m4.3-01-single-low-token-lm-head-diagnostic-v1.json",
        ],
        "decision": "F64 proximity does not change Rust arithmetic or existing contracts",
    }


def comparison_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "M4.3-01",
        "status": "frozen_future_variant_comparison_schema",
        "classifications": {
            "exact-equivalent": "all applicable exact evidence and numerical values match the frozen baseline",
            "numerically equivalent within contract": "exact structural invariants pass and every numerical delta stays within its scoped contract",
            "semantically equivalent": "required IDs and outputs agree but at least one numerical value is outside the F32-equivalent contract without quality-risk evidence",
            "quality-risk": "semantic output is currently retained but margins, errors, finite counts, or coverage indicate material risk",
            "correctness failure": "a mandatory invariant, safe discrete selection, finite-output rule, or scoped numerical contract fails",
        },
        "required_measurements": [
            "correctness_status",
            "generated_token_agreement",
            "router_id_agreement",
            "top_k_rank_agreement",
            "top_1_safe_margin_status",
            "maximum_error_by_checkpoint",
            "nan_count",
            "positive_infinity_count",
            "negative_infinity_count",
            "peak_working_set_bytes",
            "modeled_explicit_memory_bytes",
            "artifact_size_bytes",
            "logical_bytes_read_per_token",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "expert_cache_hit_rate",
            "startup_seconds",
        ],
        "mandatory_deltas": [
            "correctness",
            "numerical_contract",
            "memory",
            "io",
            "latency_and_throughput",
            "artifact_size",
            "platform_and_dependency_requirements",
        ],
        "mandatory_invariants": [
            "canonical_model_identity_and_source_provenance",
            "deterministic_router_tie_policy",
            "frozen_fixture_generated_ids",
            "safe_F32_router_and_token_classifications",
            "selected_intermediate_structure_and_membership",
            "finite_guarded_outputs",
            "KV_cache_no_overwrite_and_bounded_allocation",
            "bounded_expert_residency",
            "evidence_schema_compatibility_or_explicit_versioning",
        ],
        "tradeable_metrics": [
            "working_set_within_declared_budget",
            "artifact_size",
            "logical_bytes_read",
            "startup_latency",
            "prefill_throughput",
            "decode_throughput",
            "cache_hit_rate",
        ],
        "acceptance_rule": "performance_gain_alone_is_never_sufficient",
    }


def evidence_reference(root: Path, relative: str) -> dict[str, Any]:
    path = root / Path(relative)
    if not path.is_file():
        raise FileNotFoundError(relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def manifest_document(root: Path, generated: dict[str, bytes]) -> dict[str, Any]:
    evidence_paths = [
        "models/qwen3-30b-a3b/model-manifest-v1.json",
        "models/qwen3-30b-a3b/source-manifest-v1.json",
        "models/qwen3-30b-a3b/m4.2-tolerance-contract-registry-v1.json",
        "models/qwen3-30b-a3b/m4.2-03-rust-intermediate-evidence-v1.tsv",
        "models/qwen3-30b-a3b/m4.2-04-transformers-f32-generation-evidence-v1.json",
        "models/qwen3-30b-a3b/m4.2-04-transformers-bf16-generation-evidence-v1.json",
        "models/qwen3-30b-a3b/m4.2-04-rust-short-generation-evidence-v1.tsv",
        "models/qwen3-30b-a3b/m4.2-05-baseline-summary-v1.json",
        "models/qwen3-30b-a3b/m4.3-01-tier-b-transformers-f32-v1.json",
        "models/qwen3-30b-a3b/m4.3-01-tier-b-transformers-f32-v1.tsv",
        "models/qwen3-30b-a3b/m4.3-01-rust-tier-b-characterization-v1.tsv",
        "models/qwen3-30b-a3b/m4.3-01-single-low-token-lm-head-diagnostic-v1.json",
    ]
    references = [evidence_reference(root, path) for path in evidence_paths]
    for key in ("rust_evidence", "fixtures", "f64", "comparison"):
        payload = generated[key]
        references.append(
            {
                "path": f"models/qwen3-30b-a3b/{OUTPUT_NAMES[key]}",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "task": "M4.3-01",
        "status": "authoritative_unquantized_f32_baseline_frozen",
        "model": {
            "model_id": "Qwen/Qwen3-30B-A3B",
            "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "architecture": "Qwen3MoeForCausalLM",
            "source_dtype": "BF16",
            "storage_and_compute_dtype": "F32",
            "canonical_root_manifest_sha256": "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2",
        },
        "runtime": {
            "name": "colibri-lite-rs",
            "m4_2_closure_commit": "7321c46ae62543f9aa8c6025130d9b142e31a455",
            "last_runtime_implementation_commit": "4c02b611ebe77e4a0daca3afc16dddcfeccaba74",
            "build_mode": "release",
            "target": "x86_64-pc-windows-msvc",
            "rust_version": "1.96.1",
            "arithmetic": "safe_scalar_ordered_F32",
            "expert_cache_budget_bytes": 18_874_368,
        },
        "precision_hierarchy": [
            "selected_F64_diagnostics",
            "Transformers_F32_from_pinned_BF16_weights",
            "Rust_ordered_F32_runtime",
            "future_lower_precision_or_optimized_variants_not_implemented",
        ],
        "fixtures": "models/qwen3-30b-a3b/m4.3-01-fixtures-v1.json",
        "frozen_end_to_end": {
            "input_token_ids": [9707, 11, 1879, 0],
            "generated_token_ids": [1096, 374],
            "f32_classifications": ["exact_match_safe", "exact_match_safe"],
            "guard_router_ids": "referenced Tier A and Tier B evidence",
            "selected_intermediates": "referenced M4.2-03 evidence",
            "final_norm_lm_head_top_k": "referenced Tier A and Tier B evidence",
        },
        "kv_cache": {
            "layers": 48,
            "tier_a_capacity_and_final_length": 6,
            "key_and_value_shape_per_layer": [6, 4, 128],
            "allocated_bytes": 1_179_648,
            "allocation_growth": False,
            "previous_position_overwrite": False,
        },
        "numerical_contract_registry": "models/qwen3-30b-a3b/m4.2-tolerance-contract-registry-v1.json",
        "performance_reference": {
            "source": "models/qwen3-30b-a3b/m4.2-05-baseline-summary-v1.json",
            "host": "Windows 11 Enterprise 10.0.26200; Intel Core i7-1165G7; 8 logical processors; 51,249,209,344 RAM bytes",
            "filesystem_cache_assumption": "uncontrolled; run C potentially warm; no cold-cache claim",
            "logical_not_physical_io": True,
            "initialization_seconds": [2.459, 2.463, 2.364],
            "prefill_seconds": [286.068, 272.686, 254.771],
            "decode_seconds_by_token": [[70.030, 78.466], [62.406, 61.952], [65.269, 71.416]],
            "process_peak_working_set_bytes_range": [145_424_384, 148_066_304],
            "process_peak_private_bytes_range": [143_036_416, 143_269_888],
            "modeled_explicit_memory_bytes": 127_823_000,
            "logical_dense_bytes": 29_518_290_944,
            "logical_expert_bytes": 43_486_543_872,
            "logical_total_bytes": 73_004_834_816,
            "logical_bytes_per_cached_decode_token": 12_167_471_104,
            "total_read_amplification": 2.428603196361342,
            "expert_cache_hit_rate": 0.0,
            "kv_cache_bytes": 1_179_648,
        },
        "future_comparison_schema": "models/qwen3-30b-a3b/m4.3-01-comparison-schema-v1.json",
        "optimization_gate": "every future variant reports correctness, numerical, memory, I/O, performance, artifact-size, and platform/dependency deltas; speed alone cannot pass",
        "supporting_evidence": sorted(references, key=lambda item: item["path"]),
        "documentation": [
            "docs/reports/m4.3-01-f32-baseline.md",
            "docs/adr/0028-f32-baseline-bundle-and-fixtures.md",
            "docs/m4.3-f32-baseline-invariants.md",
            "docs/m4.2-optimization-invariants.md",
        ],
    }


def build_bundle(root: Path, output_root: Path) -> dict[str, str]:
    characterization = root / MODEL_DIR / "m4.3-01-rust-tier-b-characterization-v1.tsv"
    rust_evidence = final_rust_evidence(characterization)
    fixtures = canonical_json(fixture_document(root, rust_evidence))
    f64_diagnostics = canonical_json(f64_document(root))
    comparison = canonical_json(comparison_document())
    generated = {
        "rust_evidence": rust_evidence,
        "fixtures": fixtures,
        "f64": f64_diagnostics,
        "comparison": comparison,
    }
    manifest = canonical_json(manifest_document(root, generated))
    generated["manifest"] = manifest
    output_root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for key, payload in generated.items():
        output = output_root / OUTPUT_NAMES[key]
        atomic_bytes(output, payload)
        hashes[OUTPUT_NAMES[key]] = hashlib.sha256(payload).hexdigest()
    return hashes


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(arguments)
    root = args.repository_root.resolve()
    output_root = (args.output_root or root / MODEL_DIR).resolve()
    hashes = build_bundle(root, output_root)
    print(json.dumps({"status": "passed", "outputs": hashes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
