#!/usr/bin/env python3
"""Build and validate the deterministic M4 release-provenance record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "colibri-qwen3-moe-m4-release-provenance-v1"
RELEASE_ID = "colibri-lite-rs-m4-qwen3-30b-a3b-f32-v1"
RELEASE_TAG = "m4-full-qwen3-baseline-v1"
RUNTIME_COMMIT = "a230074959fc3b55ff73e8f4eb24e377a0a6b79f"
PARENT_COMMIT = "80099f05246a4450ded6f42baf6b8db5a4b2e623"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
BASELINE_ID = "qwen3-30b-a3b-colibri-f32-windows-x64-v1"
BASELINE_SCHEMA = "colibri-qwen3-moe-m4.4-performance-baseline-v1"
BASELINE_SHA256 = "29b2d95fa9eb74c1085cb31d2f63adbaa711fe8739d3051fa04f7b2b1c27ce9d"

PATHS = {
    "baseline": "models/qwen3-30b-a3b/m4.4-performance-baseline-v1.json",
    "f32_baseline_manifest": "models/qwen3-30b-a3b/m4.3-01-f32-baseline-manifest-v1.json",
    "model_manifest": "models/qwen3-30b-a3b/model-manifest-v1.json",
    "source_manifest": "models/qwen3-30b-a3b/source-manifest-v1.json",
    "canonical_root_registry": "models/qwen3-30b-a3b/canonical-root-registry-v1.json",
    "tolerance_registry": "models/qwen3-30b-a3b/m4.2-tolerance-contract-registry-v1.json",
    "f32_invariants": "docs/m4.3-f32-baseline-invariants.md",
    "baseline_report": "docs/reports/m4.4-01-performance-baseline.md",
    "resource_report": "docs/reports/m4.2-05-resource-baseline.md",
    "closure_report": "docs/reports/m4.2-correctness-and-variance-closure.md",
    "m43_closure": "docs/reports/m4.3-phase-closure.md",
    "candidate_registry": "models/qwen3-30b-a3b/m4.3-06-candidate-status-registry-v1.json",
    "decision_summary": "models/qwen3-30b-a3b/m4.3-06-decision-summary-v1.json",
    "decision_report": "docs/reports/m4.3-06-candidate-decision.md",
    "ik_environment": "models/qwen3-30b-a3b/m4.3-05-ik-llama-environment-v1.json",
    "ik_metrics": "models/qwen3-30b-a3b/m4.3-05-ik-llama-run-metrics-v1.json",
    "ik_report": "docs/reports/m4.3-05-ik-llama-comparison.md",
    "adr_0030": "docs/adr/0030-sensitive-dense-precision-policy.md",
    "adr_0031": "docs/adr/0031-expert-int8-degradation-decision.md",
    "adr_0032": "docs/adr/0032-m4.3-05-ik-llama-comparison.md",
    "adr_0033": "docs/adr/0033-m4.3-06-candidate-rejection-and-memory-pivot.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return value


def reference(root: Path, role: str, path_text: str) -> dict[str, Any]:
    path = root / path_text
    if not path.is_file():
        raise ValueError(f"missing provenance reference {role}: {path_text}")
    return {"role": role, "path": path_text, "bytes": path.stat().st_size, "sha256": sha256(path)}


def validate_no_started_m5(root: Path) -> None:
    tasks = (root / "docs" / "tasks.md").read_text(encoding="utf-8")
    for line in tasks.splitlines():
        # M5.1-00 through M5.1-03 and M5.2-01/M5.2-02 are post-release
        # evidence tasks.
        # Keep rejecting unrelated M5 implementation work from this historical
        # M4 validator while allowing the approved evidence chain to advance.
        if re.search(r"- \[(?:~|x)\].*M5", line) and not re.search(r"M5\.1-0[0-3]|M5\.2-0[12]", line):
            raise ValueError("an M5 implementation task is marked started or complete")


def validate_unique_release_id(root: Path) -> None:
    matches = []
    for path in (root / "models" / "qwen3-30b-a3b").glob("m4-release-provenance-v1.json"):
        try:
            document = read_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if document.get("release_id") == RELEASE_ID:
            matches.append(path)
    if len(matches) > 1:
        names = ", ".join(str(path.relative_to(root)) for path in matches)
        raise ValueError(f"release ID {RELEASE_ID!r} is duplicated: {names}")


def validate_tag(root: Path, expected_commit: str | None) -> None:
    result = subprocess.run(
        ["git", "rev-parse", f"{RELEASE_TAG}^{{commit}}"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"release tag is missing: {RELEASE_TAG}")
    target = result.stdout.strip()
    if expected_commit is not None and target != expected_commit:
        raise ValueError(f"release tag points to {target}, expected {expected_commit}")


def build(root: Path) -> dict[str, Any]:
    refs = {role: reference(root, role, path) for role, path in PATHS.items()}
    baseline = read_json(root / PATHS["baseline"])
    f32_baseline = read_json(root / PATHS["f32_baseline_manifest"])
    model = read_json(root / PATHS["model_manifest"])
    source = read_json(root / PATHS["source_manifest"])
    canonical = read_json(root / PATHS["canonical_root_registry"])
    candidate = read_json(root / PATHS["candidate_registry"])
    decision = read_json(root / PATHS["decision_summary"])
    ik_env = read_json(root / PATHS["ik_environment"])
    ik_metrics = read_json(root / PATHS["ik_metrics"])

    if baseline.get("baseline_id") != BASELINE_ID or baseline.get("schema") != BASELINE_SCHEMA:
        raise ValueError("M4.4 baseline identity changed")
    if f32_baseline.get("status") != "authoritative_unquantized_f32_baseline_frozen":
        raise ValueError("M4.3 F32 baseline is not authoritative")
    if refs["baseline"]["sha256"] != BASELINE_SHA256:
        raise ValueError("M4.4 baseline hash changed")
    if model.get("revision") != MODEL_REVISION or model.get("model_id") != "Qwen/Qwen3-30B-A3B":
        raise ValueError("pinned model identity changed")
    if model["source_contract"]["sha256"] != refs["source_manifest"]["sha256"]:
        raise ValueError("source-provenance hash does not match source manifest")
    if model["inventory"]["component_bytes"] != 122147666917:
        raise ValueError("canonical artifact byte count changed")
    if canonical["root_manifest_sha256"] != "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2":
        raise ValueError("canonical root hash changed")
    statuses = {item["item"]: item["status"] for item in candidate["phase_verdict"]}
    if statuses.get("f32_baseline") != "authoritative_and_accepted":
        raise ValueError("F32 baseline is not accepted")
    if statuses.get("int8_per_output_channel") != "rejected_full_model_candidate":
        raise ValueError("INT8 candidate status changed")
    if any(item["status"] in {"accepted_for_runtime_prototype", "authoritative_and_accepted"} for item in candidate["phase_verdict"] if item["item"] != "f32_baseline"):
        raise ValueError("a rejected/provisional candidate is marked accepted")
    if decision["investment_decision"]["immediate_production_int8"] is not False:
        raise ValueError("investment decision changed")
    if ik_env["model"]["quantization"] != "Q4_K_M" or ik_env["repository"]["commit"] != "1fddd12ba861c4815a8633f14d9c5670692099cc":
        raise ValueError("external reference identity changed")
    validate_no_started_m5(root)
    validate_unique_release_id(root)

    source_files = {item["path"]: item for item in source["files"]}
    index = source_files.get(source["safetensors"]["index_file"])
    if index is None:
        raise ValueError("source tensor index is not recorded")
    tokenizer_files = model["components"]["tokenizer"]["files"]
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "release_tag": RELEASE_TAG,
        "creation": {
            "tool": "python/reference/build_m4_release_provenance.py",
            "tool_version": "m4.4-02-release-provenance-generator-v1",
            "serialization": "UTF-8 JSON, sorted keys, compact separators, trailing newline, no timestamp",
            "deterministic_payload": True,
        },
        "runtime_source": {
            "repository": "colibri-lite-rs",
            "branch": "milestone/m4-full-qwen3",
            "runtime_commit": RUNTIME_COMMIT,
            "parent_baseline_commit": PARENT_COMMIT,
            "rust_toolchain": "1.96.1",
            "target": "x86_64-pc-windows-msvc",
            "build_profile": "release",
            "feature_flags": [],
            "configuration": baseline["runtime_identity"],
            "machine_observation": {
                "operating_system": baseline["runtime_identity"]["operating_system"],
                "threads": baseline["runtime_identity"]["threads"],
                "filesystem_cache_assumption": baseline["performance_baseline"]["filesystem_cache_assumption"],
            },
        },
        "model_source": {
            "repository": source["model"]["repository_url"],
            "revision": MODEL_REVISION,
            "immutable_tree_url": source["model"]["immutable_tree_url"],
            "architecture": source["model"]["architecture"],
            "model_type": source["model"]["model_type"],
            "license": source["model"]["license"],
            "license_evidence": {"source_manifest": refs["source_manifest"], "license_file": source["model"]["license_file"], "license_url": source["model"]["license_url"]},
            "source_tensor_index": {"path": source["safetensors"]["index_file"], "sha256": index["sha256"], "bytes": index["bytes"]},
            "source_shard_count": source["safetensors"]["shard_count"],
            "source_snapshot_bytes": source["download_plan"]["source_snapshot_bytes"],
            "source_manifest": refs["source_manifest"],
            "tokenizer": {"class": model["components"]["tokenizer"]["tokenizer_class"], "files": tokenizer_files, "manifest": {"path": model["components"]["tokenizer"]["manifest"]["path"], "sha256": model["components"]["tokenizer"]["manifest"]["sha256"], "bytes": model["components"]["tokenizer"]["manifest"]["bytes"]}},
        },
        "canonical_artifact": {
            "format": model["artifact_format"],
            "version": model["format_version"],
            "root_manifest_sha256": canonical["root_manifest_sha256"],
            "root_manifest": refs["canonical_root_registry"],
            "payload_file_count": model["inventory"]["required_file_count"],
            "root_registry_file_count": canonical["canonical_file_count"],
            "total_bytes": model["inventory"]["component_bytes"],
            "dense": model["components"]["dense"],
            "experts": {key: value for key, value in model["components"]["experts"].items() if key != "shards"},
            "tokenizer_manifest_sha256": model["components"]["tokenizer"]["manifest"]["sha256"],
            "source_provenance_sha256": refs["source_manifest"]["sha256"],
            "dtype": {"source": model["dtypes"]["source"], "storage": model["dtypes"]["storage"], "compute": model["dtypes"]["compute"]},
            "endianness": model["endianness"],
            "dimensions": model["dimensions"],
            "compatibility": model["runtime_compatibility"],
            "operational_root": canonical["canonical_artifact_root"],
        },
        "m4_baseline": {
            "baseline_id": BASELINE_ID,
            "baseline_schema": BASELINE_SCHEMA,
            "baseline_sha256": BASELINE_SHA256,
            "performance_baseline": refs["baseline"],
            "f32_manifest": refs["f32_baseline_manifest"],
            "tolerance_registry": refs["tolerance_registry"],
            "optimization_invariants": refs["f32_invariants"],
            "generated_token_ids": [1096, 374],
            "tier_a_b_c_evidence": {"tier_a": "referenced by M4.4 baseline", "tier_b": "referenced by M4.4 baseline", "tier_c": "referenced by M4.4 baseline"},
            "kv_cache": baseline["correctness_baseline"]["kv_cache_contract"],
            "resource_baseline": refs["resource_report"],
            "quantization_decisions": refs["candidate_registry"],
            "external_reference": refs["ik_report"],
            "supporting_references": sorted((item for role, item in refs.items() if role not in {"baseline", "model_manifest", "source_manifest", "canonical_root_registry"}), key=lambda item: item["role"]),
        },
        "external_reference": {
            "repository": "ik_llama.cpp",
            "commit": ik_env["repository"]["commit"],
            "build": ik_env["build"],
            "cpu_features": ik_env["build"]["cpu_instruction_targets"],
            "model": {"format": ik_env["model"]["quantization"], "file": ik_env["model"]["file"], "bytes": ik_env["model"]["file_bytes"], "sha256": ik_env["model"].get("file_sha256")},
            "benchmark": {"prompt_tokens_per_second": ik_metrics["llama_bench"]["prompt_4_tokens"], "long_decode_tokens_per_second": ik_metrics["long_decode"]["decode_tokens_per_second"], "short_decode_range": [min(item["decode_tokens_per_second"] for item in ik_metrics["short_runs"]), max(item["decode_tokens_per_second"] for item in ik_metrics["short_runs"])]},
            "classification": "directional_not_format_controlled",
            "quality_equivalence_to_f32": False,
            "evidence": [refs["ik_environment"], refs["ik_metrics"], refs["ik_report"]],
        },
        "m4_verdict": {
            "full_model_feasibility": "passed",
            "artifact_conversion_and_integrity": "passed",
            "rust_f32_correctness": "passed_with_documented_numerical_variance",
            "deterministic_generation": "passed",
            "low_memory_feasibility": "passed",
            "performance_readiness": "not_ready",
            "first_full_model_int8_candidate": "rejected",
            "hardware_capability": "validated_by_external_optimized_reference",
            "investment_decision": "continue_with_memory_hierarchy_and_performance_recovery_pivot",
            "next_phase": "F32 memory-hierarchy and performance recovery",
            "correctness_limitations": ["documented cross-runtime numerical variance", "full lower-precision runtime remains unvalidated"],
            "performance_limitations": ["scalar F32 disk-streaming path", "logical reads are not physical device I/O", "filesystem cache was uncontrolled"],
        },
        "references": sorted(refs.values(), key=lambda item: item["role"]),
        "m5_gate": {"next_task": "M5.1-01 Trace-driven memory hierarchy simulation", "started": False},
    }


def canonical_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/qwen3-30b-a3b/m4-release-provenance-v1.json"))
    parser.add_argument("--verify-tag", action="store_true")
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.verify_tag:
        validate_tag(root, args.expected_commit)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(build(root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
