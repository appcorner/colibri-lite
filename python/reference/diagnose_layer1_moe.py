#!/usr/bin/env python3
"""Diagnose the first Layer-1 MoE budget failure with frozen Rust inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.reference.export_full_model_router_reference import (
    RouterReferenceError,
    atomic_bytes,
    canonical_json,
    read_bf16,
    require,
)
from python.reference.export_layer1_router_reference import (
    execute_one_expert,
    parse_expert_source_plan,
    selected_occurrences,
)
from python.reference.validate_full_model_tensor_values import sha256_file


def read_f32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    payload = path.read_bytes()
    expected = 4
    for dimension in shape:
        expected *= dimension
    require(len(payload) == expected, f"unexpected F32 byte length for {path.name}")
    return torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(shape).clone()


def diagnostic_plan(path: Path) -> list[dict[str, int]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        token, position, expert, offset, length = (int(value) for value in line.split("\t"))
        records.append(
            {
                "token": token,
                "position": position,
                "expert": expert,
                "offset": offset,
                "length": length,
            }
        )
    require(len(records) == 32, "Layer-1 diagnostic occurrence count")
    require(
        [(record["token"], record["position"]) for record in records]
        == [(token, position) for token in range(4) for position in range(8)],
        "Layer-1 diagnostic occurrence order",
    )
    return records


def comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    require(left.shape == right.shape, "diagnostic comparison shape mismatch")
    difference = (left - right).abs()
    flat_index = int(difference.flatten().argmax())
    width = left.shape[-1]
    maximum = float(difference.flatten()[flat_index])
    relative = difference / right.abs().clamp_min(torch.finfo(torch.float32).tiny)
    violations = difference > (1.0e-6 + 1.0e-5 * right.abs())
    violation_indices = torch.where(violations.flatten())[0]
    first = None
    if violation_indices.numel():
        index = int(violation_indices[0])
        first = {
            "absolute_error": float(difference.flatten()[index]),
            "actual": float(left.flatten()[index]),
            "element": index % width,
            "expected": float(right.flatten()[index]),
            "scalar_budget": float(1.0e-6 + 1.0e-5 * right.flatten()[index].abs()),
            "token": index // width,
        }
    return {
        "first_scalar_contract_failure": first,
        "maximum_absolute_difference": maximum,
        "maximum_location": {"token": flat_index // width, "element": flat_index % width},
        "maximum_relative_difference": float(relative.max()),
        "scalar_contract_failure_count": int(violations.sum()),
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    input_tensor = read_f32(args.diagnostic_root / "layer1-expert-input-f32.bin", (4, 2048))
    routing_weights = read_f32(args.diagnostic_root / "layer1-routing-weights-f32.bin", (4, 8))
    rust_moe = read_f32(args.diagnostic_root / "layer1-moe-output-f32.bin", (4, 2048))
    records = diagnostic_plan(args.diagnostic_root / "layer1-expert-output-plan-v1.tsv")
    output_payload = (args.diagnostic_root / "layer1-expert-outputs-f32.bin").read_bytes()
    require(len(output_payload) == 32 * 2048 * 4, "Layer-1 expert output payload length")
    rust_expert_outputs = {}
    selected_ids = torch.empty((4, 8), dtype=torch.int64)
    for record in records:
        selected_ids[record["token"], record["position"]] = record["expert"]
        start = record["offset"]
        end = start + record["length"]
        rust_expert_outputs[(record["token"], record["position"])] = torch.frombuffer(
            bytearray(output_payload[start:end]), dtype=torch.float32
        ).clone()

    shards, projections = parse_expert_source_plan(args.expert_source_plan)
    selected_experts = sorted(set(int(value) for value in selected_ids.flatten().tolist()))
    selected_shards = sorted(
        {
            projections[(1, expert, role)]["shard_id"]
            for expert in selected_experts
            for role in ("gate", "up", "down")
        }
    )
    verified_shards = []
    source_hash_bytes = 0
    for shard_id in selected_shards:
        shard = shards[shard_id]
        source = args.source_root / shard["path"]
        require(source.stat().st_size == shard["bytes"], f"source shard {shard_id} size mismatch")
        require(sha256_file(source) == shard["sha256"], f"source shard {shard_id} hash mismatch")
        verified_shards.append({"shard_id": shard_id, **shard})
        source_hash_bytes += shard["bytes"]

    same_input_moe = torch.zeros_like(input_tensor)
    occurrence_comparisons = []
    source_payload_bytes = 0
    for expert in selected_experts:
        weights = {}
        for role in ("gate", "up", "down"):
            record = projections[(1, expert, role)]
            shard = shards[record["shard_id"]]
            weights[role] = read_bf16(
                args.source_root / shard["path"],
                record["offset"],
                record["length"],
                record["shape"],
            )
            source_payload_bytes += record["length"]
        positions, tokens = selected_occurrences(selected_ids, expert)
        with torch.inference_mode():
            expert_output, weighted = execute_one_expert(
                input_tensor,
                routing_weights,
                positions,
                tokens,
                weights["gate"],
                weights["up"],
                weights["down"],
                torch.float32,
            )
            same_input_moe.index_add_(0, tokens, weighted)
        for row, (position, token) in enumerate(zip(positions.tolist(), tokens.tolist(), strict=True)):
            metrics = comparison(expert_output[row], rust_expert_outputs[(token, position)])
            occurrence_comparisons.append(
                {
                    "expert": expert,
                    "position": position,
                    "token": token,
                    **metrics,
                }
            )

    reference = load_file(args.f32_checkpoints)
    require(
        torch.equal(reference["layer1_selected_expert_ids"], selected_ids),
        "Layer-1 Rust and reference expert IDs differ",
    )
    original_input = reference["layer1_post_attention_rmsnorm"]
    original_routing = reference["layer1_routing_weights"]
    original_moe = reference["layer1_moe_output"]
    original_comparison = comparison(rust_moe, original_moe)
    same_input_comparison = comparison(rust_moe, same_input_moe)
    original_failure = original_comparison["first_scalar_contract_failure"]
    same_failure_at_original_location = None
    if original_failure is not None:
        token = original_failure["token"]
        element = original_failure["element"]
        difference = abs(float(rust_moe[token, element] - same_input_moe[token, element]))
        same_failure_at_original_location = {
            "absolute_error": difference,
            "actual": float(rust_moe[token, element]),
            "element": element,
            "expected": float(same_input_moe[token, element]),
            "scalar_budget": float(1.0e-6 + 1.0e-5 * same_input_moe[token, element].abs()),
            "token": token,
        }
    cause = (
        "local_cross_runtime_expert_arithmetic_variance"
        if same_input_comparison["scalar_contract_failure_count"] > 0
        else "accumulated_incoming_drift"
    )
    evidence = {
        "schema_version": 1,
        "status": "layer1_moe_isolated_diagnostic_complete",
        "classification": cause,
        "input_f32_vs_reference": comparison(input_tensor, original_input),
        "routing_f32_vs_reference": comparison(routing_weights, original_routing),
        "original_path_rust_vs_transformers_f32": original_comparison,
        "same_rust_input_rust_vs_transformers_f32": same_input_comparison,
        "same_input_error_at_original_first_failure": same_failure_at_original_location,
        "per_occurrence_same_input_comparisons": occurrence_comparisons,
        "selected_expert_ids": selected_ids.tolist(),
        "selected_unique_experts": selected_experts,
        "selected_occurrences": 32,
        "source_shards_verified": verified_shards,
        "source_hash_bytes_read": source_hash_bytes,
        "source_payload_bytes_read": source_payload_bytes,
        "tolerance_changed": False,
        "runtime_arithmetic_changed": False,
    }
    payload = canonical_json(evidence)
    atomic_bytes(args.output, payload)
    return {
        "classification": cause,
        "evidence_sha256": hashlib.sha256(payload).hexdigest(),
        "original_path": original_comparison,
        "same_input": same_input_comparison,
        "status": "passed",
    }


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expert-source-plan", type=Path, required=True)
    parser.add_argument("--f32-checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    for name, value in vars(args).items():
        setattr(args, name, value.resolve())
    try:
        result = export(args)
    except (RouterReferenceError, OSError, KeyError, ValueError) as error:
        print(f"Layer-1 MoE diagnostic error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
