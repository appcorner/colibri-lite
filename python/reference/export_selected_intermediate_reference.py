#!/usr/bin/env python3
"""Export compact M4.2-03 expert-intermediate reference checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import safetensors
from safetensors.torch import load_file
import torch
import torch.nn.functional as functional
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.reference.export_full_model_router_reference import (
    RouterReferenceError,
    atomic_bytes,
    atomic_safetensors,
    canonical_json,
    checkpoint_plan,
    process_peak_working_set,
    read_bf16,
    require,
)
from python.reference.export_layer1_router_reference import (
    parse_expert_source_plan,
    selected_occurrences,
)
from python.reference.validate_full_model_tensor_values import read_json, sha256_file


CASES = (
    {"layer": 0, "token": 0, "positions": (0, 7)},
    {"layer": 1, "token": 0, "positions": (0, 7)},
    {"layer": 24, "token": 1, "positions": (0, 7)},
    {"layer": 47, "token": 0, "positions": (0, 7)},
)
ROLES = ("gate", "up", "down")
SHAPES = {
    "gate": [768, 2048],
    "up": [768, 2048],
    "down": [2048, 768],
}
ORIENTATIONS = {
    "gate": "output_by_input",
    "up": "output_by_input",
    "down": "output_by_intermediate",
}


def tensor_name(layer: int, token: int, position: int, expert: int, checkpoint: str) -> str:
    return f"layer{layer}_token{token}_position{position}_expert{expert}_{checkpoint}"


def layer_name(layer: int, token: int, checkpoint: str) -> str:
    return f"layer{layer}_token{token}_{checkpoint}"


def execute_expert(
    expert_input: torch.Tensor,
    routing_weight: torch.Tensor,
    weights: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    current = expert_input.to(dtype)
    gate_projection = functional.linear(current, weights["gate"].to(dtype))
    up_projection = functional.linear(current, weights["up"].to(dtype))
    activated_gate = functional.silu(gate_projection)
    activated_product = activated_gate * up_projection
    down_projection = functional.linear(activated_product, weights["down"].to(dtype))
    weight = routing_weight.to(dtype)
    weighted_output = down_projection * weight.reshape(-1, 1)
    return {
        "expert_input": current.float().contiguous(),
        "gate_projection": gate_projection.float().contiguous(),
        "up_projection": up_projection.float().contiguous(),
        "activated_gate": activated_gate.float().contiguous(),
        "activated_product": activated_product.float().contiguous(),
        "down_projection": down_projection.float().contiguous(),
        "routing_weight": weight.float().contiguous(),
        "weighted_expert_output": weighted_output.float().contiguous(),
    }


def selected_runtime_plan(
    contract: dict[str, Any],
    registry: dict[str, Any],
    expert_manifest: dict[str, Any],
    selected: list[int],
) -> bytes:
    shards = {record["shard_id"]: record for record in expert_manifest["shards"]}
    experts = {
        (record["layer"], record["expert"]): record
        for record in expert_manifest["experts"]
    }
    shard = shards[47]
    lines = [
        "format_version\t1",
        f"model_id\t{contract['model_id']}",
        f"revision\t{contract['revision']}",
        f"root_manifest_sha256\t{registry['root_manifest_sha256']}",
        "artifact_component\texperts",
        f"shard\t47\t{shard['path']}\t{shard['byte_length']}\t{shard['sha256']}",
    ]
    for expert in selected:
        record = experts[(47, expert)]
        lines.append(
            f"expert\t47\t{expert}\tlayer.47.expert.{expert}\t{shard['path']}\t"
            f"{record['payload_offset']}\t{record['payload_length']}\t{record['sha256']}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def export(args: argparse.Namespace) -> dict[str, Any]:
    contract = read_json(args.contract)
    registry = read_json(args.registry)
    source_manifest = read_json(args.source_manifest)
    require(contract["schema_version"] == 1, "unsupported M4.2 contract version")
    require(contract["model_id"] == source_manifest["model"]["id"], "model ID mismatch")
    require(contract["revision"] == source_manifest["model"]["revision"], "revision mismatch")
    require(registry["model_id"] == contract["model_id"], "registry model ID mismatch")
    require(registry["revision"] == contract["revision"], "registry revision mismatch")
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }
    require(versions == contract["environment"], f"reference environment drift: {versions}")

    artifact_root = Path(registry["canonical_artifact_root"])
    root_manifest_path = artifact_root / "model-manifest-v1.json"
    require(sha256_file(root_manifest_path) == registry["root_manifest_sha256"], "root manifest hash mismatch")
    root_manifest = read_json(root_manifest_path)
    expert_manifest_path = artifact_root / root_manifest["components"]["experts"]["manifest"]["path"]
    require(
        sha256_file(expert_manifest_path) == root_manifest["components"]["experts"]["manifest"]["sha256"],
        "expert manifest hash mismatch",
    )
    expert_manifest = read_json(expert_manifest_path)
    artifact_experts = {
        (record["layer"], record["expert"]): record
        for record in expert_manifest["experts"]
    }
    source_shards, projections = parse_expert_source_plan(args.expert_source_plan)

    bf16_source = load_file(str(args.bf16_checkpoints), device="cpu")
    f32_source = load_file(str(args.f32_checkpoints), device="cpu")
    selected_by_layer: dict[int, list[int]] = {}
    case_selected_by_layer: dict[int, list[int]] = {}
    for case in CASES:
        layer = case["layer"]
        token = case["token"]
        bf16_ids = bf16_source[f"layer{layer}_selected_expert_ids"][token].tolist()
        f32_ids = f32_source[f"layer{layer}_selected_expert_ids"][token].tolist()
        require(bf16_ids == f32_ids, f"Layer-{layer} token-{token} BF16/F32 selected IDs differ")
        case_selected_by_layer[layer] = [int(value) for value in f32_ids]
        selected_by_layer[layer] = sorted(
            set(int(value) for value in bf16_source[f"layer{layer}_selected_expert_ids"].flatten().tolist())
            | set(int(value) for value in f32_source[f"layer{layer}_selected_expert_ids"].flatten().tolist())
        )

    required_shards = {
        projections[(layer, expert, role)]["shard_id"]
        for layer, selected in selected_by_layer.items()
        for expert in selected
        for role in ROLES
    }
    verified_shards = []
    source_hash_bytes = 0
    for shard_id in sorted(required_shards):
        record = source_shards[shard_id]
        path = args.source_root / record["path"]
        require(path.stat().st_size == record["bytes"], f"source shard {shard_id} size mismatch")
        require(sha256_file(path) == record["sha256"], f"source shard {shard_id} hash mismatch")
        verified_shards.append({"shard_id": shard_id, **record})
        source_hash_bytes += record["bytes"]

    torch.manual_seed(contract["determinism"]["seed"])
    torch.set_num_threads(contract["determinism"]["torch_threads"])
    torch.set_num_interop_threads(contract["determinism"]["torch_interop_threads"])
    torch.use_deterministic_algorithms(contract["determinism"]["torch_deterministic_algorithms"])
    bf16_output: dict[str, torch.Tensor] = {}
    f32_output: dict[str, torch.Tensor] = {}
    structure_lines = [
        "record\tlayer\ttoken\tposition\texpert\trole\tshape\torientation\tsource_name\t"
        "source_shard\tsource_offset\tsource_length\tartifact_path\tartifact_payload_offset\t"
        "artifact_projection_offset\tartifact_projection_length\taggregation_experts"
    ]
    case_records: list[dict[str, Any]] = []
    source_payload_bytes = 0

    with torch.inference_mode():
        for case in CASES:
            layer = case["layer"]
            token = case["token"]
            selected = case_selected_by_layer[layer]
            runs = {
                "bf16": {
                    "dtype": torch.bfloat16,
                    "source": bf16_source,
                    "output": bf16_output,
                    "ids": bf16_source[f"layer{layer}_selected_expert_ids"],
                },
                "f32": {
                    "dtype": torch.float32,
                    "source": f32_source,
                    "output": f32_output,
                    "ids": f32_source[f"layer{layer}_selected_expert_ids"],
                },
            }
            combined = {
                name: torch.zeros((4, 2048), dtype=run["dtype"])
                for name, run in runs.items()
            }
            detailed = {selected[position]: position for position in case["positions"]}
            loaded_weights: dict[int, dict[str, torch.Tensor]] = {}
            for expert in selected_by_layer[layer]:
                weights: dict[str, torch.Tensor] = {}
                for role in ROLES:
                    record = projections[(layer, expert, role)]
                    require(record["shape"] == SHAPES[role], f"Layer-{layer} expert-{expert} {role} shape mismatch")
                    weights[role] = read_bf16(
                        args.source_root / source_shards[record["shard_id"]]["path"],
                        record["offset"],
                        record["length"],
                        record["shape"],
                    )
                    source_payload_bytes += record["length"]
                loaded_weights[expert] = weights
                for name, run in runs.items():
                    if expert not in run["ids"]:
                        continue
                    source = run["source"]
                    positions, tokens = selected_occurrences(run["ids"], expert)
                    trace = execute_expert(
                        source[f"layer{layer}_post_attention_rmsnorm"][tokens],
                        source[f"layer{layer}_routing_weights"][tokens, positions],
                        weights,
                        run["dtype"],
                    )
                    combined[name].index_add_(
                        0,
                        tokens,
                        trace["weighted_expert_output"].to(run["dtype"]),
                    )
                    if expert in detailed:
                        position = detailed[expert]
                        occurrence = torch.where((tokens == token) & (positions == position))[0]
                        require(
                            occurrence.numel() == 1,
                            f"missing Layer-{layer} token-{token} position-{position} expert-{expert} occurrence",
                        )
                        occurrence_index = int(occurrence.item())
                        for checkpoint, tensor in trace.items():
                            key = tensor_name(layer, token, position, expert, checkpoint)
                            run["output"][key] = tensor[occurrence_index].reshape(-1).contiguous()

            aggregation = ",".join(str(value) for value in selected)
            for position in case["positions"]:
                expert = selected[position]
                artifact = artifact_experts[(layer, expert)]
                for role in ROLES:
                    source = projections[(layer, expert, role)]
                    artifact_projection = artifact[role]
                    require(artifact_projection["shape"] == SHAPES[role], "artifact projection shape mismatch")
                    structure_lines.append(
                        "\t".join(
                            [
                                "projection",
                                str(layer),
                                str(token),
                                str(position),
                                str(expert),
                                role,
                                ",".join(str(value) for value in SHAPES[role]),
                                ORIENTATIONS[role],
                                source["name"],
                                str(source["shard_id"]),
                                str(source["offset"]),
                                str(source["length"]),
                                f"experts/{expert_manifest['shards'][layer]['path']}",
                                str(artifact["payload_offset"]),
                                str(artifact_projection["offset"]),
                                str(artifact_projection["length"]),
                                aggregation,
                            ]
                        )
                    )
                case_records.append(
                    {
                        "layer": layer,
                        "token": token,
                        "position": position,
                        "expert": expert,
                        "routing_weight_index": token * 8 + position,
                    }
                )

            for name, run in runs.items():
                source = run["source"]
                dtype = run["dtype"]
                moe = combined[name][token].float().contiguous()
                residual = source[f"layer{layer}_residual_output"][token].to(dtype)
                residual_addition = (residual + combined[name][token]).float().contiguous()
                run["output"][layer_name(layer, token, "aggregated_moe_output")] = moe
                run["output"][layer_name(layer, token, "moe_residual_addition")] = residual_addition
                run["output"][layer_name(layer, token, "final_block_output")] = residual_addition.clone()
                if layer < 47:
                    expected_moe = source[f"layer{layer}_moe_output"][token]
                    expected_block = source[f"layer{layer}_block_output"][token]
                    require(torch.equal(moe, expected_moe), f"{name} Layer-{layer} aggregate drift")
                    require(torch.equal(residual_addition, expected_block), f"{name} Layer-{layer} block drift")
            del loaded_weights

    require(bf16_output.keys() == f32_output.keys(), "BF16/F32 output key mismatch")
    atomic_safetensors(args.bf16_output, bf16_output)
    atomic_safetensors(args.f32_output, f32_output)
    atomic_bytes(args.bf16_plan, checkpoint_plan(args.bf16_output))
    atomic_bytes(args.f32_plan, checkpoint_plan(args.f32_output))
    structure_payload = ("\n".join(structure_lines) + "\n").encode("utf-8")
    atomic_bytes(args.structure_output, structure_payload)
    layer47_selected = sorted(
        set(int(value) for value in f32_source["layer47_selected_expert_ids"].flatten().tolist())
    )
    runtime_payload = selected_runtime_plan(contract, registry, expert_manifest, layer47_selected)
    atomic_bytes(args.runtime_plan, runtime_payload)

    maximum_pairwise: dict[str, float] = {}
    for key in sorted(bf16_output):
        checkpoint = key.rsplit("_", maxsplit=1)[-1]
        if key.endswith("weighted_expert_output"):
            checkpoint = "weighted_expert_output"
        elif key.endswith("aggregated_moe_output"):
            checkpoint = "aggregated_moe_output"
        elif key.endswith("moe_residual_addition"):
            checkpoint = "moe_residual_addition"
        elif key.endswith("final_block_output"):
            checkpoint = "final_block_output"
        elif key.endswith("gate_projection"):
            checkpoint = "gate_projection"
        elif key.endswith("up_projection"):
            checkpoint = "up_projection"
        elif key.endswith("activated_gate"):
            checkpoint = "activated_gate"
        elif key.endswith("activated_product"):
            checkpoint = "activated_product"
        elif key.endswith("down_projection"):
            checkpoint = "down_projection"
        elif key.endswith("routing_weight"):
            checkpoint = "routing_weight"
        elif key.endswith("expert_input"):
            checkpoint = "expert_input"
        difference = float((bf16_output[key] - f32_output[key]).abs().max())
        maximum_pairwise[checkpoint] = max(maximum_pairwise.get(checkpoint, 0.0), difference)

    base = {
        "schema_version": 1,
        "model_id": contract["model_id"],
        "revision": contract["revision"],
        "status": "selected_intermediate_reference_passed",
        "selected_cases": case_records,
        "structural_contract": {
            "expert_input_shape": [2048],
            "intermediate_shape": [768],
            "expert_output_shape": [2048],
            "projection_order": ["gate", "up", "down"],
            "aggregation_order": "ascending_expert_id",
            "structure_sha256": hashlib.sha256(structure_payload).hexdigest(),
        },
        "source_shards_verified": verified_shards,
        "source_hash_bytes_read": source_hash_bytes,
        "source_payload_bytes_read": source_payload_bytes,
        "bf16_vs_f32_maximum_absolute_by_checkpoint": maximum_pairwise,
        "runtime_plan": {
            "selected_layer47_experts": layer47_selected,
            "bytes": len(runtime_payload),
            "sha256": hashlib.sha256(runtime_payload).hexdigest(),
        },
        "environment": versions,
    }
    outputs = {
        "bf16": (args.bf16_output, bf16_output, args.bf16_evidence, "BF16"),
        "f32": (args.f32_output, f32_output, args.f32_evidence, "F32"),
    }
    hashes: dict[str, str] = {}
    for name, (path, tensors, evidence_path, dtype_name) in outputs.items():
        document = {
            **base,
            "compute_dtype": dtype_name,
            "checkpoint_file": {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "tensor_count": len(tensors),
            },
        }
        payload = canonical_json(document)
        atomic_bytes(evidence_path, payload)
        hashes[f"{name}_checkpoints"] = document["checkpoint_file"]["sha256"]
        hashes[f"{name}_evidence"] = hashlib.sha256(payload).hexdigest()
    return {
        "status": "passed",
        "cases": len(case_records),
        "source_hash_bytes_read": source_hash_bytes,
        "source_payload_bytes_read": source_payload_bytes,
        "peak_process_working_set_bytes": process_peak_working_set(),
        **hashes,
    }


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source_root",
        "registry",
        "source_manifest",
        "expert_source_plan",
        "contract",
        "bf16_checkpoints",
        "f32_checkpoints",
        "bf16_output",
        "f32_output",
        "bf16_plan",
        "f32_plan",
        "structure_output",
        "runtime_plan",
        "bf16_evidence",
        "f32_evidence",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args(arguments)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    try:
        result = export(args)
    except (RouterReferenceError, OSError, KeyError, ValueError) as error:
        print(f"selected intermediate reference error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
