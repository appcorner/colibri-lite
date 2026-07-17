#!/usr/bin/env python3
"""Export bounded Layer-0 expert and Layer-1 pre-router reference checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import safetensors
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
    execute_pre_router,
    process_peak_working_set,
    read_bf16,
    reference_config,
    require,
    router_boundaries,
)
from python.reference.validate_full_model_tensor_values import parse_plan, read_json, sha256_file


def parse_expert_source_plan(path: Path) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, int, str], dict[str, Any]]]:
    shards: dict[int, dict[str, Any]] = {}
    projections: dict[tuple[int, int, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if fields[0] == "shard":
            shard_id = int(fields[1])
            shards[shard_id] = {
                "path": fields[2],
                "bytes": int(fields[3]),
                "sha256": fields[4],
            }
        elif fields[0] == "projection":
            layer, expert, role = int(fields[1]), int(fields[2]), fields[3]
            key = (layer, expert, role)
            require(key not in projections, f"duplicate expert projection {key}")
            projections[key] = {
                "name": fields[4],
                "shard_id": int(fields[5]),
                "offset": int(fields[6]),
                "length": int(fields[7]),
                "shape": [int(value) for value in fields[8].split(",")],
            }
    return shards, projections


def dense_names(layer: int) -> list[str]:
    prefix = f"model.layers.{layer}"
    return [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.self_attn.q_norm.weight",
        f"{prefix}.self_attn.k_norm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.mlp.gate.weight",
    ]


def selected_occurrences(expert_ids: torch.Tensor, expert: int) -> tuple[torch.Tensor, torch.Tensor]:
    expert_mask = functional.one_hot(expert_ids, num_classes=128).permute(2, 1, 0)
    positions, tokens = torch.where(expert_mask[expert])
    return positions, tokens


def execute_one_expert(
    hidden: torch.Tensor,
    routing_weights: torch.Tensor,
    positions: torch.Tensor,
    tokens: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = hidden[tokens].to(dtype)
    gate_up = torch.cat((gate, up), dim=0).to(dtype)
    gate_value, up_value = functional.linear(current, gate_up).chunk(2, dim=-1)
    activated = functional.silu(gate_value) * up_value
    expert_output = functional.linear(activated, down.to(dtype))
    weighted = expert_output * routing_weights[tokens, positions, None].to(dtype)
    return expert_output, weighted


def execute_selected_experts(
    source_path: Path,
    projections: dict[tuple[int, int, str], dict[str, Any]],
    bf16_run: dict[str, torch.Tensor],
    f32_run: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, list[int], float]:
    bf16_ids = bf16_run["selected_expert_ids"]
    f32_ids = f32_run["selected_expert_ids"]
    require(torch.equal(bf16_ids, f32_ids), "Layer-0 BF16 and F32 expert IDs differ")
    unique_experts = sorted(set(int(value) for value in f32_ids.flatten().tolist()))
    require(len(unique_experts) == 27, f"unexpected unique Layer-0 expert count: {len(unique_experts)}")

    bf16_hidden = bf16_run["post_attention_rmsnorm"].squeeze(0)
    f32_hidden = f32_run["post_attention_rmsnorm"].squeeze(0)
    bf16_combined = torch.zeros_like(bf16_hidden)
    f32_combined = torch.zeros_like(f32_hidden)
    bf16_outputs: dict[str, torch.Tensor] = {}
    f32_outputs: dict[str, torch.Tensor] = {}
    source_bytes = 0
    started = time.perf_counter()
    for expert in unique_experts:
        weights = {}
        for role, shape in (("gate", [768, 2048]), ("up", [768, 2048]), ("down", [2048, 768])):
            record = projections[(0, expert, role)]
            require(record["shape"] == shape, f"Layer-0 expert {expert} {role} shape mismatch")
            weights[role] = read_bf16(
                source_path,
                record["offset"],
                record["length"],
                record["shape"],
            )
            source_bytes += record["length"]
        positions, tokens = selected_occurrences(f32_ids, expert)
        require(positions.numel() > 0, f"selected expert {expert} has no occurrence")
        with torch.inference_mode():
            bf16_expert, bf16_weighted = execute_one_expert(
                bf16_hidden,
                bf16_run["routing_weights"],
                positions,
                tokens,
                weights["gate"],
                weights["up"],
                weights["down"],
                torch.bfloat16,
            )
            f32_expert, f32_weighted = execute_one_expert(
                f32_hidden,
                f32_run["routing_weights"],
                positions,
                tokens,
                weights["gate"],
                weights["up"],
                weights["down"],
                torch.float32,
            )
            bf16_combined.index_add_(0, tokens, bf16_weighted.to(bf16_combined.dtype))
            f32_combined.index_add_(0, tokens, f32_weighted.to(f32_combined.dtype))
        for row, (position, token) in enumerate(zip(positions.tolist(), tokens.tolist(), strict=True)):
            name = f"layer0_expert_output_t{token}_p{position}_e{expert}"
            bf16_outputs[name] = bf16_expert[row].float().contiguous()
            f32_outputs[name] = f32_expert[row].float().contiguous()
        del weights
    elapsed = time.perf_counter() - started
    bf16_outputs["layer0_moe_output"] = bf16_combined.float().contiguous()
    f32_outputs["layer0_moe_output"] = f32_combined.float().contiguous()
    bf16_outputs["layer0_block_output"] = (
        bf16_run["residual_output"].squeeze(0) + bf16_combined
    ).float().contiguous()
    f32_outputs["layer0_block_output"] = (
        f32_run["residual_output"].squeeze(0) + f32_combined
    ).float().contiguous()
    return bf16_outputs, f32_outputs, source_bytes, unique_experts, elapsed


def serialize_run(
    contract: dict[str, Any],
    layer0: dict[str, torch.Tensor],
    experts: dict[str, torch.Tensor],
    layer1: dict[str, torch.Tensor],
    position_ids: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    checkpoints = {
        "attention_mask": mask.float().contiguous(),
        "embedding_output": layer0["embedding_output"].squeeze(0).float().contiguous(),
        "input_ids": torch.tensor(contract["input_token_ids"], dtype=torch.int64),
        "layer0_attention_output": layer0["attention_output"].squeeze(0).float().contiguous(),
        "layer0_expert_input": layer0["post_attention_rmsnorm"].squeeze(0).float().contiguous(),
        "layer0_residual_output": layer0["residual_output"].squeeze(0).float().contiguous(),
        "layer0_router_logits": layer0["router_logits"].float().contiguous(),
        "layer0_routing_weights": layer0["routing_weights"].float().contiguous(),
        "layer0_selected_expert_ids": layer0["selected_expert_ids"].to(torch.int64).contiguous(),
        "layer1_attention_output": layer1["attention_output"].squeeze(0).float().contiguous(),
        "layer1_input": experts["layer0_block_output"].clone().contiguous(),
        "layer1_input_rmsnorm": layer1["input_rmsnorm"].squeeze(0).float().contiguous(),
        "layer1_post_attention_rmsnorm": layer1["post_attention_rmsnorm"].squeeze(0).float().contiguous(),
        "layer1_residual_output": layer1["residual_output"].squeeze(0).float().contiguous(),
        "layer1_router_logits": layer1["router_logits"].float().contiguous(),
        "layer1_routing_weights": layer1["routing_weights"].float().contiguous(),
        "layer1_selected_expert_ids": layer1["selected_expert_ids"].to(torch.int64).contiguous(),
        "position_ids": position_ids.squeeze(0).contiguous(),
    }
    checkpoints.update(experts)
    return checkpoints


def inventory(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "bytes": tensor.numel() * tensor.element_size(),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(tensors.items())
    }


def expert_runtime_plan(
    contract: dict[str, Any],
    registry: dict[str, Any],
    expert_manifest: dict[str, Any],
    unique_experts: list[int],
) -> bytes:
    records = {(record["layer"], record["expert"]): record for record in expert_manifest["experts"]}
    shard = expert_manifest["shards"][0]
    lines = [
        "format_version\t1",
        f"model_id\t{contract['model_id']}",
        f"revision\t{contract['revision']}",
        f"root_manifest_sha256\t{registry['root_manifest_sha256']}",
        "artifact_component\texperts",
        f"shard\t0\t{shard['path']}\t{shard['byte_length']}\t{shard['sha256']}",
    ]
    for expert in unique_experts:
        record = records[(0, expert)]
        require(record["shard_id"] == 0, f"Layer-0 expert {expert} is not in shard 0")
        require(record["payload_length"] == 18_874_368, f"Layer-0 expert {expert} payload length")
        lines.append(
            f"expert\t0\t{expert}\tlayer.0.expert.{expert}\t{shard['path']}\t{record['payload_offset']}\t{record['payload_length']}\t{record['sha256']}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def export(args: argparse.Namespace) -> dict[str, Any]:
    contract = read_json(args.contract)
    registry = read_json(args.registry)
    source_manifest = read_json(args.source_manifest)
    require(contract["schema_version"] == 1, "unsupported M4.2 contract version")
    require(contract["model_id"] == source_manifest["model"]["id"], "contract model ID mismatch")
    require(contract["revision"] == source_manifest["model"]["revision"], "contract revision mismatch")
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
    dense_manifest = read_json(artifact_root / root_manifest["components"]["dense"]["manifest"]["path"])
    dense_records = {record["name"]: record for record in dense_manifest["tensors"]}
    expert_manifest_path = artifact_root / root_manifest["components"]["experts"]["manifest"]["path"]
    require(
        sha256_file(expert_manifest_path) == root_manifest["components"]["experts"]["manifest"]["sha256"],
        "expert manifest hash mismatch",
    )
    expert_manifest = read_json(expert_manifest_path)

    shards, dense_plan, _ = parse_plan(args.dense_plan)
    expert_shards, projections = parse_expert_source_plan(args.expert_source_plan)
    required_names = ["model.embed_tokens.weight", *dense_names(0), *dense_names(1)]
    require(all(name in dense_plan and name in dense_records for name in required_names), "required dense tensor missing")
    source_shard_ids = {dense_plan[name]["shard_id"] for name in required_names}
    source_shard_ids.update(record["shard_id"] for key, record in projections.items() if key[0] == 0)
    require(source_shard_ids == {0}, f"Layer-0/1 source tensors span unexpected shards: {source_shard_ids}")
    require(expert_shards[0] == shards[0], "dense and expert source shard identity differs")
    shard = shards[0]
    source_path = args.source_root / shard["path"]
    require(source_path.stat().st_size == shard["bytes"], "source shard size mismatch")
    require(sha256_file(source_path) == shard["sha256"], "source shard hash mismatch")

    loaded: dict[str, torch.Tensor] = {}
    source_payload_bytes = 0
    for name in required_names[1:]:
        record = dense_plan[name]
        loaded[name] = read_bf16(source_path, record["offset"], record["length"], record["shape"])
        source_payload_bytes += record["length"]
    embedding_record = dense_plan["model.embed_tokens.weight"]
    row_bytes = 2048 * 2
    rows = []
    for token_id in contract["input_token_ids"]:
        rows.append(read_bf16(source_path, embedding_record["offset"] + token_id * row_bytes, row_bytes, [2048]))
        source_payload_bytes += row_bytes
    embedding = torch.stack(rows).unsqueeze(0)

    torch.manual_seed(contract["determinism"]["seed"])
    torch.set_num_threads(contract["determinism"]["torch_threads"])
    torch.set_num_interop_threads(contract["determinism"]["torch_interop_threads"])
    torch.use_deterministic_algorithms(contract["determinism"]["torch_deterministic_algorithms"])
    config = reference_config()
    sequence_length = len(contract["input_token_ids"])
    position_ids = torch.tensor([contract["position_ids"]], dtype=torch.long)
    mask = torch.zeros((1, 1, sequence_length, sequence_length), dtype=torch.bfloat16)
    disallowed = torch.triu(torch.ones((sequence_length, sequence_length), dtype=torch.bool), diagonal=1)
    mask[0, 0].masked_fill_(disallowed, torch.finfo(torch.bfloat16).min)

    bf16_layer0, bf16_layer0_seconds = execute_pre_router(
        config, loaded, embedding, position_ids, mask, torch.bfloat16, 0
    )
    f32_layer0, f32_layer0_seconds = execute_pre_router(
        config, loaded, embedding, position_ids, mask, torch.float32, 0
    )
    bf16_experts, f32_experts, expert_source_bytes, unique_experts, expert_seconds = execute_selected_experts(
        source_path, projections, bf16_layer0, f32_layer0
    )
    source_payload_bytes += expert_source_bytes
    bf16_layer1, bf16_layer1_seconds = execute_pre_router(
        config,
        loaded,
        bf16_experts["layer0_block_output"].unsqueeze(0).to(torch.bfloat16),
        position_ids,
        mask,
        torch.bfloat16,
        1,
    )
    f32_layer1, f32_layer1_seconds = execute_pre_router(
        config,
        loaded,
        f32_experts["layer0_block_output"].unsqueeze(0),
        position_ids,
        mask,
        torch.float32,
        1,
    )

    bf16_checkpoints = serialize_run(contract, bf16_layer0, bf16_experts, bf16_layer1, position_ids, mask)
    f32_checkpoints = serialize_run(contract, f32_layer0, f32_experts, f32_layer1, position_ids, mask)
    require(bf16_checkpoints.keys() == f32_checkpoints.keys(), "checkpoint key mismatch")
    atomic_safetensors(args.checkpoints, bf16_checkpoints)
    atomic_bytes(args.checkpoint_plan, checkpoint_plan(args.checkpoints))
    atomic_safetensors(args.f32_checkpoints, f32_checkpoints)
    atomic_bytes(args.f32_checkpoint_plan, checkpoint_plan(args.f32_checkpoints))

    runtime_lines = [
        "format_version\t1",
        f"model_id\t{contract['model_id']}",
        f"revision\t{contract['revision']}",
        f"root_manifest_sha256\t{registry['root_manifest_sha256']}",
        f"payload\t{root_manifest['components']['dense']['payload']['path']}\t{dense_manifest['artifact']['byte_length']}\t{dense_manifest['artifact']['sha256']}",
    ]
    for name in required_names:
        record = dense_records[name]
        runtime_lines.append(
            f"tensor\t{name}\t{record['offset']}\t{record['byte_length']}\t{','.join(str(value) for value in record['shape'])}"
        )
    runtime_payload = ("\n".join(runtime_lines) + "\n").encode("utf-8")
    atomic_bytes(args.runtime_plan, runtime_payload)
    expert_runtime_payload = expert_runtime_plan(contract, registry, expert_manifest, unique_experts)
    atomic_bytes(args.expert_runtime_plan, expert_runtime_payload)

    pairwise = {}
    for name in sorted(bf16_checkpoints):
        if bf16_checkpoints[name].dtype == torch.int64:
            pairwise[name] = {"exact": torch.equal(bf16_checkpoints[name], f32_checkpoints[name])}
            continue
        difference = (bf16_checkpoints[name] - f32_checkpoints[name]).abs()
        pairwise[name] = {"maximum_absolute_difference": float(difference.max())}
    peak_working_set = process_peak_working_set()
    base_evidence = {
        "schema_version": 1,
        "model_id": contract["model_id"],
        "revision": contract["revision"],
        "validated_path": "complete_layer_0_then_layer_1_pre_router",
        "input_token_ids": contract["input_token_ids"],
        "position_ids": contract["position_ids"],
        "environment": versions,
        "determinism": contract["determinism"],
        "source_shards_verified": [{"shard_id": 0, **shard}],
        "source_hash_bytes_read": shard["bytes"],
        "source_payload_bytes_read": source_payload_bytes,
        "selected_layer0_unique_experts": unique_experts,
        "selected_layer0_logical_expert_count": len(unique_experts),
        "selected_layer0_occurrence_count": 32,
        "runtime_plan": {"bytes": len(runtime_payload), "sha256": hashlib.sha256(runtime_payload).hexdigest()},
        "expert_runtime_plan": {
            "bytes": len(expert_runtime_payload),
            "sha256": hashlib.sha256(expert_runtime_payload).hexdigest(),
        },
        "bf16_vs_f32": pairwise,
        "layer0_router_boundaries": router_boundaries(bf16_layer0),
        "layer1_router_boundaries": router_boundaries(bf16_layer1),
        "layer1_expert_execution": False,
    }
    bf16_evidence = {
        **base_evidence,
        "status": "bf16_reference_export_passed",
        "compute_dtype": "BF16",
        "checkpoint_file": {"bytes": args.checkpoints.stat().st_size, "sha256": sha256_file(args.checkpoints)},
        "checkpoint_inventory": inventory(bf16_checkpoints),
    }
    f32_evidence = {
        **base_evidence,
        "status": "f32_control_export_passed",
        "compute_dtype": "F32",
        "weight_derivation": "exact BF16 values decoded to F32; no different pretrained weights",
        "checkpoint_file": {
            "bytes": args.f32_checkpoints.stat().st_size,
            "sha256": sha256_file(args.f32_checkpoints),
        },
        "checkpoint_inventory": inventory(f32_checkpoints),
        "layer0_router_boundaries": router_boundaries(f32_layer0),
        "layer1_router_boundaries": router_boundaries(f32_layer1),
    }
    bf16_evidence_payload = canonical_json(bf16_evidence)
    f32_evidence_payload = canonical_json(f32_evidence)
    atomic_bytes(args.evidence, bf16_evidence_payload)
    atomic_bytes(args.f32_evidence, f32_evidence_payload)
    return {
        "status": "passed",
        "bf16_checkpoint_sha256": bf16_evidence["checkpoint_file"]["sha256"],
        "f32_checkpoint_sha256": f32_evidence["checkpoint_file"]["sha256"],
        "bf16_evidence_sha256": hashlib.sha256(bf16_evidence_payload).hexdigest(),
        "f32_evidence_sha256": hashlib.sha256(f32_evidence_payload).hexdigest(),
        "unique_experts": len(unique_experts),
        "source_payload_bytes_read": source_payload_bytes,
        "peak_process_working_set_bytes": peak_working_set,
        "execution_seconds": {
            "layer0_bf16": bf16_layer0_seconds,
            "layer0_f32": f32_layer0_seconds,
            "selected_experts_both_paths": expert_seconds,
            "layer1_bf16": bf16_layer1_seconds,
            "layer1_f32": f32_layer1_seconds,
        },
    }


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source_root",
        "registry",
        "source_manifest",
        "dense_plan",
        "expert_source_plan",
        "contract",
        "checkpoints",
        "checkpoint_plan",
        "runtime_plan",
        "expert_runtime_plan",
        "evidence",
        "f32_checkpoints",
        "f32_checkpoint_plan",
        "f32_evidence",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args(arguments)
    for name, value in vars(args).items():
        setattr(args, name, value.resolve())
    try:
        result = export(args)
    except (RouterReferenceError, OSError, KeyError, ValueError) as error:
        print(f"Layer-1 reference error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
