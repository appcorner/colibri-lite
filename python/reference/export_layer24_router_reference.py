#!/usr/bin/env python3
"""Export genuine Layers 0-23 and Layer-24 router reference checkpoints."""

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
from python.reference.export_layer1_router_reference import (
    dense_names,
    execute_one_expert,
    parse_expert_source_plan,
    selected_occurrences,
)
from python.reference.validate_full_model_tensor_values import parse_plan, read_json, sha256_file


COMPLETED_LAYERS = range(24)
GUARD_LAYERS = {0, 1, 8, 16, 23, 24}
EXPERT_SHAPE_BY_ROLE = {
    "gate": [768, 2048],
    "up": [768, 2048],
    "down": [2048, 768],
}


def source_tensor(
    source_root: Path,
    shards: dict[int, dict[str, Any]],
    record: dict[str, Any],
) -> torch.Tensor:
    shard = shards[record["shard_id"]]
    return read_bf16(
        source_root / shard["path"],
        record["offset"],
        record["length"],
        record["shape"],
    )


def load_dense_layer(
    source_root: Path,
    shards: dict[int, dict[str, Any]],
    dense_plan: dict[str, dict[str, Any]],
    layer: int,
) -> tuple[dict[str, torch.Tensor], int]:
    loaded: dict[str, torch.Tensor] = {}
    payload_bytes = 0
    for name in dense_names(layer):
        record = dense_plan[name]
        loaded[name] = source_tensor(source_root, shards, record)
        payload_bytes += record["length"]
    return loaded, payload_bytes


def execute_selected_layer_experts(
    source_root: Path,
    shards: dict[int, dict[str, Any]],
    projections: dict[tuple[int, int, str], dict[str, Any]],
    layer: int,
    bf16_run: dict[str, torch.Tensor],
    f32_run: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, int, list[int], list[int], float]:
    bf16_ids = bf16_run["selected_expert_ids"]
    f32_ids = f32_run["selected_expert_ids"]
    bf16_unique = sorted(set(int(value) for value in bf16_ids.flatten().tolist()))
    f32_unique = sorted(set(int(value) for value in f32_ids.flatten().tolist()))
    union = sorted(set(bf16_unique) | set(f32_unique))
    bf16_hidden = bf16_run["post_attention_rmsnorm"].squeeze(0)
    f32_hidden = f32_run["post_attention_rmsnorm"].squeeze(0)
    bf16_combined = torch.zeros_like(bf16_hidden)
    f32_combined = torch.zeros_like(f32_hidden)
    payload_bytes = 0
    started = time.perf_counter()
    for expert in union:
        weights: dict[str, torch.Tensor] = {}
        for role, expected_shape in EXPERT_SHAPE_BY_ROLE.items():
            record = projections[(layer, expert, role)]
            require(
                record["shape"] == expected_shape,
                f"Layer-{layer} expert {expert} {role} shape mismatch",
            )
            weights[role] = source_tensor(source_root, shards, record)
            payload_bytes += record["length"]
        if expert in bf16_unique:
            positions, tokens = selected_occurrences(bf16_ids, expert)
            with torch.inference_mode():
                _, weighted = execute_one_expert(
                    bf16_hidden,
                    bf16_run["routing_weights"],
                    positions,
                    tokens,
                    weights["gate"],
                    weights["up"],
                    weights["down"],
                    torch.bfloat16,
                )
                bf16_combined.index_add_(0, tokens, weighted.to(torch.bfloat16))
        if expert in f32_unique:
            positions, tokens = selected_occurrences(f32_ids, expert)
            with torch.inference_mode():
                _, weighted = execute_one_expert(
                    f32_hidden,
                    f32_run["routing_weights"],
                    positions,
                    tokens,
                    weights["gate"],
                    weights["up"],
                    weights["down"],
                    torch.float32,
                )
                f32_combined.index_add_(0, tokens, weighted)
        del weights
    return (
        bf16_combined,
        f32_combined,
        payload_bytes,
        bf16_unique,
        f32_unique,
        time.perf_counter() - started,
    )


def record_pre_router(
    checkpoints: dict[str, torch.Tensor],
    layer: int,
    run: dict[str, torch.Tensor],
) -> None:
    prefix = f"layer{layer}"
    checkpoints[f"{prefix}_input"] = (
        run["embedding_output"].squeeze(0).float().clone().contiguous()
    )
    checkpoints[f"{prefix}_router_logits"] = run["router_logits"].float().contiguous()
    checkpoints[f"{prefix}_routing_weights"] = run["routing_weights"].float().contiguous()
    checkpoints[f"{prefix}_selected_expert_ids"] = (
        run["selected_expert_ids"].to(torch.int64).contiguous()
    )
    checkpoints[f"{prefix}_input_rmsnorm"] = (
        run["input_rmsnorm"].squeeze(0).float().contiguous()
    )
    checkpoints[f"{prefix}_attention_output"] = (
        run["attention_output"].squeeze(0).float().contiguous()
    )
    checkpoints[f"{prefix}_residual_output"] = (
        run["residual_output"].squeeze(0).float().contiguous()
    )
    checkpoints[f"{prefix}_post_attention_rmsnorm"] = (
        run["post_attention_rmsnorm"].squeeze(0).float().contiguous()
    )


def record_completed_layer(
    checkpoints: dict[str, torch.Tensor],
    layer: int,
    moe_output: torch.Tensor,
    block_output: torch.Tensor,
) -> None:
    checkpoints[f"layer{layer}_moe_output"] = moe_output.float().clone().contiguous()
    checkpoints[f"layer{layer}_block_output"] = block_output.float().clone().contiguous()


def inventory(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "bytes": tensor.numel() * tensor.element_size(),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(tensors.items())
    }


def dense_runtime_plan(
    contract: dict[str, Any],
    registry: dict[str, Any],
    root_manifest: dict[str, Any],
    dense_manifest: dict[str, Any],
) -> bytes:
    records = {record["name"]: record for record in dense_manifest["tensors"]}
    names = ["model.embed_tokens.weight"]
    for layer in range(25):
        names.extend(dense_names(layer))
    lines = [
        "format_version\t1",
        f"model_id\t{contract['model_id']}",
        f"revision\t{contract['revision']}",
        f"root_manifest_sha256\t{registry['root_manifest_sha256']}",
        f"payload\t{root_manifest['components']['dense']['payload']['path']}\t{dense_manifest['artifact']['byte_length']}\t{dense_manifest['artifact']['sha256']}",
    ]
    for name in names:
        record = records[name]
        shape = ",".join(str(value) for value in record["shape"])
        lines.append(f"tensor\t{name}\t{record['offset']}\t{record['byte_length']}\t{shape}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def expert_runtime_plan(
    contract: dict[str, Any],
    registry: dict[str, Any],
    expert_manifest: dict[str, Any],
) -> bytes:
    shards = {record["shard_id"]: record for record in expert_manifest["shards"]}
    experts = {
        (record["layer"], record["expert"]): record
        for record in expert_manifest["experts"]
    }
    lines = [
        "format_version\t1",
        f"model_id\t{contract['model_id']}",
        f"revision\t{contract['revision']}",
        f"root_manifest_sha256\t{registry['root_manifest_sha256']}",
        "artifact_component\texperts",
    ]
    for layer in COMPLETED_LAYERS:
        shard = shards[layer]
        lines.append(
            f"shard\t{layer}\t{shard['path']}\t{shard['byte_length']}\t{shard['sha256']}"
        )
        for expert in range(128):
            record = experts[(layer, expert)]
            require(record["shard_id"] == layer, f"Layer-{layer} expert shard mismatch")
            require(record["payload_length"] == 18_874_368, "packed expert length mismatch")
            lines.append(
                f"expert\t{layer}\t{expert}\tlayer.{layer}.expert.{expert}\t{shard['path']}\t{record['payload_offset']}\t{record['payload_length']}\t{record['sha256']}"
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
    expert_manifest = read_json(artifact_root / root_manifest["components"]["experts"]["manifest"]["path"])
    source_shards, dense_plan, _ = parse_plan(args.dense_plan)
    expert_shards, projections = parse_expert_source_plan(args.expert_source_plan)
    require(source_shards == expert_shards, "dense and expert source shard identities differ")

    required_dense_names = ["model.embed_tokens.weight"]
    for layer in range(25):
        required_dense_names.extend(dense_names(layer))
    require(all(name in dense_plan for name in required_dense_names), "required dense source tensor missing")
    required_source_shards = {dense_plan[name]["shard_id"] for name in required_dense_names}
    required_source_shards.update(
        record["shard_id"]
        for key, record in projections.items()
        if key[0] in COMPLETED_LAYERS
    )
    verified_shards = []
    source_hash_bytes = 0
    for shard_id in sorted(required_source_shards):
        shard = source_shards[shard_id]
        path = args.source_root / shard["path"]
        require(path.stat().st_size == shard["bytes"], f"source shard {shard_id} size mismatch")
        require(sha256_file(path) == shard["sha256"], f"source shard {shard_id} hash mismatch")
        verified_shards.append({"shard_id": shard_id, **shard})
        source_hash_bytes += shard["bytes"]

    torch.manual_seed(contract["determinism"]["seed"])
    torch.set_num_threads(contract["determinism"]["torch_threads"])
    torch.set_num_interop_threads(contract["determinism"]["torch_interop_threads"])
    torch.use_deterministic_algorithms(contract["determinism"]["torch_deterministic_algorithms"])
    config = reference_config()
    position_ids = torch.tensor([contract["position_ids"]], dtype=torch.long)
    sequence_length = len(contract["input_token_ids"])
    mask = torch.zeros((1, 1, sequence_length, sequence_length), dtype=torch.bfloat16)
    disallowed = torch.triu(torch.ones((sequence_length, sequence_length), dtype=torch.bool), diagonal=1)
    mask[0, 0].masked_fill_(disallowed, torch.finfo(torch.bfloat16).min)

    embedding_record = dense_plan["model.embed_tokens.weight"]
    rows = []
    source_payload_bytes = 0
    for token_id in contract["input_token_ids"]:
        row_record = {**embedding_record, "offset": embedding_record["offset"] + token_id * 4096, "length": 4096, "shape": [2048]}
        rows.append(source_tensor(args.source_root, source_shards, row_record))
        source_payload_bytes += 4096
    bf16_hidden = torch.stack(rows).unsqueeze(0)
    f32_hidden = bf16_hidden.float()
    bf16_checkpoints: dict[str, torch.Tensor] = {
        "attention_mask": mask.float().contiguous(),
        "embedding_output": bf16_hidden.squeeze(0).float().contiguous(),
        "input_ids": torch.tensor(contract["input_token_ids"], dtype=torch.int64),
        "position_ids": position_ids.squeeze(0).contiguous(),
    }
    f32_checkpoints = {name: tensor.clone() for name, tensor in bf16_checkpoints.items()}
    bf16_boundaries: dict[str, Any] = {}
    f32_boundaries: dict[str, Any] = {}
    layer_experts: list[dict[str, Any]] = []
    layer_seconds: list[dict[str, Any]] = []

    for layer in range(25):
        loaded, dense_bytes = load_dense_layer(args.source_root, source_shards, dense_plan, layer)
        source_payload_bytes += dense_bytes
        bf16_run, bf16_seconds = execute_pre_router(
            config, loaded, bf16_hidden, position_ids, mask, torch.bfloat16, layer
        )
        f32_run, f32_seconds = execute_pre_router(
            config, loaded, f32_hidden, position_ids, mask, torch.float32, layer
        )
        record_pre_router(bf16_checkpoints, layer, bf16_run)
        record_pre_router(f32_checkpoints, layer, f32_run)
        bf16_boundaries[str(layer)] = router_boundaries(bf16_run)
        f32_boundaries[str(layer)] = router_boundaries(f32_run)
        timing = {"layer": layer, "bf16_pre_router": bf16_seconds, "f32_pre_router": f32_seconds}
        if layer == 24:
            layer_seconds.append(timing)
            break
        (
            bf16_moe,
            f32_moe,
            expert_bytes,
            bf16_unique,
            f32_unique,
            expert_seconds,
        ) = execute_selected_layer_experts(
            args.source_root, source_shards, projections, layer, bf16_run, f32_run
        )
        source_payload_bytes += expert_bytes
        bf16_block = bf16_run["residual_output"].squeeze(0) + bf16_moe
        f32_block = f32_run["residual_output"].squeeze(0) + f32_moe
        record_completed_layer(bf16_checkpoints, layer, bf16_moe, bf16_block)
        record_completed_layer(f32_checkpoints, layer, f32_moe, f32_block)
        layer_experts.append(
            {
                "layer": layer,
                "bf16_ids": [int(value) for value in bf16_run["selected_expert_ids"].flatten()],
                "bf16_unique_experts": bf16_unique,
                "f32_ids": [int(value) for value in f32_run["selected_expert_ids"].flatten()],
                "f32_unique_experts": f32_unique,
                "occurrences": 32,
            }
        )
        timing["selected_experts_both_paths"] = expert_seconds
        layer_seconds.append(timing)
        bf16_hidden = bf16_block.unsqueeze(0).to(torch.bfloat16)
        f32_hidden = f32_block.unsqueeze(0)
        del loaded, bf16_run, f32_run, bf16_moe, f32_moe, bf16_block, f32_block

    require(bf16_checkpoints.keys() == f32_checkpoints.keys(), "checkpoint key mismatch")
    atomic_safetensors(args.checkpoints, bf16_checkpoints)
    atomic_bytes(args.checkpoint_plan, checkpoint_plan(args.checkpoints))
    atomic_safetensors(args.f32_checkpoints, f32_checkpoints)
    atomic_bytes(args.f32_checkpoint_plan, checkpoint_plan(args.f32_checkpoints))
    runtime_payload = dense_runtime_plan(contract, registry, root_manifest, dense_manifest)
    expert_runtime_payload = expert_runtime_plan(contract, registry, expert_manifest)
    atomic_bytes(args.runtime_plan, runtime_payload)
    atomic_bytes(args.expert_runtime_plan, expert_runtime_payload)

    pairwise: dict[str, Any] = {}
    for name in sorted(bf16_checkpoints):
        if bf16_checkpoints[name].dtype == torch.int64:
            pairwise[name] = {"exact": torch.equal(bf16_checkpoints[name], f32_checkpoints[name])}
        else:
            difference = (bf16_checkpoints[name] - f32_checkpoints[name]).abs()
            pairwise[name] = {"maximum_absolute_difference": float(difference.max())}
    base_evidence = {
        "schema_version": 1,
        "model_id": contract["model_id"],
        "revision": contract["revision"],
        "validated_path": "embedding_complete_layers_0_through_23_layer_24_router_stop",
        "input_token_ids": contract["input_token_ids"],
        "position_ids": contract["position_ids"],
        "environment": versions,
        "determinism": contract["determinism"],
        "guard_layers": sorted(GUARD_LAYERS),
        "source_shards_verified": verified_shards,
        "source_hash_bytes_read": source_hash_bytes,
        "source_payload_bytes_read": source_payload_bytes,
        "layer_experts": layer_experts,
        "layer24_expert_execution": False,
        "bf16_vs_f32": pairwise,
        "runtime_plan": {"bytes": len(runtime_payload), "sha256": hashlib.sha256(runtime_payload).hexdigest()},
        "expert_runtime_plan": {"bytes": len(expert_runtime_payload), "sha256": hashlib.sha256(expert_runtime_payload).hexdigest()},
    }
    bf16_evidence = {
        **base_evidence,
        "status": "bf16_reference_export_passed",
        "compute_dtype": "BF16",
        "router_boundaries": bf16_boundaries,
        "checkpoint_file": {"bytes": args.checkpoints.stat().st_size, "sha256": sha256_file(args.checkpoints)},
        "checkpoint_inventory": inventory(bf16_checkpoints),
    }
    f32_evidence = {
        **base_evidence,
        "status": "f32_control_export_passed",
        "compute_dtype": "F32",
        "weight_derivation": "exact BF16 values decoded to F32; no different pretrained weights",
        "router_boundaries": f32_boundaries,
        "checkpoint_file": {"bytes": args.f32_checkpoints.stat().st_size, "sha256": sha256_file(args.f32_checkpoints)},
        "checkpoint_inventory": inventory(f32_checkpoints),
    }
    bf16_payload = canonical_json(bf16_evidence)
    f32_payload = canonical_json(f32_evidence)
    atomic_bytes(args.evidence, bf16_payload)
    atomic_bytes(args.f32_evidence, f32_payload)
    return {
        "status": "passed",
        "bf16_checkpoint_sha256": bf16_evidence["checkpoint_file"]["sha256"],
        "f32_checkpoint_sha256": f32_evidence["checkpoint_file"]["sha256"],
        "bf16_evidence_sha256": hashlib.sha256(bf16_payload).hexdigest(),
        "f32_evidence_sha256": hashlib.sha256(f32_payload).hexdigest(),
        "source_hash_bytes_read": source_hash_bytes,
        "source_payload_bytes_read": source_payload_bytes,
        "peak_process_working_set_bytes": process_peak_working_set(),
        "layer_seconds": layer_seconds,
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
        print(f"Layer-24 reference error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
