#!/usr/bin/env python3
"""Export the M4.2-04 short cached-generation reference."""

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
from safetensors.torch import load_file
import torch
import torch.nn.functional as functional
from transformers import DynamicCache
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeTopKRouter,
)
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.reference.export_full_model_router_reference import (
    RouterReferenceError,
    assign_weight,
    atomic_bytes,
    atomic_safetensors,
    canonical_json,
    checkpoint_plan,
    execute_pre_router,
    process_peak_working_set,
    read_bf16,
    reference_config,
    require,
)
from python.reference.export_layer1_router_reference import (
    dense_names,
    parse_expert_source_plan,
    selected_occurrences,
)
from python.reference.export_layer24_router_reference import (
    EXPERT_SHAPE_BY_ROLE,
    load_dense_layer,
    source_tensor,
)
from python.reference.validate_full_model_tensor_values import parse_plan, read_json, sha256_file


PROMPT = [9707, 11, 1879, 0]
GENERATED_COUNT = 2
GUARD_LAYERS = (0, 24, 47)
FIXED_VOCABULARY_INDICES = (0, 1, 2, 11, 1879, 9707, 32768, 65535, 100000, 151935)


def execute_cached_pre_router(
    config: Any,
    loaded: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    position: int,
    cache: DynamicCache,
    dtype: torch.dtype,
    layer: int,
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}"
    with torch.device("meta"):
        input_norm = Qwen3MoeRMSNorm(2048, eps=1.0e-6)
        attention = Qwen3MoeAttention(config, layer)
        post_norm = Qwen3MoeRMSNorm(2048, eps=1.0e-6)
        router = Qwen3MoeTopKRouter(config)
    rotary = Qwen3MoeRotaryEmbedding(config, device="cpu")
    assign_weight(input_norm, "weight", loaded[f"{prefix}.input_layernorm.weight"].to(dtype))
    assign_weight(attention, "q_proj.weight", loaded[f"{prefix}.self_attn.q_proj.weight"].to(dtype))
    assign_weight(attention, "k_proj.weight", loaded[f"{prefix}.self_attn.k_proj.weight"].to(dtype))
    assign_weight(attention, "v_proj.weight", loaded[f"{prefix}.self_attn.v_proj.weight"].to(dtype))
    assign_weight(attention, "o_proj.weight", loaded[f"{prefix}.self_attn.o_proj.weight"].to(dtype))
    assign_weight(attention, "q_norm.weight", loaded[f"{prefix}.self_attn.q_norm.weight"].to(dtype))
    assign_weight(attention, "k_norm.weight", loaded[f"{prefix}.self_attn.k_norm.weight"].to(dtype))
    assign_weight(post_norm, "weight", loaded[f"{prefix}.post_attention_layernorm.weight"].to(dtype))
    assign_weight(router, "weight", loaded[f"{prefix}.mlp.gate.weight"].to(dtype))
    for module in (input_norm, attention, post_norm, router, rotary):
        module.eval()
    position_ids = torch.tensor([[position]], dtype=torch.long)
    with torch.inference_mode():
        normalized = input_norm(hidden.to(dtype))
        position_embeddings = rotary(hidden.to(dtype), position_ids)
        attention_output, _ = attention(
            normalized,
            position_embeddings,
            None,
            past_key_values=cache,
        )
        residual = hidden.to(dtype) + attention_output
        post_attention = post_norm(residual)
        router_logits, routing_weights, expert_ids = router(post_attention)
    return {
        "attention_output": attention_output,
        "input_rmsnorm": normalized,
        "post_attention_rmsnorm": post_attention,
        "residual_output": residual,
        "router_logits": router_logits,
        "routing_weights": routing_weights,
        "selected_expert_ids": expert_ids,
    }


def execute_experts(
    source_root: Path,
    shards: dict[int, dict[str, Any]],
    projections: dict[tuple[int, int, str], dict[str, Any]],
    layer: int,
    runs: dict[str, dict[str, Any]],
    capture_outputs: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    selected = {
        name: sorted(set(int(value) for value in run["selected_expert_ids"].flatten().tolist()))
        for name, run in runs.items()
    }
    union = sorted(set().union(*selected.values()))
    combined = {
        name: torch.zeros_like(run["post_attention_rmsnorm"].squeeze(0))
        for name, run in runs.items()
    }
    captured = {
        name: torch.zeros((8, 2048), dtype=run["dtype"])
        for name, run in runs.items()
    }
    payload_bytes = 0
    for expert in union:
        weights: dict[str, torch.Tensor] = {}
        for role, shape in EXPERT_SHAPE_BY_ROLE.items():
            record = projections[(layer, expert, role)]
            require(record["shape"] == shape, f"Layer-{layer} expert-{expert} {role} shape")
            weights[role] = source_tensor(source_root, shards, record)
            payload_bytes += record["length"]
        for name, run in runs.items():
            if expert not in selected[name]:
                continue
            positions, tokens = selected_occurrences(run["selected_expert_ids"], expert)
            dtype = run["dtype"]
            current = run["post_attention_rmsnorm"].squeeze(0)[tokens].to(dtype)
            gate_up = torch.cat((weights["gate"], weights["up"]), dim=0).to(dtype)
            gate, up = functional.linear(current, gate_up).chunk(2, dim=-1)
            output = functional.linear(functional.silu(gate) * up, weights["down"].to(dtype))
            weighted = output * run["routing_weights"][tokens, positions, None].to(dtype)
            combined[name].index_add_(0, tokens, weighted.to(dtype))
            if capture_outputs:
                for index, position in enumerate(positions.tolist()):
                    captured[name][position] = output[index]
    return combined, captured, payload_bytes


def final_norm_module(weight: torch.Tensor, dtype: torch.dtype) -> Qwen3MoeRMSNorm:
    with torch.device("meta"):
        module = Qwen3MoeRMSNorm(2048, eps=1.0e-6)
    assign_weight(module, "weight", weight.to(dtype))
    module.eval()
    return module


def top_summary(logits: torch.Tensor) -> dict[str, Any]:
    values = logits.float().reshape(-1)
    require(values.numel() == 151936, "vocabulary logit count")
    require(bool(torch.isfinite(values).all()), "non-finite vocabulary logit")
    top_values, top_ids = torch.topk(values, 20)
    pairs = sorted(
        ((int(token), float(value)) for token, value in zip(top_ids, top_values)),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return {
        "argmax_token_id": pairs[0][0],
        "argmax_logit": pairs[0][1],
        "second_highest_logit": pairs[1][1],
        "top1_margin": pairs[0][1] - pairs[1][1],
        "top20_token_ids": [pair[0] for pair in pairs],
        "top20_logits": [pair[1] for pair in pairs],
        "fixed_indices": list(FIXED_VOCABULARY_INDICES),
        "fixed_logits": [float(values[index]) for index in FIXED_VOCABULARY_INDICES],
        "nan_count": int(torch.isnan(values).sum()),
        "positive_infinity_count": int(torch.isposinf(values).sum()),
        "negative_infinity_count": int(torch.isneginf(values).sum()),
        "vocabulary_size": values.numel(),
    }


def final_dense_runtime_plan(
    contract: dict[str, Any],
    registry: dict[str, Any],
    root_manifest: dict[str, Any],
    dense_manifest: dict[str, Any],
) -> bytes:
    records = {record["name"]: record for record in dense_manifest["tensors"]}
    lines = [
        "format_version\t1",
        f"model_id\t{contract['model_id']}",
        f"revision\t{contract['revision']}",
        f"root_manifest_sha256\t{registry['root_manifest_sha256']}",
        f"payload\t{root_manifest['components']['dense']['payload']['path']}\t{dense_manifest['artifact']['byte_length']}\t{dense_manifest['artifact']['sha256']}",
    ]
    for name in ("model.norm.weight", "lm_head.weight"):
        record = records[name]
        shape = ",".join(str(value) for value in record["shape"])
        lines.append(
            f"tensor\t{name}\t{record['offset']}\t{record['byte_length']}\t{shape}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def layer47_expert_runtime_plan(
    contract: dict[str, Any],
    registry: dict[str, Any],
    expert_manifest: dict[str, Any],
) -> bytes:
    shard = expert_manifest["shards"][47]
    records = {
        record["expert"]: record
        for record in expert_manifest["experts"]
        if record["layer"] == 47
    }
    lines = [
        "format_version\t1",
        f"model_id\t{contract['model_id']}",
        f"revision\t{contract['revision']}",
        f"root_manifest_sha256\t{registry['root_manifest_sha256']}",
        "artifact_component\texperts",
        f"shard\t47\t{shard['path']}\t{shard['byte_length']}\t{shard['sha256']}",
    ]
    for expert in range(128):
        record = records[expert]
        lines.append(
            f"expert\t47\t{expert}\tlayer.47.expert.{expert}\t{shard['path']}\t"
            f"{record['payload_offset']}\t{record['payload_length']}\t{record['sha256']}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def execute_incremental(
    args: argparse.Namespace,
    config: Any,
    source_shards: dict[int, dict[str, Any]],
    dense_plan: dict[str, dict[str, Any]],
    projections: dict[tuple[int, int, str], dict[str, Any]],
    final_norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    list[dict[str, Any]],
    list[float],
    int,
]:
    caches = {
        "bf16": DynamicCache(config=config),
        "f32": DynamicCache(config=config),
    }
    dtypes = {"bf16": torch.bfloat16, "f32": torch.float32}
    norm_modules = {
        name: final_norm_module(final_norm_weight, dtype) for name, dtype in dtypes.items()
    }
    compact = {"bf16": {}, "f32": {}}
    full_logits = {"bf16": {}, "f32": {}}
    inputs = list(PROMPT)
    generated: list[int] = []
    step_evidence: list[dict[str, Any]] = []
    step_seconds: list[float] = []
    payload_bytes = 0
    for step in range(len(PROMPT) + GENERATED_COUNT):
        token = inputs[step]
        embedding_record = dense_plan["model.embed_tokens.weight"]
        row_record = {
            **embedding_record,
            "offset": embedding_record["offset"] + token * 4096,
            "length": 4096,
            "shape": [2048],
        }
        embedding = source_tensor(args.source_root, source_shards, row_record)
        payload_bytes += 4096
        hidden = {
            "bf16": embedding.reshape(1, 1, 2048),
            "f32": embedding.float().reshape(1, 1, 2048),
        }
        guards: dict[str, dict[int, list[int]]] = {"bf16": {}, "f32": {}}
        layer47: dict[str, dict[str, torch.Tensor]] = {}
        started = time.perf_counter()
        for layer in range(48):
            loaded, dense_bytes = load_dense_layer(
                args.source_root, source_shards, dense_plan, layer
            )
            payload_bytes += dense_bytes
            runs: dict[str, dict[str, Any]] = {}
            for name, dtype in dtypes.items():
                run = execute_cached_pre_router(
                    config,
                    loaded,
                    hidden[name],
                    step,
                    caches[name],
                    dtype,
                    layer,
                )
                run["dtype"] = dtype
                runs[name] = run
                if layer in GUARD_LAYERS:
                    guards[name][layer] = [
                        int(value) for value in run["selected_expert_ids"].reshape(-1).tolist()
                    ]
            combined, expert_outputs, expert_bytes = execute_experts(
                args.source_root,
                source_shards,
                projections,
                layer,
                runs,
                capture_outputs=layer == 47,
            )
            payload_bytes += expert_bytes
            for name in dtypes:
                block = runs[name]["residual_output"].squeeze(0) + combined[name]
                hidden[name] = block.unsqueeze(0).to(dtypes[name])
                if layer == 47:
                    layer47[name] = {
                        "expert_ids": runs[name]["selected_expert_ids"].reshape(-1).to(torch.int64),
                        "routing_weights": runs[name]["routing_weights"].reshape(-1).float(),
                        "expert_outputs": expert_outputs[name].float(),
                        "moe_output": combined[name].reshape(-1).float(),
                        "block_output": block.reshape(-1).float(),
                    }
            del loaded, runs, combined, expert_outputs

        summaries: dict[str, Any] = {}
        for name, dtype in dtypes.items():
            with torch.inference_mode():
                normalized = norm_modules[name](hidden[name].to(dtype))
                logits = functional.linear(normalized, lm_head.to(dtype)).reshape(-1).float()
            prefix = f"step{step}"
            compact[name][f"{prefix}_input_token"] = torch.tensor([token], dtype=torch.int64)
            compact[name][f"{prefix}_position"] = torch.tensor([step], dtype=torch.int64)
            compact[name][f"{prefix}_guard_router_ids"] = torch.tensor(
                [guards[name][layer] for layer in GUARD_LAYERS], dtype=torch.int64
            )
            for checkpoint, tensor in layer47[name].items():
                compact[name][f"{prefix}_layer47_{checkpoint}"] = tensor.contiguous()
            compact[name][f"{prefix}_final_norm"] = normalized.reshape(-1).float().contiguous()
            summary = top_summary(logits)
            compact[name][f"{prefix}_top20_ids"] = torch.tensor(
                summary["top20_token_ids"], dtype=torch.int64
            )
            compact[name][f"{prefix}_top20_logits"] = torch.tensor(
                summary["top20_logits"], dtype=torch.float32
            )
            compact[name][f"{prefix}_fixed_logits"] = torch.tensor(
                summary["fixed_logits"], dtype=torch.float32
            )
            full_logits[name][f"{prefix}_logits"] = logits.contiguous()
            summaries[name] = summary
        selected = summaries["f32"]["argmax_token_id"]
        if step >= len(PROMPT) - 1 and len(generated) < GENERATED_COUNT:
            generated.append(selected)
            inputs.append(selected)
        step_seconds.append(time.perf_counter() - started)
        step_evidence.append(
            {
                "step": step,
                "input_token": token,
                "position": step,
                "cache_lengths": {
                    name: caches[name].get_seq_length() for name in caches
                },
                "guard_router_ids": guards,
                "bf16": summaries["bf16"],
                "f32": summaries["f32"],
            }
        )
    require(len(generated) == GENERATED_COUNT, "generated token count")
    return (
        compact["bf16"],
        compact["f32"],
        full_logits["bf16"],
        full_logits["f32"],
        step_evidence,
        step_seconds,
        payload_bytes,
    )


def execute_f32_recompute(
    args: argparse.Namespace,
    config: Any,
    source_shards: dict[int, dict[str, Any]],
    dense_plan: dict[str, dict[str, Any]],
    projections: dict[tuple[int, int, str], dict[str, Any]],
    final_norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    sequence: list[int],
) -> tuple[torch.Tensor, int]:
    embedding_record = dense_plan["model.embed_tokens.weight"]
    rows = []
    payload_bytes = 0
    for token in sequence:
        row_record = {
            **embedding_record,
            "offset": embedding_record["offset"] + token * 4096,
            "length": 4096,
            "shape": [2048],
        }
        rows.append(source_tensor(args.source_root, source_shards, row_record))
        payload_bytes += 4096
    hidden = torch.stack(rows).unsqueeze(0).float()
    length = len(sequence)
    positions = torch.arange(length, dtype=torch.long).unsqueeze(0)
    mask = torch.zeros((1, 1, length, length), dtype=torch.bfloat16)
    disallowed = torch.triu(torch.ones((length, length), dtype=torch.bool), diagonal=1)
    mask[0, 0].masked_fill_(disallowed, torch.finfo(torch.bfloat16).min)
    for layer in range(48):
        loaded, dense_bytes = load_dense_layer(args.source_root, source_shards, dense_plan, layer)
        payload_bytes += dense_bytes
        run, _ = execute_pre_router(
            config, loaded, hidden, positions, mask, torch.float32, layer
        )
        run["dtype"] = torch.float32
        combined, _, expert_bytes = execute_experts(
            args.source_root,
            source_shards,
            projections,
            layer,
            {"f32": run},
            capture_outputs=False,
        )
        payload_bytes += expert_bytes
        hidden = (run["residual_output"].squeeze(0) + combined["f32"]).unsqueeze(0)
    norm = final_norm_module(final_norm_weight, torch.float32)
    with torch.inference_mode():
        normalized = norm(hidden)
        logits = functional.linear(normalized[:, -1:], lm_head.float()).reshape(-1).float()
    return logits, payload_bytes


def export(args: argparse.Namespace) -> dict[str, Any]:
    contract = read_json(args.contract)
    registry = read_json(args.registry)
    source_manifest = read_json(args.source_manifest)
    require(contract["model_id"] == source_manifest["model"]["id"], "model ID mismatch")
    require(contract["revision"] == source_manifest["model"]["revision"], "revision mismatch")
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }
    require(versions == contract["environment"], f"reference environment drift: {versions}")
    artifact_root = Path(registry["canonical_artifact_root"])
    root_manifest_path = artifact_root / "model-manifest-v1.json"
    require(sha256_file(root_manifest_path) == registry["root_manifest_sha256"], "root hash")
    root_manifest = read_json(root_manifest_path)
    dense_manifest = read_json(artifact_root / root_manifest["components"]["dense"]["manifest"]["path"])
    expert_manifest = read_json(artifact_root / root_manifest["components"]["experts"]["manifest"]["path"])
    source_shards, dense_plan, _ = parse_plan(args.dense_plan)
    expert_shards, projections = parse_expert_source_plan(args.expert_source_plan)
    require(source_shards == expert_shards, "source shard identities")

    required_names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer in range(48):
        required_names.extend(dense_names(layer))
    required_shards = {dense_plan[name]["shard_id"] for name in required_names}
    required_shards.update(record["shard_id"] for record in projections.values())
    verified_shards = []
    source_hash_bytes = 0
    for shard_id in sorted(required_shards):
        record = source_shards[shard_id]
        path = args.source_root / record["path"]
        require(path.stat().st_size == record["bytes"], f"source shard {shard_id} size")
        require(sha256_file(path) == record["sha256"], f"source shard {shard_id} hash")
        verified_shards.append({"shard_id": shard_id, **record})
        source_hash_bytes += record["bytes"]

    torch.manual_seed(contract["determinism"]["seed"])
    torch.set_num_threads(contract["determinism"]["torch_threads"])
    torch.set_num_interop_threads(contract["determinism"]["torch_interop_threads"])
    torch.use_deterministic_algorithms(contract["determinism"]["torch_deterministic_algorithms"])
    config = reference_config()
    final_norm_record = dense_plan["model.norm.weight"]
    final_norm_weight = source_tensor(args.source_root, source_shards, final_norm_record)
    lm_head_record = dense_plan["lm_head.weight"]
    lm_head = source_tensor(args.source_root, source_shards, lm_head_record)
    source_payload_bytes = final_norm_record["length"] + lm_head_record["length"]

    (
        bf16_compact,
        f32_compact,
        bf16_logits,
        f32_logits,
        steps,
        step_seconds,
        incremental_bytes,
    ) = execute_incremental(
        args,
        config,
        source_shards,
        dense_plan,
        projections,
        final_norm_weight,
        lm_head,
    )
    source_payload_bytes += incremental_bytes
    generated = [steps[3]["f32"]["argmax_token_id"], steps[4]["f32"]["argmax_token_id"]]
    recompute = []
    sequence = list(PROMPT)
    for checkpoint_step in (3, 4, 5):
        if checkpoint_step > 3:
            sequence.append(generated[checkpoint_step - 4])
        logits, recompute_bytes = execute_f32_recompute(
            args,
            config,
            source_shards,
            dense_plan,
            projections,
            final_norm_weight,
            lm_head,
            sequence,
        )
        source_payload_bytes += recompute_bytes
        incremental = f32_logits[f"step{checkpoint_step}_logits"]
        difference = (incremental - logits).abs()
        recompute.append(
            {
                "step": checkpoint_step,
                "sequence": list(sequence),
                "maximum_incremental_vs_recompute_absolute_difference": float(difference.max()),
                "incremental_argmax": int(torch.argmax(incremental)),
                "recompute_argmax": int(torch.argmax(logits)),
                "recompute_top1_margin": top_summary(logits)["top1_margin"],
            }
        )
        f32_compact[f"step{checkpoint_step}_recompute_argmax"] = torch.tensor(
            [int(torch.argmax(logits))], dtype=torch.int64
        )

    atomic_safetensors(args.bf16_checkpoints, bf16_compact)
    atomic_safetensors(args.f32_checkpoints, f32_compact)
    atomic_bytes(args.bf16_plan, checkpoint_plan(args.bf16_checkpoints))
    atomic_bytes(args.f32_plan, checkpoint_plan(args.f32_checkpoints))
    atomic_safetensors(args.bf16_full_logits, bf16_logits)
    atomic_safetensors(args.f32_full_logits, f32_logits)
    atomic_bytes(args.bf16_full_logits_plan, checkpoint_plan(args.bf16_full_logits))
    atomic_bytes(args.f32_full_logits_plan, checkpoint_plan(args.f32_full_logits))
    dense_runtime = final_dense_runtime_plan(contract, registry, root_manifest, dense_manifest)
    expert_runtime = layer47_expert_runtime_plan(contract, registry, expert_manifest)
    atomic_bytes(args.final_dense_runtime_plan, dense_runtime)
    atomic_bytes(args.layer47_expert_runtime_plan, expert_runtime)

    base = {
        "schema_version": 1,
        "model_id": contract["model_id"],
        "revision": contract["revision"],
        "status": "short_cached_generation_reference_passed",
        "prompt_token_ids": PROMPT,
        "generated_token_ids": generated,
        "processed_token_ids": PROMPT + generated,
        "generated_count": GENERATED_COUNT,
        "guard_layers": list(GUARD_LAYERS),
        "fixed_vocabulary_indices": list(FIXED_VOCABULARY_INDICES),
        "steps": steps,
        "f32_incremental_vs_full_recompute": recompute,
        "source_shards_verified": verified_shards,
        "source_hash_bytes_read": source_hash_bytes,
        "source_payload_bytes_read": source_payload_bytes,
        "runtime_plans": {
            "final_dense_sha256": hashlib.sha256(dense_runtime).hexdigest(),
            "layer47_expert_sha256": hashlib.sha256(expert_runtime).hexdigest(),
        },
        "environment": versions,
    }
    outputs = {
        "bf16": (args.bf16_checkpoints, args.bf16_full_logits, args.bf16_evidence, "BF16"),
        "f32": (args.f32_checkpoints, args.f32_full_logits, args.f32_evidence, "F32"),
    }
    result: dict[str, Any] = {
        "status": "passed",
        "generated_token_ids": generated,
        "source_hash_bytes_read": source_hash_bytes,
        "source_payload_bytes_read": source_payload_bytes,
        "peak_process_working_set_bytes": process_peak_working_set(),
        "step_seconds": step_seconds,
    }
    for name, (checkpoints, full, evidence, dtype) in outputs.items():
        document = {
            **base,
            "compute_dtype": dtype,
            "checkpoint_file": {
                "bytes": checkpoints.stat().st_size,
                "sha256": sha256_file(checkpoints),
            },
            "temporary_full_logits_file": {
                "bytes": full.stat().st_size,
                "sha256": sha256_file(full),
                "retained_after_validation": False,
            },
        }
        payload = canonical_json(document)
        atomic_bytes(evidence, payload)
        result[f"{name}_checkpoints_sha256"] = document["checkpoint_file"]["sha256"]
        result[f"{name}_full_logits_sha256"] = document["temporary_full_logits_file"]["sha256"]
        result[f"{name}_evidence_sha256"] = hashlib.sha256(payload).hexdigest()
    return result


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source_root",
        "registry",
        "source_manifest",
        "dense_plan",
        "expert_source_plan",
        "contract",
        "bf16_checkpoints",
        "f32_checkpoints",
        "bf16_plan",
        "f32_plan",
        "bf16_full_logits",
        "f32_full_logits",
        "bf16_full_logits_plan",
        "f32_full_logits_plan",
        "final_dense_runtime_plan",
        "layer47_expert_runtime_plan",
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
    except (RouterReferenceError, OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"short generation reference error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
