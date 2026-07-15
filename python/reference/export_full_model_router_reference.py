#!/usr/bin/env python3
"""Export bounded pinned Transformers pre-router checkpoints for M4.2-02."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import time
from typing import Any, Iterable

import safetensors
from safetensors.torch import save_file
import torch
import transformers
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeTopKRouter,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.reference.validate_full_model_tensor_values import parse_plan, read_json, sha256_file


class RouterReferenceError(RuntimeError):
    """The pinned reference contract or bounded export failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RouterReferenceError(message)


def canonical_json(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        require(path.read_bytes() == payload, f"existing deterministic output differs: {path.name}")
        return "unchanged"
    temporary = path.with_name(f".{path.name}.incomplete")
    require(not temporary.exists(), f"incomplete output exists: {temporary.name}")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RouterReferenceError(f"cannot commit {path.name}: {error}") from error
    return "written"


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> str:
    temporary = path.with_name(f".{path.name}.incomplete")
    require(not temporary.exists(), f"incomplete output exists: {temporary.name}")
    save_file(tensors, temporary)
    try:
        payload = temporary.read_bytes()
        temporary.unlink()
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RouterReferenceError(f"cannot finalize checkpoint bytes: {error}") from error
    return atomic_bytes(path, payload)


def process_peak_working_set() -> int | None:
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None


def read_bf16(path: Path, offset: int, length: int, shape: list[int]) -> torch.Tensor:
    try:
        with path.open("rb") as source:
            source.seek(offset)
            payload = source.read(length)
    except OSError as error:
        raise RouterReferenceError(f"cannot read {path.name}: {error}") from error
    require(len(payload) == length, f"truncated BF16 tensor range in {path.name}")
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(shape).clone()
    return tensor


def assign_weight(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    target = module
    parts = name.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], torch.nn.Parameter(value, requires_grad=False))


def execute_pre_router(
    config: Qwen3MoeConfig,
    loaded: dict[str, torch.Tensor],
    embedding: torch.Tensor,
    position_ids: torch.Tensor,
    mask: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], float]:
    with torch.device("meta"):
        input_norm = Qwen3MoeRMSNorm(2048, eps=1.0e-6)
        attention = Qwen3MoeAttention(config, 0)
        post_norm = Qwen3MoeRMSNorm(2048, eps=1.0e-6)
        router = Qwen3MoeTopKRouter(config)
    rotary = Qwen3MoeRotaryEmbedding(config, device="cpu")
    assign_weight(input_norm, "weight", loaded["model.layers.0.input_layernorm.weight"].to(dtype))
    assign_weight(attention, "q_proj.weight", loaded["model.layers.0.self_attn.q_proj.weight"].to(dtype))
    assign_weight(attention, "k_proj.weight", loaded["model.layers.0.self_attn.k_proj.weight"].to(dtype))
    assign_weight(attention, "v_proj.weight", loaded["model.layers.0.self_attn.v_proj.weight"].to(dtype))
    assign_weight(attention, "o_proj.weight", loaded["model.layers.0.self_attn.o_proj.weight"].to(dtype))
    assign_weight(attention, "q_norm.weight", loaded["model.layers.0.self_attn.q_norm.weight"].to(dtype))
    assign_weight(attention, "k_norm.weight", loaded["model.layers.0.self_attn.k_norm.weight"].to(dtype))
    assign_weight(post_norm, "weight", loaded["model.layers.0.post_attention_layernorm.weight"].to(dtype))
    assign_weight(router, "weight", loaded["model.layers.0.mlp.gate.weight"].to(dtype))
    for module in (input_norm, attention, post_norm, router, rotary):
        module.eval()

    hidden = embedding.to(dtype)
    started = time.perf_counter()
    with torch.inference_mode():
        normalized = input_norm(hidden)
        position_embeddings = rotary(hidden, position_ids)
        attention_output, _ = attention(normalized, position_embeddings, mask.to(dtype))
        residual = hidden + attention_output
        post_attention = post_norm(residual)
        router_logits, routing_weights, expert_ids = router(post_attention)
    elapsed = time.perf_counter() - started
    return (
        {
            "attention_output": attention_output,
            "embedding_output": hidden,
            "input_rmsnorm": normalized,
            "post_attention_rmsnorm": post_attention,
            "residual_output": residual,
            "router_logits": router_logits,
            "routing_weights": routing_weights,
            "selected_expert_ids": expert_ids,
        },
        elapsed,
    )


def serialized_checkpoints(
    run: dict[str, torch.Tensor],
    contract: dict[str, Any],
    position_ids: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "attention_mask": mask.float().contiguous(),
        "attention_output": run["attention_output"].squeeze(0).float().contiguous(),
        "embedding_output": run["embedding_output"].squeeze(0).float().contiguous(),
        "input_ids": torch.tensor(contract["input_token_ids"], dtype=torch.int64),
        "input_rmsnorm": run["input_rmsnorm"].squeeze(0).float().contiguous(),
        "position_ids": position_ids.squeeze(0).contiguous(),
        "post_attention_rmsnorm": run["post_attention_rmsnorm"].squeeze(0).float().contiguous(),
        "residual_output": run["residual_output"].squeeze(0).float().contiguous(),
        "router_logits": run["router_logits"].float().contiguous(),
        "routing_weights": run["routing_weights"].float().contiguous(),
        "selected_expert_ids": run["selected_expert_ids"].to(torch.int64).contiguous(),
    }


def router_boundaries(run: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    boundaries = []
    logits = run["router_logits"].float()
    expert_ids = run["selected_expert_ids"]
    routing_weights = run["routing_weights"]
    for token_index in range(logits.shape[0]):
        selected = expert_ids[token_index].tolist()
        selected_logits = [float(logits[token_index, expert]) for expert in selected]
        unselected = [expert for expert in range(128) if expert not in set(selected)]
        highest_unselected = max(float(logits[token_index, expert]) for expert in unselected)
        kth = min(selected_logits)
        boundaries.append(
            {
                "expert_ids": selected,
                "highest_unselected_logit": highest_unselected,
                "kth_selected_logit": kth,
                "routing_weights": [float(value) for value in routing_weights[token_index].float()],
                "selected_logits": selected_logits,
                "selection_margin": kth - highest_unselected,
                "token_index": token_index,
                "top_k_boundary_tied": kth == highest_unselected,
            }
        )
    return boundaries


def checkpoint_plan(path: Path) -> bytes:
    payload = path.read_bytes()
    header_length = struct.unpack("<Q", payload[:8])[0]
    header = json.loads(payload[8 : 8 + header_length])
    data_start = 8 + header_length
    lines = ["format_version\t1"]
    for name in sorted(key for key in header if key != "__metadata__"):
        record = header[name]
        start, end = record["data_offsets"]
        shape = ",".join(str(value) for value in record["shape"])
        lines.append(f"tensor\t{name}\t{record['dtype']}\t{data_start + start}\t{end - start}\t{shape}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def export(args: argparse.Namespace) -> dict[str, Any]:
    contract = read_json(args.contract)
    registry = read_json(args.registry)
    source_manifest = read_json(args.source_manifest)
    require(contract["schema_version"] == 1, "unsupported router contract version")
    require(contract["validated_layers"] == [0], "initial export must contain layer 0 only")
    for document, context in ((contract, "contract"), (registry, "registry")):
        require(document["model_id"] == source_manifest["model"]["id"], f"{context} model ID mismatch")
        require(document["revision"] == source_manifest["model"]["revision"], f"{context} revision mismatch")
    actual_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }
    require(actual_versions == contract["environment"], f"reference environment drift: {actual_versions}")

    artifact_root = Path(registry["canonical_artifact_root"])
    root_manifest = read_json(artifact_root / "model-manifest-v1.json")
    require(sha256_file(artifact_root / "model-manifest-v1.json") == registry["root_manifest_sha256"], "root manifest hash mismatch")
    dense_manifest_path = artifact_root / root_manifest["components"]["dense"]["manifest"]["path"]
    dense_manifest = read_json(dense_manifest_path)
    dense_records = {record["name"]: record for record in dense_manifest["tensors"]}

    shards, plan, _ = parse_plan(args.dense_plan)
    required_names = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate.weight",
    ]
    require(all(name in plan and name in dense_records for name in required_names), "required layer-0 tensor is missing")
    selected_shards = {plan[name]["shard_id"] for name in required_names}
    require(selected_shards == {0}, f"initial layer-0 tensors unexpectedly span shards: {selected_shards}")
    shard = shards[0]
    source_path = args.source_root / shard["path"]
    require(source_path.stat().st_size == shard["bytes"], "layer-0 source shard size mismatch")
    require(sha256_file(source_path) == shard["sha256"], "layer-0 source shard hash mismatch")

    source_payload_bytes = 0
    loaded: dict[str, torch.Tensor] = {}
    for name in required_names[1:]:
        record = plan[name]
        loaded[name] = read_bf16(source_path, record["offset"], record["length"], record["shape"])
        source_payload_bytes += record["length"]

    embedding_record = plan[required_names[0]]
    hidden_size = embedding_record["shape"][1]
    row_bytes = hidden_size * 2
    rows = []
    for token_id in contract["input_token_ids"]:
        require(0 <= token_id < embedding_record["shape"][0], f"token ID out of range: {token_id}")
        rows.append(read_bf16(source_path, embedding_record["offset"] + token_id * row_bytes, row_bytes, [hidden_size]))
        source_payload_bytes += row_bytes
    embedding = torch.stack(rows).unsqueeze(0)

    torch.manual_seed(contract["determinism"]["seed"])
    torch.set_num_threads(contract["determinism"]["torch_threads"])
    torch.set_num_interop_threads(contract["determinism"]["torch_interop_threads"])
    torch.use_deterministic_algorithms(contract["determinism"]["torch_deterministic_algorithms"])
    config = Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=40960,
        rms_norm_eps=1.0e-6,
        rope_theta=1_000_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=None,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=True,
    )
    config._attn_implementation = "eager"
    sequence_length = len(contract["input_token_ids"])
    position_ids = torch.tensor([contract["position_ids"]], dtype=torch.long)
    mask = torch.zeros((1, 1, sequence_length, sequence_length), dtype=torch.bfloat16)
    disallowed = torch.triu(torch.ones((sequence_length, sequence_length), dtype=torch.bool), diagonal=1)
    mask[0, 0].masked_fill_(disallowed, torch.finfo(torch.bfloat16).min)

    bf16_run, elapsed = execute_pre_router(config, loaded, embedding, position_ids, mask, torch.bfloat16)
    f32_run, f32_elapsed = execute_pre_router(config, loaded, embedding, position_ids, mask, torch.float32)
    checkpoints = serialized_checkpoints(bf16_run, contract, position_ids, mask)
    f32_checkpoints = serialized_checkpoints(f32_run, contract, position_ids, mask)
    checkpoint_disposition = atomic_safetensors(args.checkpoints, checkpoints)
    checkpoint_plan_payload = checkpoint_plan(args.checkpoints)
    checkpoint_plan_disposition = atomic_bytes(args.checkpoint_plan, checkpoint_plan_payload)
    f32_checkpoint_disposition = atomic_safetensors(args.f32_checkpoints, f32_checkpoints)
    f32_checkpoint_plan_payload = checkpoint_plan(args.f32_checkpoints)
    f32_checkpoint_plan_disposition = atomic_bytes(args.f32_checkpoint_plan, f32_checkpoint_plan_payload)

    payload_relative = root_manifest["components"]["dense"]["payload"]["path"]
    runtime_lines = [
        "format_version\t1",
        f"model_id\t{contract['model_id']}",
        f"revision\t{contract['revision']}",
        f"root_manifest_sha256\t{registry['root_manifest_sha256']}",
        f"payload\t{payload_relative}\t{dense_manifest['artifact']['byte_length']}\t{dense_manifest['artifact']['sha256']}",
    ]
    for name in required_names:
        record = dense_records[name]
        runtime_lines.append(
            f"tensor\t{name}\t{record['offset']}\t{record['byte_length']}\t{','.join(str(value) for value in record['shape'])}"
        )
    runtime_plan_payload = ("\n".join(runtime_lines) + "\n").encode("utf-8")
    runtime_plan_disposition = atomic_bytes(args.runtime_plan, runtime_plan_payload)

    boundaries = router_boundaries(bf16_run)
    f32_boundaries = router_boundaries(f32_run)

    checkpoint_inventory = {
        name: {"bytes": tensor.numel() * tensor.element_size(), "dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape)}
        for name, tensor in sorted(checkpoints.items())
    }
    explicit_bytes = sum(tensor.numel() * tensor.element_size() for tensor in loaded.values())
    explicit_bytes += sum(tensor.numel() * tensor.element_size() for tensor in checkpoints.values())
    evidence = {
        "schema_version": 1,
        "status": "reference_export_passed",
        "model_id": contract["model_id"],
        "revision": contract["revision"],
        "layer": 0,
        "input_token_ids": contract["input_token_ids"],
        "position_ids": contract["position_ids"],
        "attention_mask": contract["attention_mask"],
        "source_dtype": contract["source_dtype"],
        "runtime_dtype": contract["runtime_dtype"],
        "environment": actual_versions,
        "determinism": contract["determinism"],
        "tolerances": contract["tolerances"],
        "source_shards_verified": [{"shard_id": 0, **shard}],
        "source_hash_bytes_read": shard["bytes"],
        "source_payload_bytes_read": source_payload_bytes,
        "artifact_bytes_read": 0,
        "peak_process_working_set_bytes": process_peak_working_set(),
        "peak_explicit_tensor_bytes": explicit_bytes,
        "execution_seconds": elapsed,
        "checkpoint_file": {"bytes": args.checkpoints.stat().st_size, "sha256": sha256_file(args.checkpoints)},
        "checkpoint_inventory": checkpoint_inventory,
        "transformers_raw_checkpoint_shapes": {
            name: list(value.shape) for name, value in bf16_run.items()
        },
        "comparison_batch_axis_removed": True,
        "runtime_plan": {"bytes": len(runtime_plan_payload), "sha256": hashlib.sha256(runtime_plan_payload).hexdigest()},
        "router_boundaries": boundaries,
        "expert_execution": False,
    }
    evidence_payload = canonical_json(evidence)
    evidence_disposition = atomic_bytes(args.evidence, evidence_payload)

    pairwise_errors = {}
    for name in (
        "input_rmsnorm",
        "attention_output",
        "residual_output",
        "post_attention_rmsnorm",
        "router_logits",
        "routing_weights",
    ):
        difference = (f32_checkpoints[name] - checkpoints[name]).abs()
        pairwise_errors[name] = {
            "maximum_absolute_difference": float(difference.max()),
            "maximum_relative_difference": float(
                (difference / checkpoints[name].abs().clamp_min(torch.finfo(torch.float32).tiny)).max()
            ),
        }
    f32_inventory = {
        name: {"bytes": tensor.numel() * tensor.element_size(), "dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape)}
        for name, tensor in sorted(f32_checkpoints.items())
    }
    f32_evidence = {
        "schema_version": 1,
        "status": "f32_control_export_passed",
        "model_id": contract["model_id"],
        "revision": contract["revision"],
        "layer": 0,
        "input_token_ids": contract["input_token_ids"],
        "position_ids": contract["position_ids"],
        "attention_mask": contract["attention_mask"],
        "source_storage_dtype": "BF16",
        "control_compute_dtype": "F32",
        "weight_derivation": "exact BF16 values decoded to F32; no different pretrained weights",
        "environment": actual_versions,
        "determinism": contract["determinism"],
        "f32_correctness_tolerance": {
            "default": {"absolute": 1.0e-6, "relative": 1.0e-5},
            "router_logits": {"absolute": 1.0e-7, "relative": 1.0e-6},
        },
        "execution_seconds": f32_elapsed,
        "checkpoint_file": {"bytes": args.f32_checkpoints.stat().st_size, "sha256": sha256_file(args.f32_checkpoints)},
        "checkpoint_inventory": f32_inventory,
        "router_boundaries": f32_boundaries,
        "bf16_vs_f32": pairwise_errors,
        "expert_execution": False,
    }
    f32_evidence_payload = canonical_json(f32_evidence)
    f32_evidence_disposition = atomic_bytes(args.f32_evidence, f32_evidence_payload)
    return {
        "checkpoint_write": checkpoint_disposition,
        "checkpoint_plan_write": checkpoint_plan_disposition,
        "evidence_sha256": hashlib.sha256(evidence_payload).hexdigest(),
        "evidence_write": evidence_disposition,
        "f32_checkpoint_plan_write": f32_checkpoint_plan_disposition,
        "f32_checkpoint_write": f32_checkpoint_disposition,
        "f32_evidence_sha256": hashlib.sha256(f32_evidence_payload).hexdigest(),
        "f32_evidence_write": f32_evidence_disposition,
        "runtime_plan_write": runtime_plan_disposition,
        "status": "passed",
    }


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dense-plan", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--checkpoint-plan", type=Path, required=True)
    parser.add_argument("--runtime-plan", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--f32-checkpoints", type=Path, required=True)
    parser.add_argument("--f32-checkpoint-plan", type=Path, required=True)
    parser.add_argument("--f32-evidence", type=Path, required=True)
    args = parser.parse_args(arguments)
    for name in ("source_root", "registry", "source_manifest", "dense_plan", "contract", "checkpoints", "checkpoint_plan", "runtime_plan", "evidence", "f32_checkpoints", "f32_checkpoint_plan", "f32_evidence"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        result = export(args)
    except (RouterReferenceError, OSError, KeyError, ValueError) as error:
        print(f"router reference error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
