#!/usr/bin/env python3
"""Measure non-expert precision sensitivity without building a mixed model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any, Iterable

import numpy as np
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.reference.export_full_model_router_reference import atomic_bytes, canonical_json, require


MODEL_DIR = Path("models/qwen3-30b-a3b")
ROOT_MANIFEST_SHA256 = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
LAYERS = (0, 1, 24, 47)
TOKEN_IDS = (0, 11, 9707, 125451, 151643)
GROUPS = (
    "embedding_weights", "attention_q_projection", "attention_k_projection",
    "attention_v_projection", "attention_o_projection", "input_rmsnorm_weights",
    "post_attention_rmsnorm_weights", "q_norm_weights", "k_norm_weights",
    "router_weights", "final_rmsnorm_weights", "lm_head_weights",
)
GROUP_TO_SUFFIX = {
    "attention_q_projection": "self_attn.q_proj.weight",
    "attention_k_projection": "self_attn.k_proj.weight",
    "attention_v_projection": "self_attn.v_proj.weight",
    "attention_o_projection": "self_attn.o_proj.weight",
    "input_rmsnorm_weights": "input_layernorm.weight",
    "post_attention_rmsnorm_weights": "post_attention_layernorm.weight",
    "q_norm_weights": "self_attn.q_norm.weight",
    "k_norm_weights": "self_attn.k_norm.weight",
    "router_weights": "mlp.gate.weight",
}
VARIANTS = ("f32", "bf16_rounded_f32", "int8_per_output_channel")
INTERMEDIATE = MODEL_DIR / "m4.2-03-transformers-f32-intermediate-v1.safetensors"


def bf16_round(values: np.ndarray) -> np.ndarray:
    """Round F32 values to BF16 then represent them as F32 (RNE)."""
    bits = values.astype("<f4", copy=False).view("<u4")
    lsb = (bits >> 16) & 1
    rounded = (bits + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return rounded.view("<f4").copy()


def quantize_per_row(matrix: np.ndarray) -> np.ndarray:
    maximum = np.max(np.abs(matrix), axis=1, keepdims=True)
    scales = maximum / np.float32(127.0)
    scaled = np.divide(matrix, scales, out=np.zeros_like(matrix), where=scales != 0)
    quantized = np.clip(np.rint(scaled), -127, 127).astype(np.int8)
    return quantized.astype(np.float32) * scales


def variant_weight(matrix: np.ndarray, variant: str) -> np.ndarray:
    if variant == "f32":
        return matrix
    if variant == "bf16_rounded_f32":
        return bf16_round(matrix)
    if variant == "int8_per_output_channel":
        if matrix.ndim != 2:
            raise ValueError("INT8 output-channel diagnostic requires a matrix")
        return quantize_per_row(matrix)
    raise ValueError(variant)


def parse_dense_manifest(path: Path) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: item for item in document["tensors"]}


def read_tensor(data_path: Path, record: dict[str, Any], rows: tuple[int, ...] | None = None) -> np.ndarray:
    shape = tuple(record["shape"])
    if rows is None:
        with data_path.open("rb") as source:
            source.seek(record["offset"])
            payload = source.read(record["byte_length"])
        require(len(payload) == record["byte_length"], "truncated dense tensor")
        return np.frombuffer(payload, dtype="<f4").copy().reshape(shape)
    row_width = int(np.prod(shape[1:]))
    output = np.empty((len(rows), row_width), dtype=np.float32)
    with data_path.open("rb") as source:
        for index, row in enumerate(rows):
            source.seek(record["offset"] + row * row_width * 4)
            payload = source.read(row_width * 4)
            require(len(payload) == row_width * 4, "truncated dense row")
            output[index] = np.frombuffer(payload, dtype="<f4")
    return output.reshape((len(rows),) + shape[1:])


def max_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / denominator) if denominator else 1.0


def topk_ids(logits: np.ndarray, k: int = 8) -> list[int]:
    order = sorted(range(len(logits)), key=lambda index: (-float(logits[index]), index))
    return order[:k]


def storage_impact() -> dict[str, Any]:
    shapes = {
        "embedding_weights": (151936, 2048), "lm_head_weights": (151936, 2048),
        "attention_q_projection": (48, 4096, 2048), "attention_k_projection": (48, 512, 2048),
        "attention_v_projection": (48, 512, 2048), "attention_o_projection": (48, 2048, 4096),
        "input_rmsnorm_weights": (48, 2048), "post_attention_rmsnorm_weights": (48, 2048),
        "q_norm_weights": (48, 128), "k_norm_weights": (48, 128),
        "router_weights": (48, 128, 2048), "final_rmsnorm_weights": (2048,),
    }
    records: dict[str, Any] = {}
    for group, shape in shapes.items():
        elements = int(np.prod(shape))
        f32 = elements * 4
        bf16 = elements * 2
        if len(shape) >= 3:
            rows = int(np.prod(shape[:-1]))
        elif group in ("embedding_weights", "lm_head_weights"):
            rows = shape[0]
        elif len(shape) == 2:
            rows = shape[0]
        else:
            rows = None
        int8 = elements + (rows * 4 if rows is not None and len(shape) > 1 else 0)
        records[group] = {
            "f32_bytes": f32, "bf16_bytes": bf16,
            "int8_per_output_channel_bytes": int8 if rows is not None else None,
            "bf16_saved_bytes": f32 - bf16,
            "int8_saved_bytes": f32 - int8 if rows is not None else None,
            "bf16_saved_percent": (f32 - bf16) / f32 * 100.0,
            "int8_saved_percent": (f32 - int8) / f32 * 100.0 if rows is not None else None,
        }
    return {"groups": records, "dense_artifact_bytes": 6164373504}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = parse_dense_manifest(args.dense_manifest)
    intermediate = load_file(str(args.intermediate))
    vectors = [
        intermediate["layer0_token0_position0_expert62_expert_input"].astype(np.float32),
        intermediate["layer1_token0_position0_expert68_expert_input"].astype(np.float32),
        intermediate["layer24_token1_position0_expert85_expert_input"].astype(np.float32),
        intermediate["layer47_token0_position0_expert54_expert_input"].astype(np.float32),
    ]
    vectors = [value.reshape(-1) for value in vectors]
    records: list[dict[str, Any]] = []
    router_records: list[dict[str, Any]] = []

    def tensor_name(group: str, layer: int | None) -> str:
        if group == "embedding_weights":
            return "model.embed_tokens.weight"
        if group == "lm_head_weights":
            return "lm_head.weight"
        if group == "final_rmsnorm_weights":
            return "model.norm.weight"
        require(layer is not None, "layer required")
        return f"model.layers.{layer}.{GROUP_TO_SUFFIX[group]}"

    for group in GROUPS:
        layer_values = (None,) if group in ("embedding_weights", "lm_head_weights", "final_rmsnorm_weights") else LAYERS
        for layer in layer_values:
            name = tensor_name(group, layer)
            record = manifest[name]
            rows = TOKEN_IDS if group in ("embedding_weights", "lm_head_weights") else None
            matrix = read_tensor(args.dense_payload, record, rows)
            source = matrix.astype(np.float32)
            for variant in VARIANTS:
                applicable = not (variant == "int8_per_output_channel" and source.ndim != 2)
                if not applicable:
                    records.append({"group": group, "layer": layer, "variant": variant, "status": "not_structurally_meaningful"})
                    continue
                reconstructed = variant_weight(source, variant)
                records.append({
                    "group": group, "layer": layer, "variant": variant,
                    "status": "measured", "shape": list(source.shape),
                    "weight_max_error": max_error(reconstructed, source),
                    "weight_mean_absolute_error": float(np.mean(np.abs(reconstructed - source))),
                    "weight_rmse": float(np.sqrt(np.mean(np.square(reconstructed - source)))),
                    "weight_cosine_similarity": cosine(source, reconstructed),
                    "finite": bool(np.isfinite(reconstructed).all()),
                })
                if group == "embedding_weights":
                    records[-1].update({
                        "operation": "row_lookup",
                        "selected_row_output_max_error": max_error(reconstructed, source),
                        "selected_row_output_rmse": float(np.sqrt(np.mean(np.square(reconstructed - source)))),
                    })
                    continue
                if group == "lm_head_weights":
                    lm_input = vectors[0]
                    baseline_logits = source @ lm_input
                    candidate_logits = reconstructed @ lm_input
                    baseline_order = topk_ids(baseline_logits, min(20, len(baseline_logits)))
                    candidate_order = topk_ids(candidate_logits, min(20, len(candidate_logits)))
                    records[-1].update({
                        "operation": "selected_lm_head_rows",
                        "selected_logit_max_error": max_error(candidate_logits, baseline_logits),
                        "selected_logit_rmse": float(np.sqrt(np.mean(np.square(candidate_logits - baseline_logits)))),
                        "sampled_argmax_id": int(baseline_order[0]) if baseline_order else None,
                        "sampled_candidate_argmax_id": int(candidate_order[0]) if candidate_order else None,
                        "sampled_top20_overlap": len(set(baseline_order) & set(candidate_order)) / float(len(baseline_order) or 1),
                        "full_vocabulary_rank_check": "not_run; selected deterministic rows only",
                    })
                    continue
                vector = vectors[LAYERS.index(layer) if layer is not None else 0]
                if group == "attention_o_projection":
                    q_matrix = read_tensor(args.dense_payload, manifest[f"model.layers.{layer}.self_attn.q_proj.weight"], None)
                    vector = (variant_weight(q_matrix, variant) @ vector).astype(np.float32)
                if source.ndim == 2 and source.shape[1] == vector.shape[0]:
                    output = reconstructed @ vector
                    baseline = source @ vector
                    records[-1].update({"operation": "matvec", "local_output_max_error": max_error(output, baseline), "local_output_rmse": float(np.sqrt(np.mean(np.square(output - baseline))))})
                elif source.ndim == 1:
                    output = reconstructed * vector[: source.shape[0]]
                    baseline = source * vector[: source.shape[0]]
                    records[-1].update({"operation": "elementwise", "local_output_max_error": max_error(output, baseline), "local_output_rmse": float(np.sqrt(np.mean(np.square(output - baseline))))})
                if group == "router_weights" and source.ndim == 2:
                    logits = source @ vector
                    candidate = reconstructed @ vector
                    f32_ids = topk_ids(logits)
                    candidate_ids = topk_ids(candidate)
                    kth = sorted(logits, reverse=True)[7]
                    unselected = max(logits[i] for i in range(len(logits)) if i not in f32_ids)
                    margin = float(kth - unselected)
                    error = max_error(candidate, logits)
                    classification = "exact_match_safe" if f32_ids == candidate_ids and margin > 2 * error else ("numerically_ambiguous" if f32_ids == candidate_ids else "true_mismatch")
                    router_records.append({"layer": layer, "variant": variant, "f32_ids": f32_ids, "candidate_ids": candidate_ids, "max_logit_error": error, "kth_selected_logit": float(kth), "highest_unselected_logit": float(unselected), "boundary_margin": margin, "required_safe_margin": 2 * error, "classification": classification})

    document = {
        "schema_version": 1, "task": "M4.3-03", "status": "precision_sensitivity_diagnostic_complete",
        "model": {"model_id": "Qwen/Qwen3-30B-A3B", "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "canonical_root_manifest_sha256": ROOT_MANIFEST_SHA256, "canonical_f32_unchanged": True},
        "variants": list(VARIANTS), "layers": list(LAYERS), "token_ids": list(TOKEN_IDS),
        "groups": list(GROUPS), "records": records, "router_records": router_records,
        "storage_impact": storage_impact(),
        "coverage_note": "same-input local diagnostics; no mixed-precision runtime or complete artifact was created",
        "determinism": {"bf16_rounding": "IEEE nearest-even bit truncation", "int8": "symmetric per output channel, F32 scales, nearest-even, [-127,127]"},
    }
    payload = canonical_json(document)
    atomic_bytes(args.output_json, payload)
    return {"status": "passed", "json_sha256": hashlib.sha256(payload).hexdigest(), "record_count": len(records), "router_record_count": len(router_records)}


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--dense-payload", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, default=INTERMEDIATE)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(arguments)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    try:
        print(json.dumps(evaluate(args), sort_keys=True))
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"precision sensitivity error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
