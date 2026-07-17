#!/usr/bin/env python3
"""Evaluate representative expert INT8 formats without creating an artifact."""

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
INTERMEDIATE_PLAN = MODEL_DIR / "m4.2-03-intermediate-structure-v1.tsv"
INTERMEDIATE_DATA = MODEL_DIR / "m4.2-03-transformers-f32-intermediate-v1.safetensors"
SELECTED_CASES = (
    (0, 0, 0, 62),
    (0, 0, 7, 91),
    (1, 0, 0, 68),
    (1, 0, 7, 127),
    (24, 1, 0, 85),
    (24, 1, 7, 8),
    (47, 0, 0, 54),
    (47, 0, 7, 36),
)
MATRIX_SHAPES = {
    "gate": (768, 2048),
    "up": (768, 2048),
    "down": (2048, 768),
}
FORMATS = {
    "int8_per_tensor": {
        "axis": None,
        "group_size": None,
        "scale_count": lambda rows, cols: 1,
    },
    "int8_per_output_channel": {
        "axis": 0,
        "group_size": None,
        "scale_count": lambda rows, cols: rows,
    },
    "int8_per_input_group_128": {
        "axis": 1,
        "group_size": 128,
        "scale_count": lambda rows, cols: rows * ((cols + 127) // 128),
    },
}


def parse_plan(path: Path) -> dict[tuple[int, int, int, int, str], dict[str, Any]]:
    records: dict[tuple[int, int, int, int, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split("\t")
        if not fields or fields[0] != "projection":
            continue
        layer, token, position, expert = (int(fields[index]) for index in (1, 2, 3, 4))
        role = fields[5]
        records[(layer, token, position, expert, role)] = {
            "artifact_path": fields[12],
            "artifact_payload_offset": int(fields[13]),
            "artifact_projection_offset": int(fields[14]),
            "artifact_projection_length": int(fields[15]),
            "shape": tuple(int(value) for value in fields[6].split(",")),
            "source_name": fields[8],
            "source_shard": int(fields[9]),
            "source_offset": int(fields[10]),
            "source_length": int(fields[11]),
        }
    return records


def read_matrix(artifact_root: Path, record: dict[str, Any]) -> np.ndarray:
    require(record["shape"] in MATRIX_SHAPES.values(), f"unexpected matrix shape {record['shape']}")
    require(record["artifact_projection_length"] == np.prod(record["shape"]) * 4, "F32 range length")
    path = artifact_root / record["artifact_path"]
    with path.open("rb") as source:
        source.seek(record["artifact_payload_offset"] + record["artifact_projection_offset"])
        payload = source.read(record["artifact_projection_length"])
    require(len(payload) == record["artifact_projection_length"], "truncated F32 range")
    return np.frombuffer(payload, dtype="<f4").copy().reshape(record["shape"])


def round_even(values: np.ndarray) -> np.ndarray:
    return np.rint(values).astype(np.int16)


def quantize(matrix: np.ndarray, format_name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = FORMATS[format_name]
    rows, cols = matrix.shape
    if spec["group_size"] is None and spec["axis"] is None:
        max_abs = np.float32(np.max(np.abs(matrix)))
        scales = np.array([max_abs / 127.0 if max_abs else 0.0], dtype=np.float32)
        scaled = matrix / scales[0] if scales[0] else np.zeros_like(matrix)
    elif spec["group_size"] is None:
        max_abs = np.max(np.abs(matrix), axis=1, keepdims=True)
        scales = (max_abs / 127.0).astype(np.float32).reshape(-1)
        scaled = np.divide(matrix, scales[:, None], out=np.zeros_like(matrix), where=scales[:, None] != 0)
    else:
        group = spec["group_size"]
        group_count = (cols + group - 1) // group
        scales = np.zeros((rows, group_count), dtype=np.float32)
        scaled = np.zeros_like(matrix)
        for group_index in range(group_count):
            start = group_index * group
            end = min(cols, start + group)
            maximum = np.max(np.abs(matrix[:, start:end]), axis=1)
            scales[:, group_index] = maximum / 127.0
            np.divide(
                matrix[:, start:end],
                scales[:, group_index, None],
                out=scaled[:, start:end],
                where=scales[:, group_index, None] != 0,
            )
    quantized = np.clip(round_even(scaled), -127, 127).astype(np.int8)
    return quantized, scales


def dequantize(quantized: np.ndarray, scales: np.ndarray, format_name: str, shape: tuple[int, int]) -> np.ndarray:
    spec = FORMATS[format_name]
    if spec["group_size"] is None and spec["axis"] is None:
        return quantized.astype(np.float32) * scales[0]
    if spec["group_size"] is None:
        return quantized.astype(np.float32) * scales[:, None]
    group = spec["group_size"]
    output = np.zeros(shape, dtype=np.float32)
    for group_index, start in enumerate(range(0, shape[1], group)):
        end = min(shape[1], start + group)
        output[:, start:end] = quantized[:, start:end].astype(np.float32) * scales[:, group_index, None]
    return output


def alignment(value: int, boundary: int = 64) -> int:
    return ((value + boundary - 1) // boundary) * boundary


def matrix_metrics(matrix: np.ndarray, quantized: np.ndarray, scales: np.ndarray, reconstructed: np.ndarray) -> dict[str, Any]:
    error = reconstructed - matrix
    denominator = np.abs(matrix)
    meaningful = denominator > 1.0e-12
    cosine = float(np.dot(matrix.reshape(-1), reconstructed.reshape(-1)) / (np.linalg.norm(matrix) * np.linalg.norm(reconstructed)))
    row_worst = np.max(np.abs(error), axis=1)
    return {
        "source_min": float(np.min(matrix)),
        "source_max": float(np.max(matrix)),
        "quantized_min": int(np.min(quantized)),
        "quantized_max": int(np.max(quantized)),
        "maximum_reconstruction_error": float(np.max(np.abs(error))),
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "maximum_relative_error_meaningful": float(np.max(np.abs(error[meaningful] / matrix[meaningful]))) if np.any(meaningful) else 0.0,
        "cosine_similarity": cosine,
        "per_row_worst_case": float(np.max(row_worst)),
        "per_row_worst_case_mean": float(np.mean(row_worst)),
        "scale_min": float(np.min(scales)),
        "scale_max": float(np.max(scales)),
        "scale_mean": float(np.mean(scales)),
        "scale_zero_count": int(np.count_nonzero(scales == 0)),
        "saturation_count": int(np.count_nonzero((quantized == -127) | (quantized == 127))),
        "zero_scale_count": int(np.count_nonzero(scales == 0)),
    }


def serialized_hash(matrix: np.ndarray, quantized: np.ndarray, scales: np.ndarray, format_name: str, role: str) -> str:
    metadata = canonical_json({
        "format": format_name,
        "role": role,
        "shape": list(matrix.shape),
        "source_dtype": "F32",
        "quantized_dtype": "INT8",
        "scale_dtype": "F32",
        "byte_order": "little",
        "alignment": 64,
    })
    payload = metadata + quantized.astype("i1", copy=False).tobytes(order="C") + scales.astype("<f4", copy=False).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def projection_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))


def storage_analysis() -> dict[str, Any]:
    rows = {"gate": 768, "up": 768, "down": 2048}
    cols = {"gate": 2048, "up": 2048, "down": 768}
    experts = 6144
    records = {}
    for name, spec in FORMATS.items():
        projections = {}
        total = 208
        for role in ("gate", "up", "down"):
            payload = rows[role] * cols[role]
            scale_count = spec["scale_count"](rows[role], cols[role])
            scale_bytes = alignment(scale_count * 4)
            projection_total = alignment(payload) + scale_bytes + 48
            projections[role] = {
                "quantized_payload_bytes": payload,
                "scale_count": scale_count,
                "scale_bytes_aligned": scale_bytes,
                "descriptor_bytes": 48,
                "projection_bytes": projection_total,
            }
            total += projection_total
        full = 128 + experts * total
        records[name] = {
            "per_projection": projections,
            "per_expert_bytes": total,
            "file_header_bytes": 128,
            "expert_record_and_projection_metadata_bytes": 208,
            "full_6144_expert_bytes_excluding_external_manifest": full,
            "compression_ratio_vs_f32": 122147678312 / full,
            "cache_capacity_by_binary_gib": {
                str(gib): (gib * (1 << 30)) // total for gib in (1, 2, 4, 8, 16, 24, 32)
            },
            "cache_capacity_by_decimal_gb": {
                str(gb): (gb * 1_000_000_000) // total for gb in (1, 2, 4, 8, 16, 24, 32)
            },
        }
    return {
        "f32_reference_artifact_bytes": 122147678312,
        "expert_count": experts,
        "alignment_bytes": 64,
        "external_manifest_bytes_excluded": True,
        "formats": records,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = parse_plan(args.intermediate_structure)
    intermediate = load_file(str(args.intermediate_data))
    matrix_results = []
    projection_results = []
    serialized_hashes = []
    for layer, token, position, expert in SELECTED_CASES:
        case_prefix = f"layer{layer}_token{token}_position{position}_expert{expert}"
        inputs = intermediate[f"{case_prefix}_expert_input"]
        gate_reference = intermediate[f"{case_prefix}_gate_projection"]
        up_reference = intermediate[f"{case_prefix}_up_projection"]
        activated_reference = intermediate[f"{case_prefix}_activated_gate"]
        product_reference = intermediate[f"{case_prefix}_activated_product"]
        down_reference = intermediate[f"{case_prefix}_down_projection"]
        weighted_reference = intermediate[f"{case_prefix}_weighted_expert_output"]
        routing_weight = float(intermediate[f"{case_prefix}_routing_weight"][0])
        for role in ("gate", "up", "down"):
            record = records[(layer, token, position, expert, role)]
            matrix = read_matrix(args.artifact_root, record)
            for format_name in FORMATS:
                quantized, scales = quantize(matrix, format_name)
                reconstructed = dequantize(quantized, scales, format_name, matrix.shape)
                metric = matrix_metrics(matrix, quantized, scales, reconstructed)
                matrix_results.append({
                    "layer": layer,
                    "token": token,
                    "position": position,
                    "expert": expert,
                    "role": role,
                    "format": format_name,
                    "shape": list(matrix.shape),
                    "artifact_path": record["artifact_path"],
                    "artifact_payload_offset": record["artifact_payload_offset"],
                    "artifact_projection_offset": record["artifact_projection_offset"],
                    "artifact_projection_length": record["artifact_projection_length"],
                    "source_name": record["source_name"],
                    "serialization_sha256": serialized_hash(matrix, quantized, scales, format_name, role),
                    **metric,
                })
                serialized_hashes.append(matrix_results[-1]["serialization_sha256"])

        for format_name in FORMATS:
            quantized_matrices = {}
            for role in ("gate", "up", "down"):
                matrix = read_matrix(args.artifact_root, records[(layer, token, position, expert, role)])
                q, scales = quantize(matrix, format_name)
                quantized_matrices[role] = dequantize(q, scales, format_name, matrix.shape)
            gate = quantized_matrices["gate"] @ inputs
            up = quantized_matrices["up"] @ inputs
            activated = (gate / (1.0 + np.exp(-gate))).astype(np.float32)
            product = activated * up
            down = quantized_matrices["down"] @ product
            weighted = down * routing_weight
            projection_results.append({
                "layer": layer,
                "token": token,
                "position": position,
                "expert": expert,
                "format": format_name,
                "routing_weight": routing_weight,
                "gate_projection_error": projection_error(gate, gate_reference),
                "up_projection_error": projection_error(up, up_reference),
                "activated_gate_error": projection_error(activated, activated_reference),
                "activated_product_error": projection_error(product, product_reference),
                "down_projection_error": projection_error(down, down_reference),
                "final_expert_output_error": projection_error(down, down_reference),
                "weighted_expert_output_error": projection_error(weighted, weighted_reference),
            })

    document = {
        "schema_version": 1,
        "task": "M4.3-02",
        "status": "representative_quantization_experiment_complete",
        "model": {
            "model_id": "Qwen/Qwen3-30B-A3B",
            "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "canonical_root_manifest_sha256": ROOT_MANIFEST_SHA256,
            "artifact_root": "canonical F32 artifact read-only",
        },
        "determinism": {
            "rounding": "IEEE-754 nearest-even via NumPy rint",
            "clamp": "saturate to [-127, 127]; reserve -128",
            "zero_scale": "zero quantized payload and zero reconstructed values",
            "nan_inf": "reject source non-finite values before quantization",
            "orientation": "row-major output-by-input; down is output-by-intermediate",
            "serialization": "canonical metadata JSON, little-endian INT8 payload, little-endian F32 scales, 64-byte alignment",
            "representative_serialization_hash_count": len(serialized_hashes),
            "representative_serialization_hash_digest": hashlib.sha256("".join(sorted(serialized_hashes)).encode("ascii")).hexdigest(),
        },
        "candidates": {
            "int8_per_tensor": {
                "source_dtype": "F32",
                "quantized_dtype": "INT8",
                "scale_dtype": "F32",
                "quantization_axis": "none",
                "group_size": None,
                "scale": "max(abs(weights))/127",
                "rounding": "nearest-even",
            },
            "int8_per_output_channel": {
                "source_dtype": "F32",
                "quantized_dtype": "INT8",
                "scale_dtype": "F32",
                "quantization_axis": 0,
                "group_size": None,
                "scale": "per output row max(abs(row))/127",
                "rounding": "nearest-even",
            },
            "int8_per_input_group_128": {
                "source_dtype": "F32",
                "quantized_dtype": "INT8",
                "scale_dtype": "F32",
                "quantization_axis": 1,
                "group_size": 128,
                "scale": "per output row and contiguous input group max(abs(group))/127",
                "rounding": "nearest-even",
            },
        },
        "representative_cases": [
            {"layer": layer, "token": token, "position": position, "expert": expert}
            for layer, token, position, expert in SELECTED_CASES
        ],
        "matrix_metrics": matrix_results,
        "projection_metrics": projection_results,
        "storage_analysis": storage_analysis(),
        "source_evidence": {
            "intermediate_structure": str(args.intermediate_structure.relative_to(args.repository_root)).replace("\\", "/"),
            "intermediate_data": str(args.intermediate_data.relative_to(args.repository_root)).replace("\\", "/"),
        },
    }
    payload = canonical_json(document)
    atomic_bytes(args.output_json, payload)
    tsv_lines = ["layer\ttoken\tposition\texpert\trole\tformat\tmax_reconstruction_error\tmean_absolute_error\trmse\tcosine_similarity\tsaturation_count\tserialization_sha256"]
    for item in matrix_results:
        tsv_lines.append("\t".join(str(item[key]) for key in ("layer", "token", "position", "expert", "role", "format", "maximum_reconstruction_error", "mean_absolute_error", "rmse", "cosine_similarity", "saturation_count", "serialization_sha256")))
    atomic_bytes(args.output_tsv, ("\n".join(tsv_lines) + "\n").encode("ascii"))
    return {
        "status": "passed",
        "json_sha256": hashlib.sha256(payload).hexdigest(),
        "tsv_sha256": hashlib.sha256(("\n".join(tsv_lines) + "\n").encode("ascii")).hexdigest(),
        "matrix_count": len(matrix_results),
        "projection_case_count": len(projection_results),
    }


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--intermediate-structure", type=Path, default=INTERMEDIATE_PLAN)
    parser.add_argument("--intermediate-data", type=Path, default=INTERMEDIATE_DATA)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    args = parser.parse_args(arguments)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    try:
        result = evaluate(args)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"quantization experiment error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
