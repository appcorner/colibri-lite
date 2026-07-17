#!/usr/bin/env python3
"""Diagnose ordered F32 post-attention RMSNorm across Python, PyTorch, and Rust."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Iterable

import torch


class RmsDiagnosticError(RuntimeError):
    """The focused RMSNorm diagnostic contract was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RmsDiagnosticError(message)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def ulp_key(value: float) -> int:
    raw = bits(value)
    return (~raw & 0xFFFF_FFFF) if raw & 0x8000_0000 else raw | 0x8000_0000


def ulp_difference(first: float, second: float) -> int:
    return abs(ulp_key(first) - ulp_key(second))


def read_f32_file(path: Path, expected_values: int) -> list[float]:
    payload = path.read_bytes()
    require(len(payload) == expected_values * 4, f"unexpected F32 file length: {path.name}")
    return [value[0] for value in struct.iter_unpack("<f", payload)]


def parse_records(path: Path, kind: str) -> tuple[str | None, dict[str, dict[str, Any]]]:
    payload = None
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if fields[0] == "payload":
            payload = fields[1]
        elif fields[0] == "tensor":
            if kind == "runtime":
                name, offset, length, shape = fields[1], fields[2], fields[3], fields[4]
                dtype = "F32"
            else:
                name, dtype, offset, length, shape = fields[1], fields[2], fields[3], fields[4], fields[5]
            records[name] = {
                "dtype": dtype,
                "offset": int(offset),
                "length": int(length),
                "shape": [int(value) for value in shape.split(",")],
            }
    return payload, records


def read_range_f32(path: Path, record: dict[str, Any]) -> list[float]:
    require(record["dtype"] == "F32", "expected F32 tensor")
    with path.open("rb") as source:
        source.seek(record["offset"])
        payload = source.read(record["length"])
    require(len(payload) == record["length"], f"truncated range in {path.name}")
    return [value[0] for value in struct.iter_unpack("<f", payload)]


def ordered_row(row: list[float], weight: list[float], epsilon: float) -> dict[str, Any]:
    total = f32(0.0)
    for value in row:
        total = f32(total + f32(value * value))
    divisor = f32(float(len(row)))
    mean = f32(total / divisor)
    plus = f32(mean + epsilon)
    square_root = f32(math.sqrt(plus))
    reciprocal = f32(1.0 / square_root)
    normalized = [f32(value * reciprocal) for value in row]
    weighted = [f32(value * scale) for value, scale in zip(normalized, weight, strict=True)]
    alternate_weighted = [f32(value * f32(reciprocal * scale)) for value, scale in zip(row, weight, strict=True)]
    return {
        "ordered_sum_of_squares": total,
        "divisor": divisor,
        "mean_square": mean,
        "epsilon": epsilon,
        "mean_plus_epsilon": plus,
        "square_root": square_root,
        "reciprocal_rms": reciprocal,
        "normalized": normalized,
        "weighted": weighted,
        "alternate_x_times_inv_times_weight": alternate_weighted,
    }


def f64_row(row: list[float], weight: list[float], epsilon: float) -> dict[str, Any]:
    total = math.fsum(float(value) * float(value) for value in row)
    mean = total / len(row)
    plus = mean + float(epsilon)
    square_root = math.sqrt(plus)
    reciprocal = 1.0 / square_root
    normalized = [float(value) * reciprocal for value in row]
    weighted = [value * float(scale) for value, scale in zip(normalized, weight, strict=True)]
    return {
        "sum_of_squares": total,
        "mean_square": mean,
        "mean_plus_epsilon": plus,
        "square_root": square_root,
        "reciprocal_rms": reciprocal,
        "normalized": normalized,
        "weighted": weighted,
    }


def error_metrics(first: list[float], second: list[float]) -> dict[str, Any]:
    require(len(first) == len(second), "comparison length mismatch")
    errors = [abs(left - right) for left, right in zip(first, second, strict=True)]
    maximum = max(errors)
    indices = [] if maximum == 0.0 else [index for index, error in enumerate(errors) if error == maximum]
    ulps = [ulp_difference(left, right) for left, right in zip(first, second, strict=True)]
    maximum_ulps = max(ulps)
    return {
        "maximum_absolute_error": maximum,
        "maximum_absolute_error_indices": indices,
        "maximum_ulp_difference": maximum_ulps,
        "maximum_ulp_indices": (
            [] if maximum_ulps == 0 else [index for index, value in enumerate(ulps) if value == maximum_ulps]
        ),
    }


def f64_closeness(first: list[float], second: list[float], high: list[float]) -> dict[str, Any]:
    require(len(first) == len(second) == len(high), "F64 comparison length mismatch")
    first_errors = [abs(value - reference) for value, reference in zip(first, high, strict=True)]
    second_errors = [abs(value - reference) for value, reference in zip(second, high, strict=True)]
    return {
        "ordered_python_and_rust_maximum_absolute_error": max(first_errors),
        "pytorch_same_input_maximum_absolute_error": max(second_errors),
        "ordered_python_and_rust_closer_elements": sum(
            first_error < second_error
            for first_error, second_error in zip(first_errors, second_errors, strict=True)
        ),
        "pytorch_same_input_closer_elements": sum(
            second_error < first_error
            for first_error, second_error in zip(first_errors, second_errors, strict=True)
        ),
        "equal_distance_elements": sum(
            first_error == second_error
            for first_error, second_error in zip(first_errors, second_errors, strict=True)
        ),
    }


def parse_rust_rows(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headings = lines[0].split("\t")
    output = []
    for line in lines[1:]:
        fields = line.split("\t")
        record = dict(zip(headings, fields, strict=True))
        output.append(
            {
                "token": int(record["token"]),
                **{name: float(record[name]) for name in headings[1:7]},
                **{name: int(record[name], 16) for name in headings[7:]},
            }
        )
    return output


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        require(path.read_bytes() == payload, "existing diagnostic output differs")
        return
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    payload_relative, runtime_records = parse_records(args.runtime_plan, "runtime")
    _, checkpoint_records = parse_records(args.checkpoint_plan, "checkpoint")
    require(payload_relative is not None, "runtime payload path is missing")
    payload_path = args.artifact_root / payload_relative
    weight_record = runtime_records["model.layers.0.post_attention_layernorm.weight"]
    require(weight_record["shape"] == [2048], "post-attention weight shape mismatch")
    weight = read_range_f32(payload_path, weight_record)
    rust_residual = read_f32_file(args.rust_residual, 4 * 2048)
    rust_output = read_f32_file(args.rust_output, 4 * 2048)
    checkpoint_payload = args.f32_checkpoints
    torch_original_residual = read_range_f32(checkpoint_payload, checkpoint_records["residual_output"])
    torch_original_output = read_range_f32(checkpoint_payload, checkpoint_records["post_attention_rmsnorm"])
    rust_rows = parse_rust_rows(args.rust_rows)
    epsilon = f32(1.0e-6)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch_input = torch.tensor(rust_residual, dtype=torch.float32).reshape(4, 2048)
    torch_weight = torch.tensor(weight, dtype=torch.float32)
    input_before = torch_input.clone()
    weight_before = torch_weight.clone()
    squared = torch_input.pow(2)
    torch_sum = squared.sum(dim=-1)
    torch_mean = squared.mean(dim=-1)
    torch_plus = torch_mean + epsilon
    torch_sqrt = torch.sqrt(torch_plus)
    torch_reciprocal = torch.rsqrt(torch_plus)
    torch_normalized = torch_input * torch_reciprocal[:, None]
    torch_same_input_output = torch_weight * torch_normalized
    require(torch.equal(torch_input, input_before), "PyTorch RMSNorm mutated the input")
    require(torch.equal(torch_weight, weight_before), "PyTorch RMSNorm mutated the weight")

    ordered_rows = []
    high_precision_rows = []
    for token in range(4):
        row = rust_residual[token * 2048 : (token + 1) * 2048]
        ordered_rows.append(ordered_row(row, weight, epsilon))
        high_precision_rows.append(f64_row(row, weight, epsilon))
    ordered_output = [value for row in ordered_rows for value in row["weighted"]]
    alternate_output = [value for row in ordered_rows for value in row["alternate_x_times_inv_times_weight"]]
    f64_output = [value for row in high_precision_rows for value in row["weighted"]]
    torch_same = torch_same_input_output.flatten().tolist()

    comparisons = {
        "ordered_python_f32_vs_rust_f32": error_metrics(ordered_output, rust_output),
        "pytorch_same_input_f32_vs_ordered_python_f32": error_metrics(torch_same, ordered_output),
        "pytorch_same_input_f32_vs_rust_f32": error_metrics(torch_same, rust_output),
        "pytorch_original_path_f32_vs_rust_f32": error_metrics(torch_original_output, rust_output),
        "pytorch_original_residual_vs_rust_residual": error_metrics(torch_original_residual, rust_residual),
        "alternate_multiplication_order_vs_rust": error_metrics(alternate_output, rust_output),
    }
    high_precision_closeness = f64_closeness(ordered_output, torch_same, f64_output)

    selected_indices = {1852}
    for comparison in comparisons.values():
        if comparison["maximum_absolute_error"] != 0.0:
            selected_indices.update(comparison["maximum_absolute_error_indices"])
    for token in range(4):
        row_start = token * 2048
        row_errors = [
            abs(torch_same[index] - ordered_output[index])
            for index in range(row_start, row_start + 2048)
        ]
        row_maximum = max(row_errors)
        if row_maximum != 0.0:
            selected_indices.update(
                row_start + element
                for element, error in enumerate(row_errors)
                if error == row_maximum
            )
    selected_elements = []
    for flat_index in sorted(selected_indices):
        token, element = divmod(flat_index, 2048)
        ordered = ordered_rows[token]
        high = high_precision_rows[token]
        selected_elements.append(
            {
                "flat_index": flat_index,
                "token": token,
                "element": element,
                "input": rust_residual[flat_index],
                "weight": weight[element],
                "ordered_python_f32": {
                    "normalized_before_weight": ordered["normalized"][element],
                    "final_weighted_output": ordered_output[flat_index],
                    "final_bits": f"0x{bits(ordered_output[flat_index]):08x}",
                },
                "rust_f32": {
                    "final_weighted_output": rust_output[flat_index],
                    "final_bits": f"0x{bits(rust_output[flat_index]):08x}",
                },
                "pytorch_same_input_f32": {
                    "normalized_before_weight": float(torch_normalized[token, element]),
                    "final_weighted_output": torch_same[flat_index],
                    "final_bits": f"0x{bits(torch_same[flat_index]):08x}",
                },
                "pytorch_original_path_f32": torch_original_output[flat_index],
                "f64_diagnostic": {
                    "normalized_before_weight": high["normalized"][element],
                    "final_weighted_output": f64_output[flat_index],
                },
                "ulp_differences": {
                    "ordered_vs_rust": ulp_difference(ordered_output[flat_index], rust_output[flat_index]),
                    "pytorch_same_input_vs_ordered": ulp_difference(torch_same[flat_index], ordered_output[flat_index]),
                    "pytorch_same_input_vs_rust": ulp_difference(torch_same[flat_index], rust_output[flat_index]),
                },
                "absolute_error_to_f64": {
                    "ordered_python_f32": abs(ordered_output[flat_index] - f64_output[flat_index]),
                    "rust_f32": abs(rust_output[flat_index] - f64_output[flat_index]),
                    "pytorch_same_input_f32": abs(torch_same[flat_index] - f64_output[flat_index]),
                },
            }
        )

    row_diagnostics = []
    scalar_order = [
        ("ordered_sum_of_squares", "sum_bits", torch_sum),
        ("mean_square", "mean_bits", torch_mean),
        ("mean_plus_epsilon", "plus_bits", torch_plus),
        ("square_root", "sqrt_bits", torch_sqrt),
        ("reciprocal_rms", "reciprocal_bits", torch_reciprocal),
    ]
    first_divergence = None
    first_pytorch_divergence = None
    for token in range(4):
        ordered = ordered_rows[token]
        rust = rust_rows[token]
        scalar_comparisons = {}
        for value_name, bit_name, torch_values in scalar_order:
            if bits(ordered[value_name]) != rust[bit_name] and first_divergence is None:
                first_divergence = {"token": token, "intermediate": value_name}
            torch_value = float(torch_values[token])
            if bits(ordered[value_name]) != bits(torch_value) and first_pytorch_divergence is None:
                first_pytorch_divergence = {"token": token, "intermediate": value_name}
            high_name = "sum_of_squares" if value_name == "ordered_sum_of_squares" else value_name
            ordered_high_error = abs(ordered[value_name] - high_precision_rows[token][high_name])
            torch_high_error = abs(torch_value - high_precision_rows[token][high_name])
            scalar_comparisons[value_name] = {
                "ordered_python_vs_rust_ulp_difference": ulp_difference(
                    ordered[value_name], rust[value_name]
                ),
                "pytorch_vs_ordered_absolute_error": abs(torch_value - ordered[value_name]),
                "pytorch_vs_ordered_ulp_difference": ulp_difference(torch_value, ordered[value_name]),
                "closer_to_f64": (
                    "ordered_python_and_rust"
                    if ordered_high_error < torch_high_error
                    else "pytorch"
                    if torch_high_error < ordered_high_error
                    else "equal"
                ),
            }
        high = high_precision_rows[token]
        row_start = token * 2048
        row_errors = [
            abs(torch_same[index] - ordered_output[index])
            for index in range(row_start, row_start + 2048)
        ]
        row_maximum = max(row_errors)
        row_maximum_elements = (
            []
            if row_maximum == 0.0
            else [index for index, error in enumerate(row_errors) if error == row_maximum]
        )
        row_diagnostics.append(
            {
                "token": token,
                "ordered_python_f32": {
                    "divisor": ordered["divisor"],
                    "epsilon": ordered["epsilon"],
                    **{name: ordered[name] for name, _, _ in scalar_order},
                },
                "rust_f32": {
                    "divisor": 2048.0,
                    "epsilon": rust["epsilon"],
                    **{name: rust[name] for name, _, _ in scalar_order},
                },
                "pytorch_f32": {
                    "sum_of_squares": float(torch_sum[token]),
                    "mean_square": float(torch_mean[token]),
                    "epsilon": epsilon,
                    "mean_plus_epsilon": float(torch_plus[token]),
                    "square_root": float(torch_sqrt[token]),
                    "reciprocal_rms": float(torch_reciprocal[token]),
                },
                "f64_diagnostic": {
                    "sum_of_squares": high["sum_of_squares"],
                    "mean_square": high["mean_square"],
                    "epsilon": float(epsilon),
                    "mean_plus_epsilon": high["mean_plus_epsilon"],
                    "square_root": high["square_root"],
                    "reciprocal_rms": high["reciprocal_rms"],
                },
                "scalar_comparisons": scalar_comparisons,
                "pytorch_vs_ordered_output_maximum": {
                    "absolute_error": row_maximum,
                    "elements": row_maximum_elements,
                    "maximum_ulp_difference": max(
                        ulp_difference(
                            torch_same[row_start + element], ordered_output[row_start + element]
                        )
                        for element in range(2048)
                    ),
                },
            }
        )
    if first_divergence is None:
        for index, (ordered, rust) in enumerate(zip(ordered_output, rust_output, strict=True)):
            if bits(ordered) != bits(rust):
                first_divergence = {"token": index // 2048, "element": index % 2048, "intermediate": "final_weighted_output"}
                break

    document = {
        "schema_version": 1,
        "status": "passed" if first_divergence is None else "ordered_python_rust_divergence",
        "layer": 0,
        "shape": [4, 2048],
        "stride": [2048, 1],
        "epsilon": epsilon,
        "divisor": 2048.0,
        "epsilon_placement": "mean(square(input)) + epsilon before square root",
        "runtime_multiplication_order": "(x * reciprocal_rms) * weight",
        "alternate_multiplication_order": "x * (reciprocal_rms * weight)",
        "weight_indexing": "weight[element] shared across tokens",
        "input_mutated": False,
        "weight_mutated": False,
        "pytorch_reduction_contract": "deterministic in this pinned one-thread run; reduction ordering is not a stable PyTorch public contract",
        "comparisons": comparisons,
        "f64_output_closeness": high_precision_closeness,
        "row_diagnostics": row_diagnostics,
        "selected_elements": selected_elements,
        "first_ordered_python_vs_rust_divergence": first_divergence,
        "first_pytorch_vs_ordered_python_divergence": first_pytorch_divergence,
    }
    atomic_json(args.output, document)
    return document


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--runtime-plan", type=Path, required=True)
    parser.add_argument("--f32-checkpoints", type=Path, required=True)
    parser.add_argument("--checkpoint-plan", type=Path, required=True)
    parser.add_argument("--rust-residual", type=Path, required=True)
    parser.add_argument("--rust-output", type=Path, required=True)
    parser.add_argument("--rust-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    for name in ("artifact_root", "runtime_plan", "f32_checkpoints", "checkpoint_plan", "rust_residual", "rust_output", "rust_rows", "output"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        result = diagnose(args)
    except (OSError, KeyError, ValueError, RmsDiagnosticError) as error:
        print(f"RMS diagnostic error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "first_divergence": result["first_ordered_python_vs_rust_divergence"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
