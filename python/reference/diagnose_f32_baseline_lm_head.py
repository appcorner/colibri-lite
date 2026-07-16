#!/usr/bin/env python3
"""Diagnose an M4.3-01 LM-head difference with the exact Rust input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.reference.export_full_model_router_reference import (
    RouterReferenceError,
    atomic_bytes,
    canonical_json,
    require,
)
from python.reference.export_layer24_router_reference import source_tensor
from python.reference.validate_full_model_tensor_values import parse_plan, read_json, sha256_file


def raw_f32(path: Path, expected_values: int) -> torch.Tensor:
    payload = path.read_bytes()
    require(len(payload) == expected_values * 4, f"{path.name} byte length")
    return torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()


def selected_reference_logits(fixture: dict[str, Any]) -> dict[int, float]:
    logits = fixture["logits"]
    selected = dict(zip(logits["fixed_indices"], logits["fixed_logits"]))
    selected.update(dict(zip(logits["top20_token_ids"], logits["top20_logits"])))
    return {int(index): float(value) for index, value in selected.items()}


def export(args: argparse.Namespace) -> dict[str, Any]:
    source_shards, dense_plan, _ = parse_plan(args.dense_plan)
    reference = read_json(args.reference_json)
    fixture = next(
        item for item in reference["fixtures"] if item["name"] == args.fixture
    )
    rust_hidden = raw_f32(args.rust_final_norm, 2048)
    rust_logits = raw_f32(args.rust_logits, 151_936)
    lm_head_record = dense_plan["lm_head.weight"]
    require(lm_head_record["shape"] == [151_936, 2048], "LM-head shape")
    lm_head = source_tensor(args.source_root, source_shards, lm_head_record).float()
    with torch.inference_mode():
        same_input_logits = functional.linear(rust_hidden.reshape(1, -1), lm_head).reshape(-1)
    all_difference = (same_input_logits - rust_logits).abs()
    reference_selected = selected_reference_logits(fixture)

    selected_records = []
    for index in sorted(reference_selected):
        normal_reference = reference_selected[index]
        same_input = float(same_input_logits[index])
        rust = float(rust_logits[index])
        selected_records.append(
            {
                "vocabulary_index": index,
                "normal_transformers_f32": normal_reference,
                "same_input_transformers_f32": same_input,
                "rust_f32": rust,
                "incoming_state_effect": abs(normal_reference - same_input),
                "same_input_local_difference": abs(same_input - rust),
                "normal_end_to_end_difference": abs(normal_reference - rust),
            }
        )

    f64_indices = sorted(
        {
            0,
            151_935,
            int(fixture["logits"]["argmax_token_id"]),
            int(fixture["logits"]["top20_token_ids"][1]),
        }
    )
    f64_records = []
    row_bytes = 2048 * 2
    for index in f64_indices:
        row_record = {
            **lm_head_record,
            "offset": lm_head_record["offset"] + index * row_bytes,
            "length": row_bytes,
            "shape": [2048],
        }
        row = source_tensor(args.source_root, source_shards, row_record)
        value_f64 = float(torch.dot(rust_hidden.double(), row.double()))
        transformers_f32 = float(same_input_logits[index])
        rust_f32 = float(rust_logits[index])
        transformers_difference = abs(transformers_f32 - value_f64)
        rust_difference = abs(rust_f32 - value_f64)
        f64_records.append(
            {
                "operation": "selected_lm_head_dot_product_with_rust_final_norm_input",
                "vocabulary_index": index,
                "f64": value_f64,
                "transformers_f32": transformers_f32,
                "rust_f32": rust_f32,
                "transformers_f32_absolute_difference": transformers_difference,
                "rust_f32_absolute_difference": rust_difference,
                "closer_f32_path": (
                    "transformers_f32"
                    if transformers_difference < rust_difference
                    else "rust_f32"
                    if rust_difference < transformers_difference
                    else "equal"
                ),
                "contract_impact": "diagnostic_only_no_change",
            }
        )

    document = {
        "schema_version": 1,
        "task": "M4.3-01",
        "fixture": args.fixture,
        "status": "same_input_lm_head_diagnostic_complete",
        "diagnosis": "local_same_input_difference_is_separated_from_incoming_state_effect",
        "same_input_all_logits": {
            "maximum_absolute_difference": float(all_difference.max()),
            "maximum_index": int(all_difference.argmax()),
            "nan_count": int(torch.isnan(same_input_logits).sum()),
            "positive_infinity_count": int(torch.isposinf(same_input_logits).sum()),
            "negative_infinity_count": int(torch.isneginf(same_input_logits).sum()),
        },
        "selected_normal_vs_same_input": selected_records,
        "f64_diagnostics": f64_records,
        "contract_decision": "diagnostic_only; do not change Rust arithmetic from F64 proximity",
        "inputs": {
            "dense_plan_sha256": sha256_file(args.dense_plan),
            "reference_json_sha256": sha256_file(args.reference_json),
            "rust_final_norm_sha256": sha256_file(args.rust_final_norm),
            "rust_logits_sha256": sha256_file(args.rust_logits),
        },
    }
    payload = canonical_json(document)
    atomic_bytes(args.output_json, payload)
    return {
        "status": "passed",
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "same_input_maximum_absolute_difference": document["same_input_all_logits"][
            "maximum_absolute_difference"
        ],
        "selected_f64_count": len(f64_records),
    }


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dense-plan", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--rust-final-norm", type=Path, required=True)
    parser.add_argument("--rust-logits", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(arguments)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    try:
        result = export(args)
    except (RouterReferenceError, OSError, KeyError, StopIteration, ValueError, RuntimeError) as error:
        print(f"F32 baseline LM-head diagnostic error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
