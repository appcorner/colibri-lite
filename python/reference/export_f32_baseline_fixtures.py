#!/usr/bin/env python3
"""Export compact Transformers-F32 evidence for M4.3-01 Tier B fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
from typing import Any, Iterable

import safetensors
import torch
import torch.nn.functional as functional
from tokenizers import Tokenizer
from transformers import DynamicCache
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.reference.export_full_model_router_reference import (
    RouterReferenceError,
    atomic_bytes,
    canonical_json,
    process_peak_working_set,
    reference_config,
    require,
)
from python.reference.export_layer1_router_reference import (
    dense_names,
    parse_expert_source_plan,
)
from python.reference.export_layer24_router_reference import load_dense_layer, source_tensor
from python.reference.export_short_generation_reference import (
    FIXED_VOCABULARY_INDICES,
    execute_cached_pre_router,
    execute_experts,
    final_norm_module,
    top_summary,
)
from python.reference.validate_full_model_tensor_values import parse_plan, read_json, sha256_file


FIXTURES = (
    {"name": "single_low_token", "text": "!", "token_ids": [0], "coverage": ["one_token", "low_token_id"]},
    {"name": "short_english", "text": "Hello world", "token_ids": [9707, 1879], "coverage": ["english"]},
    {"name": "short_thai", "text": "\u0e44\u0e17\u0e22", "token_ids": [125451], "coverage": ["thai", "high_token_id"]},
    {"name": "code_newline", "text": "x=1\n", "token_ids": [87, 28, 16, 198], "coverage": ["code", "newline"]},
    {"name": "repeated_pattern", "text": "ha ha", "token_ids": [4223, 6386], "coverage": ["repeated_text"]},
    {"name": "special_token", "text": "<|endoftext|>", "token_ids": [151643], "coverage": ["special_token", "high_token_id"]},
)
FIXED_HIDDEN_INDICES = (0, 1, 127, 1024, 2047)
GUARD_LAYERS = (0, 24, 47)


def f32_digest(values: torch.Tensor) -> str:
    payload = b"".join(struct.pack("<f", float(value)) for value in values.float().reshape(-1))
    return hashlib.sha256(payload).hexdigest()


def comma(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in values)


def float_comma(values: Iterable[float]) -> str:
    return ",".join(f"{float(value):.17e}" for value in values)


def execute_fixture(
    args: argparse.Namespace,
    config: Any,
    source_shards: dict[int, dict[str, Any]],
    dense_plan: dict[str, dict[str, Any]],
    projections: dict[tuple[int, int, str], dict[str, Any]],
    final_norm: torch.nn.Module,
    lm_head_f32: torch.Tensor,
    fixture: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    cache = DynamicCache(config=config)
    payload_bytes = 0
    final_guards: dict[int, list[int]] = {}
    hidden: torch.Tensor | None = None

    for position, token in enumerate(fixture["token_ids"]):
        embedding_record = dense_plan["model.embed_tokens.weight"]
        row_record = {
            **embedding_record,
            "offset": embedding_record["offset"] + token * 4096,
            "length": 4096,
            "shape": [2048],
        }
        embedding = source_tensor(args.source_root, source_shards, row_record)
        payload_bytes += 4096
        hidden = embedding.float().reshape(1, 1, 2048)
        for layer in range(48):
            loaded, dense_bytes = load_dense_layer(
                args.source_root, source_shards, dense_plan, layer
            )
            payload_bytes += dense_bytes
            run = execute_cached_pre_router(
                config,
                loaded,
                hidden,
                position,
                cache,
                torch.float32,
                layer,
            )
            run["dtype"] = torch.float32
            if position == len(fixture["token_ids"]) - 1 and layer in GUARD_LAYERS:
                final_guards[layer] = [
                    int(value) for value in run["selected_expert_ids"].reshape(-1)
                ]
            combined, _, expert_bytes = execute_experts(
                args.source_root,
                source_shards,
                projections,
                layer,
                {"f32": run},
                capture_outputs=False,
            )
            payload_bytes += expert_bytes
            block = run["residual_output"].squeeze(0) + combined["f32"]
            hidden = block.unsqueeze(0).float()
            del loaded, run, combined

    require(hidden is not None, f"{fixture['name']} empty fixture")
    require(cache.get_seq_length() == len(fixture["token_ids"]), "cache length")
    with torch.inference_mode():
        normalized = final_norm(hidden).reshape(-1).float().contiguous()
        logits = functional.linear(normalized.reshape(1, -1), lm_head_f32).reshape(-1).float()
    summary = top_summary(logits)
    return (
        {
            **fixture,
            "final_position": len(fixture["token_ids"]) - 1,
            "cache_length": cache.get_seq_length(),
            "guard_router_ids": {str(layer): final_guards[layer] for layer in GUARD_LAYERS},
            "final_norm": {
                "shape": [2048],
                "sha256_f32_le": f32_digest(normalized),
                "fixed_indices": list(FIXED_HIDDEN_INDICES),
                "fixed_values": [float(normalized[index]) for index in FIXED_HIDDEN_INDICES],
            },
            "logits": summary,
        },
        payload_bytes,
    )


def reference_tsv(fixtures: list[dict[str, Any]]) -> bytes:
    lines = ["record\tfixture\tvalues"]
    for fixture in fixtures:
        name = fixture["name"]
        lines.append(f"fixture\t{name}\t{comma(fixture['token_ids'])}")
        norm = fixture["final_norm"]
        lines.append(
            f"final_norm\t{name}\t{comma(norm['fixed_indices'])};"
            f"{float_comma(norm['fixed_values'])};{norm['sha256_f32_le']}"
        )
        for layer in GUARD_LAYERS:
            lines.append(
                f"guard_ids_{layer}\t{name}\t{comma(fixture['guard_router_ids'][str(layer)])}"
            )
        logits = fixture["logits"]
        lines.append(
            f"logits\t{name}\t{comma(logits['fixed_indices'])};"
            f"{float_comma(logits['fixed_logits'])};{comma(logits['top20_token_ids'])};"
            f"{float_comma(logits['top20_logits'])};{logits['argmax_token_id']};"
            f"{float(logits['argmax_logit']):.17e};{float(logits['second_highest_logit']):.17e};"
            f"{float(logits['top1_margin']):.17e};{logits['nan_count']};"
            f"{logits['positive_infinity_count']};{logits['negative_infinity_count']}"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def export(args: argparse.Namespace) -> dict[str, Any]:
    contract = read_json(args.contract)
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }
    require(versions == contract["environment"], f"reference environment drift: {versions}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    for fixture in FIXTURES:
        encoded = tokenizer.encode(fixture["text"], add_special_tokens=False)
        require(encoded.ids == fixture["token_ids"], f"{fixture['name']} token IDs")
        require(
            tokenizer.decode(encoded.ids, skip_special_tokens=False) == fixture["text"],
            f"{fixture['name']} tokenizer round trip",
        )

    source_shards, dense_plan, _ = parse_plan(args.dense_plan)
    expert_shards, projections = parse_expert_source_plan(args.expert_source_plan)
    require(source_shards == expert_shards, "source shard identities")
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        require(name in dense_plan, f"missing dense tensor {name}")
    for layer in range(48):
        for name in dense_names(layer):
            require(name in dense_plan, f"missing dense tensor {name}")

    torch.manual_seed(contract["determinism"]["seed"])
    torch.set_num_threads(contract["determinism"]["torch_threads"])
    torch.set_num_interop_threads(contract["determinism"]["torch_interop_threads"])
    torch.use_deterministic_algorithms(contract["determinism"]["torch_deterministic_algorithms"])
    config = reference_config()
    norm_weight = source_tensor(
        args.source_root, source_shards, dense_plan["model.norm.weight"]
    )
    final_norm = final_norm_module(norm_weight, torch.float32)
    lm_head_f32 = source_tensor(
        args.source_root, source_shards, dense_plan["lm_head.weight"]
    ).float()

    fixtures = []
    payload_bytes = dense_plan["model.norm.weight"]["length"] + dense_plan["lm_head.weight"]["length"]
    for fixture in FIXTURES:
        result, fixture_bytes = execute_fixture(
            args,
            config,
            source_shards,
            dense_plan,
            projections,
            final_norm,
            lm_head_f32,
            fixture,
        )
        fixtures.append(result)
        payload_bytes += fixture_bytes

    document = {
        "schema_version": 1,
        "task": "M4.3-01",
        "status": "transformers_f32_tier_b_passed",
        "model_id": contract["model_id"],
        "model_revision": contract["revision"],
        "compute_dtype": "F32 from pinned BF16-derived weights",
        "fixture_count": len(fixtures),
        "processed_token_positions": sum(len(fixture["token_ids"]) for fixture in fixtures),
        "fixed_hidden_indices": list(FIXED_HIDDEN_INDICES),
        "fixed_vocabulary_indices": list(FIXED_VOCABULARY_INDICES),
        "guard_layers": list(GUARD_LAYERS),
        "fixtures": fixtures,
        "environment": versions,
        "inputs": {
            "contract_sha256": sha256_file(args.contract),
            "dense_plan_sha256": sha256_file(args.dense_plan),
            "expert_source_plan_sha256": sha256_file(args.expert_source_plan),
            "tokenizer_sha256": sha256_file(args.tokenizer),
        },
        "source_integrity": "reused pinned M4.1/M4.2 source-manifest and shard-hash evidence",
        "source_payload_bytes_read": payload_bytes,
    }
    json_payload = canonical_json(document)
    tsv_payload = reference_tsv(fixtures)
    atomic_bytes(args.output_json, json_payload)
    atomic_bytes(args.output_tsv, tsv_payload)
    return {
        "status": "passed",
        "fixture_count": len(fixtures),
        "processed_token_positions": document["processed_token_positions"],
        "json_sha256": hashlib.sha256(json_payload).hexdigest(),
        "tsv_sha256": hashlib.sha256(tsv_payload).hexdigest(),
        "source_payload_bytes_read": payload_bytes,
        "peak_process_working_set_bytes": process_peak_working_set(),
    }


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source_root",
        "dense_plan",
        "expert_source_plan",
        "contract",
        "tokenizer",
        "output_json",
        "output_tsv",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args(arguments)
    for name, value in vars(args).items():
        setattr(args, name, value.resolve())
    try:
        result = export(args)
    except (RouterReferenceError, OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"F32 baseline fixture export error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
