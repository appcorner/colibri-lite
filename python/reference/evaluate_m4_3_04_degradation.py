#!/usr/bin/env python3
"""Compare the selected expert INT8 simulation with the frozen F32 path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors.numpy import load_file
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.reference import export_f32_baseline_fixtures as tier_b_ref
from python.reference import export_short_generation_reference as generation_ref
from python.reference.evaluate_m4_3_02_quant_formats import (
    MATRIX_SHAPES,
    SELECTED_CASES,
    dequantize,
    parse_plan,
    quantize,
    read_matrix,
)
from python.reference.export_full_model_router_reference import (
    RouterReferenceError,
    atomic_bytes,
    canonical_json,
    require,
    reference_config,
)
from python.reference.export_layer1_router_reference import (
    dense_names,
    parse_expert_source_plan,
)
from python.reference.export_layer24_router_reference import EXPERT_SHAPE_BY_ROLE, load_dense_layer, source_tensor
from python.reference.export_short_generation_reference import execute_cached_pre_router, final_norm_module, top_summary
from python.reference.validate_full_model_tensor_values import parse_plan as parse_source_plan, read_json, sha256_file


MODEL_DIR = Path("models/qwen3-30b-a3b")
ROOT_MANIFEST_SHA256 = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
FIXTURES = tier_b_ref.FIXTURES
GUARD_LAYERS = (0, 1, 8, 16, 24, 32, 40, 47)
PROMPT = [9707, 11, 1879, 0]
GENERATED_COUNT = 2


def quantized_projection(weight: torch.Tensor) -> torch.Tensor:
    matrix = weight.float()
    require(matrix.ndim == 2, "expert projection rank")
    scales = torch.amax(torch.abs(matrix), dim=1, keepdim=True) / 127.0
    scaled = torch.where(scales != 0, matrix / scales, torch.zeros_like(matrix))
    quantized = torch.clamp(torch.round(scaled), -127, 127)
    return quantized * scales


def execute_experts_int8(
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
    combined = {name: torch.zeros_like(run["post_attention_rmsnorm"].squeeze(0)) for name, run in runs.items()}
    captured = {name: torch.zeros((8, 2048), dtype=torch.float32) for name, run in runs.items()}
    payload_bytes = 0
    for expert in union:
        weights = {}
        for role, shape in EXPERT_SHAPE_BY_ROLE.items():
            record = projections[(layer, expert, role)]
            require(record["shape"] == shape, f"Layer-{layer} expert-{expert} {role} shape")
            weights[role] = quantized_projection(source_tensor(source_root, shards, record))
            payload_bytes += record["length"]
        for name, run in runs.items():
            if expert not in selected[name]:
                continue
            positions, tokens = generation_ref.selected_occurrences(run["selected_expert_ids"], expert)
            current = run["post_attention_rmsnorm"].squeeze(0)[tokens].float()
            gate_up = torch.cat((weights["gate"], weights["up"]), dim=0)
            gate, up = functional.linear(current, gate_up).chunk(2, dim=-1)
            output = functional.linear(functional.silu(gate) * up, weights["down"])
            weighted = output * run["routing_weights"][tokens, positions, None].float()
            combined[name].index_add_(0, tokens, weighted)
            if capture_outputs:
                for index, position in enumerate(positions.tolist()):
                    captured[name][position] = output[index]
    return combined, captured, payload_bytes


def max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.max(torch.abs(actual.float() - expected.float())))


def rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.square(actual.float() - expected.float()))))


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a, b = actual.float().reshape(-1), expected.float().reshape(-1)
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


def rank_summary(baseline: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    base = top_summary(baseline)
    cand = top_summary(candidate)
    base_ids, cand_ids = base["top20_token_ids"], cand["top20_token_ids"]
    base_rank = {token: index for index, token in enumerate(base_ids)}
    cand_rank = {token: index for index, token in enumerate(cand_ids)}
    overlap = set(base_ids) & set(cand_ids)
    displacement = max((abs(base_rank[token] - cand_rank[token]) for token in overlap), default=0)
    logit_error = max_error(candidate, baseline)
    required = 2.0 * logit_error
    margin = float(base["top1_margin"])
    if base["argmax_token_id"] == cand["argmax_token_id"] and margin > required:
        classification = "exact_match_safe"
    elif base["argmax_token_id"] == cand["argmax_token_id"]:
        classification = "numerically_ambiguous"
    else:
        classification = "true_mismatch" if margin > required else "numerically_ambiguous"
    log_p = torch.log_softmax(baseline.float(), dim=0)
    log_q = torch.log_softmax(candidate.float(), dim=0)
    kl = float(torch.sum(torch.exp(log_p) * (log_p - log_q)))
    return {
        "baseline": base, "candidate": cand,
        "maximum_logit_error": logit_error,
        "top20_overlap": len(overlap) / 20.0,
        "maximum_rank_displacement": displacement,
        "logit_kl_baseline_to_candidate": kl,
        "safe_margin": margin,
        "required_safe_margin": required,
        "safe_margin_erosion": margin - float(cand["top1_margin"]),
        "safe_margin_erosion_ratio": (margin - float(cand["top1_margin"])) / margin if margin else 0.0,
        "classification": classification,
    }


def router_summary(baseline: torch.Tensor, candidate: torch.Tensor, baseline_ids: list[int], candidate_ids: list[int]) -> dict[str, Any]:
    base = baseline.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    selected = sorted((float(base[index]) for index in baseline_ids), reverse=True)
    kth = selected[-1]
    unselected = max(float(base[index]) for index in range(len(base)) if index not in baseline_ids)
    margin = kth - unselected
    error = max_error(candidate, baseline)
    required = 2.0 * error
    if baseline_ids == candidate_ids and margin > required:
        classification = "exact_match_safe"
    elif baseline_ids == candidate_ids:
        classification = "numerically_ambiguous"
    else:
        classification = "true_mismatch" if margin > required else "numerically_ambiguous"
    return {
        "f32_ids": baseline_ids, "int8_ids": candidate_ids,
        "router_logit_max_error": error, "kth_selected_logit": kth,
        "highest_unselected_logit": unselected, "boundary_margin": margin,
        "required_safe_margin": required, "classification": classification,
    }


def run_trace_fixture(args: argparse.Namespace, config: Any, source_shards: dict[int, dict[str, Any]], dense_plan: dict[str, dict[str, Any]], projections: dict[tuple[int, int, str], dict[str, Any]], final_norm: torch.nn.Module, lm_head: torch.Tensor, fixture: dict[str, Any], expert_fn: Any) -> tuple[dict[str, Any], int]:
    cache = DynamicCache(config=config)
    hidden: torch.Tensor | None = None
    payload_bytes = 0
    final_guards: dict[int, list[int]] = {}
    layer_trace: dict[int, dict[str, Any]] = {}
    final_position = len(fixture["token_ids"]) - 1
    for position, token in enumerate(fixture["token_ids"]):
        embedding_record = dense_plan["model.embed_tokens.weight"]
        row_record = {**embedding_record, "offset": embedding_record["offset"] + token * 4096, "length": 4096, "shape": [2048]}
        embedding = source_tensor(args.source_root, source_shards, row_record)
        payload_bytes += 4096
        hidden = embedding.float().reshape(1, 1, 2048)
        for layer in range(48):
            incoming = hidden.reshape(-1).float().clone()
            loaded, dense_bytes = load_dense_layer(args.source_root, source_shards, dense_plan, layer)
            payload_bytes += dense_bytes
            run = execute_cached_pre_router(config, loaded, hidden, position, cache, torch.float32, layer)
            run["dtype"] = torch.float32
            if position == final_position:
                ids = [int(value) for value in run["selected_expert_ids"].reshape(-1).tolist()]
                final_guards[layer] = ids
            combined, _, expert_bytes = expert_fn(args.source_root, source_shards, projections, layer, {"f32": run}, capture_outputs=False)
            payload_bytes += expert_bytes
            block = run["residual_output"].squeeze(0) + combined["f32"]
            hidden = block.unsqueeze(0).float()
            if position == final_position:
                layer_trace[layer] = {
                    "incoming": incoming,
                    "router_logits": run["router_logits"].reshape(-1).float().clone(),
                    "router_ids": final_guards[layer],
                    "moe": combined["f32"].reshape(-1).float().clone(),
                    "block": block.reshape(-1).float().clone(),
                }
            del loaded, run, combined
    require(hidden is not None, f"{fixture['name']} empty fixture")
    with torch.inference_mode():
        normalized = final_norm(hidden).reshape(-1).float().contiguous()
        logits = functional.linear(normalized.reshape(1, -1), lm_head.float()).reshape(-1).float()
    return ({"fixture": fixture, "final_norm": normalized, "logits": logits, "guard_router_ids": final_guards, "layers": layer_trace, "cache_length": cache.get_seq_length()}, payload_bytes)


def run_tier_c(args: argparse.Namespace) -> dict[str, Any]:
    records = parse_plan(args.intermediate_structure)
    intermediate = load_file(str(args.intermediate_data))
    results = []
    for layer, token, position, expert in SELECTED_CASES:
        prefix = f"layer{layer}_token{token}_position{position}_expert{expert}"
        inputs = torch.from_numpy(intermediate[f"{prefix}_expert_input"]).float()
        routing = float(intermediate[f"{prefix}_routing_weight"][0])
        refs = {name: torch.from_numpy(intermediate[f"{prefix}_{name}"]).float() for name in ("gate_projection", "up_projection", "activated_gate", "activated_product", "down_projection", "weighted_expert_output")}
        matrices = {}
        for role in ("gate", "up", "down"):
            matrices[role] = torch.from_numpy(read_matrix(args.artifact_root, records[(layer, token, position, expert, role)])).float()
        qmat = {role: torch.from_numpy(dequantize(*quantize(matrices[role].numpy(), "int8_per_output_channel"), "int8_per_output_channel", tuple(matrices[role].shape))).float() for role in matrices}
        gate = qmat["gate"] @ inputs
        up = qmat["up"] @ inputs
        activated = functional.silu(gate)
        product = activated * up
        down = qmat["down"] @ product
        weighted = down * routing
        errors = {"gate_projection": max_error(gate, refs["gate_projection"]), "up_projection": max_error(up, refs["up_projection"]), "activated_gate": max_error(activated, refs["activated_gate"]), "activated_product": max_error(product, refs["activated_product"]), "down_projection": max_error(down, refs["down_projection"]), "weighted_expert_output": max_error(weighted, refs["weighted_expert_output"])}
        results.append({"layer": layer, "token": token, "position": position, "expert": expert, "routing_weight": routing, "errors": errors, "finite": bool(torch.isfinite(weighted).all()), "aggregation_scope": "selected_occurrence_only; full aggregation covered by Tier B"})
    maxima = {key: max(item["errors"][key] for item in results) for key in results[0]["errors"]}
    gates = {"gate_projection": 0.04, "up_projection": 0.06, "activated_gate": 0.05, "activated_product": 0.30, "down_projection": 0.40, "weighted_expert_output": 0.13}
    return {"cases": results, "maximum_error_by_checkpoint": maxima, "provisional_gates": gates, "all_gates_pass": all(maxima[key] <= gates[key] for key in gates)}


def run_tier_b(args: argparse.Namespace, config: Any, source_shards: dict[int, dict[str, Any]], dense_plan: dict[str, dict[str, Any]], projections: dict[tuple[int, int, str], dict[str, Any]], final_norm: torch.nn.Module, lm_head: torch.Tensor, expert_fn: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    first_failure = None
    total_bytes = 0
    for fixture in FIXTURES:
        baseline, base_bytes = run_trace_fixture(args, config, source_shards, dense_plan, projections, final_norm, lm_head, fixture, tier_b_ref.execute_experts)
        candidate, candidate_bytes = run_trace_fixture(args, config, source_shards, dense_plan, projections, final_norm, lm_head, fixture, expert_fn)
        total_bytes += base_bytes + candidate_bytes
        layer_results = []
        for layer in GUARD_LAYERS:
            base_layer, cand_layer = baseline["layers"][layer], candidate["layers"][layer]
            router = router_summary(base_layer["router_logits"], cand_layer["router_logits"], base_layer["router_ids"], cand_layer["router_ids"])
            layer_results.append({"layer": layer, "incoming_hidden_max_error": max_error(cand_layer["incoming"], base_layer["incoming"]), "combined_moe_max_error": max_error(cand_layer["moe"], base_layer["moe"]), "final_block_max_error": max_error(cand_layer["block"], base_layer["block"]), **router})
            if first_failure is None and router["classification"] == "true_mismatch":
                first_failure = {"fixture": fixture["name"], "layer": layer, "checkpoint": "router", "diagnosis": "safe-margin router-ID mismatch"}
        norm_error = max_error(candidate["final_norm"], baseline["final_norm"])
        final_metrics = rank_summary(baseline["logits"], candidate["logits"])
        final_metrics.update({"final_normalized_hidden_max_error": norm_error, "final_normalized_hidden_rmse": rmse(candidate["final_norm"], baseline["final_norm"]), "final_normalized_hidden_cosine": cosine(candidate["final_norm"], baseline["final_norm"]), "relative_norm_difference": float(torch.abs(torch.linalg.vector_norm(candidate["final_norm"]) - torch.linalg.vector_norm(baseline["final_norm"])) / torch.linalg.vector_norm(baseline["final_norm"])), "nan_count": int(torch.isnan(candidate["logits"]).sum()), "positive_infinity_count": int(torch.isposinf(candidate["logits"]).sum()), "negative_infinity_count": int(torch.isneginf(candidate["logits"]).sum())})
        results.append({"fixture": fixture, "guard_layers": layer_results, "final": final_metrics, "cache_length": candidate["cache_length"]})
        if first_failure is not None:
            break
    return results, {"first_failure": first_failure, "logical_bytes_simulated": total_bytes}


def run_tier_a(args: argparse.Namespace, config: Any, source_shards: dict[int, dict[str, Any]], dense_plan: dict[str, dict[str, Any]], projections: dict[tuple[int, int, str], dict[str, Any]], final_norm_weight: torch.Tensor, lm_head: torch.Tensor, expert_fn: Any) -> dict[str, Any]:
    def execute(expert_impl: Any) -> tuple[list[dict[str, Any]], list[int], int]:
        cache = DynamicCache(config=config)
        inputs = list(PROMPT)
        generated: list[int] = []
        steps = []
        payload = 0
        for step in range(len(PROMPT) + GENERATED_COUNT):
            token = inputs[step]
            row = {**dense_plan["model.embed_tokens.weight"], "offset": dense_plan["model.embed_tokens.weight"]["offset"] + token * 4096, "length": 4096, "shape": [2048]}
            hidden = source_tensor(args.source_root, source_shards, row).float().reshape(1, 1, 2048)
            guards = {}
            for layer in range(48):
                loaded, dense_bytes = load_dense_layer(args.source_root, source_shards, dense_plan, layer)
                payload += dense_bytes
                run = execute_cached_pre_router(config, loaded, hidden, step, cache, torch.float32, layer)
                run["dtype"] = torch.float32
                if layer in (0, 24, 47):
                    guards[str(layer)] = [int(value) for value in run["selected_expert_ids"].reshape(-1).tolist()]
                combined, _, expert_bytes = expert_impl(args.source_root, source_shards, projections, layer, {"f32": run}, capture_outputs=False)
                payload += expert_bytes
                hidden = (run["residual_output"].squeeze(0) + combined["f32"]).unsqueeze(0).float()
                del loaded, run, combined
            norm = final_norm_module(final_norm_weight, torch.float32)
            with torch.inference_mode():
                normalized = norm(hidden).reshape(-1).float()
                logits = functional.linear(normalized.reshape(1, -1), lm_head.float()).reshape(-1).float()
            summary = top_summary(logits)
            steps.append({"step": step, "input_token": token, "position": step, "summary": summary, "logits": logits.clone(), "router_ids": guards, "cache_length": cache.get_seq_length()})
            if step >= len(PROMPT) - 1 and len(generated) < GENERATED_COUNT:
                selected = summary["argmax_token_id"]
                generated.append(selected)
                inputs.append(selected)
        return steps, generated, payload
    baseline, base_ids, base_bytes = execute(tier_b_ref.execute_experts)
    candidate, candidate_ids, candidate_bytes = execute(expert_fn)
    comparisons = []
    for base, cand in zip(baseline, candidate):
        comparisons.append({"step": base["step"], "input_token": base["input_token"], "position": base["position"], "comparison": rank_summary(base["logits"], cand["logits"]), "baseline_argmax": base["summary"]["argmax_token_id"], "candidate_argmax": cand["summary"]["argmax_token_id"], "baseline_top20": base["summary"]["top20_token_ids"], "candidate_top20": cand["summary"]["top20_token_ids"], "router_ids_equal": base["router_ids"] == cand["router_ids"], "cache_length": cand["cache_length"]})
    return {"baseline_generated_ids": base_ids, "int8_generated_ids": candidate_ids, "steps": comparisons, "logical_bytes_simulated": base_bytes + candidate_bytes}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config_contract = read_json(args.contract)
    source_shards, dense_plan, _ = parse_source_plan(args.dense_plan)
    expert_shards, projections = parse_expert_source_plan(args.expert_source_plan)
    require(source_shards == expert_shards, "source shard identities")
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        require(name in dense_plan, f"missing dense tensor {name}")
    for layer in range(48):
        for name in dense_names(layer):
            require(name in dense_plan, f"missing dense tensor {name}")
    torch.manual_seed(config_contract["determinism"]["seed"])
    torch.set_num_threads(config_contract["determinism"]["torch_threads"])
    torch.set_num_interop_threads(config_contract["determinism"]["torch_interop_threads"])
    torch.use_deterministic_algorithms(config_contract["determinism"]["torch_deterministic_algorithms"])
    config = reference_config()
    final_norm_weight = source_tensor(args.source_root, source_shards, dense_plan["model.norm.weight"]).float()
    lm_head = source_tensor(args.source_root, source_shards, dense_plan["lm_head.weight"]).float()
    tier_c = run_tier_c(args)
    output: dict[str, Any] = {"schema_version": 1, "task": "M4.3-04", "status": "tier_c_complete", "model": {"model_id": config_contract["model_id"], "revision": config_contract["revision"], "canonical_root_manifest_sha256": ROOT_MANIFEST_SHA256}, "configuration": {"expert_weights": "symmetric INT8 per output channel", "expert_scales": "F32", "expert_activations": "F32", "expert_accumulation": "F32", "all_non_expert_tensors": "F32", "production_runtime_changed": False}, "tier_c": tier_c, "tier_b": None, "tier_a": None, "first_failure": None, "first_quality_risk": None, "resource_estimates": {"expert_payload_bytes_per_expert": 4733280, "full_6144_expert_bytes": 29081272448, "temporary_dequantization_bytes": 6291456, "cache_capacity_binary_gib": {str(gib): (gib * (1 << 30)) // 4733280 for gib in (1, 2, 4, 8, 16, 24, 32)}, "metrics_label": "simulated logical bytes and estimated temporary memory"}}
    if not tier_c["all_gates_pass"]:
        output["status"] = "stopped_tier_c_gate_failure"
    else:
        tier_b, bmeta = run_tier_b(args, config, source_shards, dense_plan, projections, final_norm_module(final_norm_weight, torch.float32), lm_head, execute_experts_int8)
        output["tier_b"] = tier_b
        output["first_failure"] = bmeta["first_failure"]
        for fixture_result in tier_b:
            final = fixture_result["final"]
            if final["classification"] != "exact_match_safe" or final["top20_overlap"] < 1.0:
                output["first_quality_risk"] = {"fixture": fixture_result["fixture"]["name"], "checkpoint": "final_logits", "classification": final["classification"], "top20_overlap": final["top20_overlap"], "maximum_logit_error": final["maximum_logit_error"]}
                break
        output["status"] = "tier_b_complete" if bmeta["first_failure"] is None else "stopped_safe_router_mismatch"
        if bmeta["first_failure"] is None:
            output["tier_a"] = run_tier_a(args, config, source_shards, dense_plan, projections, final_norm_weight, lm_head, execute_experts_int8)
            output["status"] = "tier_a_complete"
    if output["first_failure"]:
        output["candidate_classification"] = "rejected"
    elif output["first_quality_risk"]:
        output["candidate_classification"] = "quality_risk"
    else:
        output["candidate_classification"] = "semantically_equivalent_fixture_limited"
    payload = canonical_json(output)
    atomic_bytes(args.output_json, payload)
    return {"status": output["status"], "candidate_classification": output["candidate_classification"], "json_sha256": hashlib.sha256(payload).hexdigest(), "first_failure": output["first_failure"]}


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dense-plan", type=Path, required=True)
    parser.add_argument("--expert-source-plan", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--intermediate-structure", type=Path, default=MODEL_DIR / "m4.2-03-intermediate-structure-v1.tsv")
    parser.add_argument("--intermediate-data", type=Path, default=MODEL_DIR / "m4.2-03-transformers-f32-intermediate-v1.safetensors")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(arguments)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    try:
        print(json.dumps(evaluate(args), sort_keys=True))
    except (RouterReferenceError, OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"INT8 degradation evaluation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
