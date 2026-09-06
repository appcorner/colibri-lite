#!/usr/bin/env python3
"""Strict validator for the frozen M6.3-R2.3b 45 x 21 result grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CONTRACT_SHA256 = "97dd33b5cdb576d1ebcf9b3b7661f5462c912be40c7485c7e6bd7f8773d7771b"
REFERENCE_SHA256 = "9f8de6841ff883c062c2ed8387dba93631c9de0565943f1de2bb1a27f0fada1e"
ROOT_MANIFEST_SHA256 = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
LAYERS = tuple(range(1, 24)) + tuple(range(25, 47))
GROUPS = (32, 16, 8)
SUBSETS = ("gate_up_down", "gate_up", "gate_down", "up_down", "gate", "up", "down")
LIMIT = 0.001
F32_PROJECTION_BYTES = 6_291_456
F32_EXPERT_BYTES = 18_874_368
EXPERTS_PER_LAYER = 128
HEX = set("0123456789abcdef")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packed_count(subset: str) -> int:
    return {"gate_up_down": 3, "gate_up": 2, "gate_down": 2, "up_down": 2, "gate": 1, "up": 1, "down": 1}[subset]


def packed_projection_bytes(group: int) -> int:
    return 1_572_864 + F32_PROJECTION_BYTES // group


def packed_artifact_bytes(group: int) -> int:
    return packed_projection_bytes(group) * 3 * EXPERTS_PER_LAYER


def logical_expert_bytes(group: int, subset: str) -> int:
    packed = packed_count(subset)
    return packed * packed_projection_bytes(group) + (3 - packed) * F32_PROJECTION_BYTES


def valid_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def passing(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("same_input_pass") is True
        and candidate.get("sequence_pass") is True
        and candidate.get("candidate_finite") is True
        and candidate.get("canonical_f32_control_exact") is True
    )


def selected_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [candidate for candidate in candidates if passing(candidate)]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candidate: (
            candidate["logical_bytes_saved"],
            candidate["group_size"],
            -SUBSETS.index(candidate["subset"]),
        ),
    )


def validate(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(document.get("schema") == "colibri-m6.3-r2.3b-result-v1", "schema mismatch")
    check(document.get("schema_version") == 1, "schema version mismatch")
    identities = document.get("identities", {})
    check(identities.get("implementation_contract_sha256") == CONTRACT_SHA256, "contract identity mismatch")
    check(identities.get("r2_3a_reference_sha256") == REFERENCE_SHA256, "reference identity mismatch")
    check(identities.get("canonical_root_manifest_sha256") == ROOT_MANIFEST_SHA256, "canonical root identity mismatch")
    check(valid_hash(identities.get("execution_manifest_sha256")), "execution manifest hash malformed")
    candidates = document.get("candidates", [])
    decisions = document.get("decisions", [])
    check(isinstance(candidates, list) and len(candidates) == 945, "candidate count must be exactly 945")
    check(isinstance(decisions, list) and len(decisions) == 45, "decision count must be exactly 45")
    if not isinstance(candidates, list) or len(candidates) != 945:
        return errors

    expected_grid = [(layer, group, subset) for layer in LAYERS for group in GROUPS for subset in SUBSETS]
    actual_grid = [(row.get("layer"), row.get("group_size"), row.get("subset")) for row in candidates]
    check(actual_grid == expected_grid, "candidate grid/order differs from frozen 45 x 3 x 7 order")
    by_layer: dict[int, list[dict[str, Any]]] = {layer: [] for layer in LAYERS}
    for index, candidate in enumerate(candidates):
        layer, group, subset = expected_grid[index]
        if candidate.get("layer") in by_layer:
            by_layer[candidate["layer"]].append(candidate)
        check(candidate.get("subset_index") == SUBSETS.index(subset), f"candidate {index} subset index mismatch")
        check(candidate.get("packed_count") == packed_count(subset), f"candidate {index} packed count mismatch")
        expected_expert = logical_expert_bytes(group, subset)
        expected_layer = expected_expert * EXPERTS_PER_LAYER
        check(candidate.get("logical_expert_bytes") == expected_expert, f"candidate {index} logical expert bytes mismatch")
        check(candidate.get("logical_layer_bytes") == expected_layer, f"candidate {index} logical layer bytes mismatch")
        check(candidate.get("logical_bytes_saved") == (F32_EXPERT_BYTES * EXPERTS_PER_LAYER - expected_layer), f"candidate {index} savings mismatch")
        check(candidate.get("artifact_bytes") == packed_artifact_bytes(group), f"candidate {index} artifact bytes mismatch")
        check(valid_hash(candidate.get("source_sha256")), f"candidate {index} source hash malformed")
        check(valid_hash(candidate.get("artifact_sha256")), f"candidate {index} artifact hash malformed")
        for name in ("same_input_english_max_abs", "same_input_thai_max_abs", "same_input_max_abs", "sequence_english_max_abs", "sequence_thai_max_abs", "sequence_max_abs"):
            value = candidate.get(name)
            check(isinstance(value, (int, float)) and value >= 0.0, f"candidate {index} {name} invalid")
        same_values = [candidate.get("same_input_english_max_abs"), candidate.get("same_input_thai_max_abs")]
        sequence_values = [candidate.get("sequence_english_max_abs"), candidate.get("sequence_thai_max_abs")]
        if all(isinstance(value, (int, float)) for value in same_values):
            check(candidate.get("same_input_max_abs") == max(same_values), f"candidate {index} same-input aggregate mismatch")
            check(candidate.get("same_input_pass") is (max(same_values) <= LIMIT), f"candidate {index} same-input pass mismatch")
        if all(isinstance(value, (int, float)) for value in sequence_values):
            check(candidate.get("sequence_max_abs") == max(sequence_values), f"candidate {index} sequence aggregate mismatch")
            check(candidate.get("sequence_pass") is (max(sequence_values) <= LIMIT), f"candidate {index} sequence pass mismatch")
        check(isinstance(candidate.get("candidate_finite"), bool), f"candidate {index} finite flag missing")
        check(isinstance(candidate.get("prompt_first_token_exact"), bool), f"candidate {index} prompt token flag missing")
        check(candidate.get("canonical_f32_control_exact") is True, f"candidate {index} canonical F32 control failed")
        for name in ("english_trace_sha256", "thai_trace_sha256"):
            check(valid_hash(candidate.get(name)), f"candidate {index} {name} malformed")

    if not isinstance(decisions, list) or len(decisions) != 45:
        return errors
    check([decision.get("layer") for decision in decisions] == list(LAYERS), "decision layer order mismatch")
    for decision in decisions:
        layer = decision.get("layer")
        if layer not in by_layer:
            continue
        rows = by_layer[layer]
        expected = selected_candidate(rows)
        check(decision.get("candidates_evaluated") == 21, f"layer {layer} evaluated count mismatch")
        check(decision.get("candidates_passed") == sum(passing(row) for row in rows), f"layer {layer} pass count mismatch")
        check(decision.get("candidates_failed") == 21 - sum(passing(row) for row in rows), f"layer {layer} fail count mismatch")
        check(decision.get("worst_same_input_max_abs") == max(row["same_input_max_abs"] for row in rows), f"layer {layer} same-input maximum mismatch")
        check(decision.get("worst_sequence_max_abs") == max(row["sequence_max_abs"] for row in rows), f"layer {layer} sequence maximum mismatch")
        if expected is None:
            check(decision.get("selected_policy") == "canonical_f32", f"layer {layer} fallback mismatch")
            check(decision.get("selected_candidate_id") is None, f"layer {layer} fallback candidate must be null")
            check(decision.get("logical_bytes_saved") == 0, f"layer {layer} fallback savings must be zero")
        else:
            expected_id = f"layer{layer:02d}-group{expected['group_size']}-{expected['subset']}"
            check(decision.get("selected_policy") == "packed_subset", f"layer {layer} selected policy mismatch")
            check(decision.get("selected_candidate_id") == expected_id, f"layer {layer} deterministic selection mismatch")
            check(decision.get("group_size") == expected["group_size"], f"layer {layer} selected group mismatch")
            check(decision.get("subset") == expected["subset"], f"layer {layer} selected subset mismatch")
            check(decision.get("logical_bytes_saved") == expected["logical_bytes_saved"], f"layer {layer} selected savings mismatch")
            check(decision.get("artifact_sha256") == expected["artifact_sha256"], f"layer {layer} artifact identity mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    document = json.loads(args.result.read_text(encoding="utf-8"))
    errors = validate(document)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(json.dumps({"status": "passed", "candidates": 945, "decisions": 45}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
