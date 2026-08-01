#!/usr/bin/env python3
"""Strict validator for the M6.3-R1.1 deterministic admission record."""
import argparse
import json
import re
import sys
from pathlib import Path

SHA256 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN = "cpu-safe-rust-int8-group128-layer0-v1"


def require(condition, message, errors):
    if not condition:
        errors.append(message)


def validate(record):
    errors = []
    require(record.get("schema") == "m6.3-r1-1-candidate-admission-v1", "wrong schema", errors)
    require(record.get("schema_version") == 1, "wrong schema version", errors)
    reference = record.get("reference", {})
    require(reference.get("reference_id") == "reference-f32-v1", "reference must be reference-f32-v1", errors)
    require(SHA256.fullmatch(reference.get("canonical_root_manifest_sha256", "")) is not None, "missing/invalid canonical manifest hash", errors)
    require(reference.get("license") == "Apache-2.0", "missing/invalid license", errors)
    gates = record.get("numerical_gates", {})
    require(gates.get("registered_before_final_fixture_execution") is True, "post-hoc threshold registration", errors)
    cap = gates.get("logit_max_abs_cap")
    held_out = record.get("held_out_bilingual_validation_set", {})
    margin = held_out.get("minimum_frozen_top1_margin")
    require(isinstance(cap, (int, float)) and cap > 0, "missing numerical envelope cap", errors)
    require(isinstance(margin, (int, float)) and margin > 0 and cap <= margin / 4, "logit cap exceeds one quarter held-out margin", errors)
    require(held_out.get("fixture_ids") == ["short_english", "short_thai"], "held-out bilingual fixtures are not fixed", errors)
    candidates = record.get("candidates", [])
    require(isinstance(candidates, list) and candidates, "no candidates considered", errors)
    admitted = []
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id", "")
        require(candidate_id != FORBIDDEN, "stopped group-128 candidate was reused", errors)
        layout = candidate.get("layout", {})
        require(layout.get("source_dtype") == "F32", f"{candidate_id}: source dtype must be F32", errors)
        require(layout.get("precision") == "symmetric INT8", f"{candidate_id}: unsupported precision", errors)
        require(layout.get("group_size") in (32, 64), f"{candidate_id}: unregistered group size", errors)
        require(layout.get("tensors") == ["gate_proj", "up_proj", "down_proj"], f"{candidate_id}: malformed tensor names", errors)
        shapes = layout.get("shapes")
        require(isinstance(shapes, list) and len(shapes) == 3 and all(isinstance(shape, str) and shape.startswith("[") for shape in shapes), f"{candidate_id}: malformed shapes", errors)
        artifact = candidate.get("artifact", {})
        require(SHA256.fullmatch(artifact.get("source_sha256", "")) is not None, f"{candidate_id}: invalid source hash", errors)
        direct = candidate.get("direct_consumption", {})
        require(direct.get("packed_values_and_scales_consumed_directly") is True, f"{candidate_id}: no direct packed consumption proof", errors)
        require(direct.get("complete_f32_projection_materializations") == 0, f"{candidate_id}: complete F32 projection expansion", errors)
        require(direct.get("complete_f32_expert_materializations") == 0, f"{candidate_id}: complete F32 expert expansion", errors)
        if candidate.get("status") == "admitted":
            admitted.append(candidate_id)
            require(candidate.get("conversion", {}).get("status") == "passed", f"{candidate_id}: admitted without conversion", errors)
            require(candidate.get("conversion", {}).get("byte_identical_reconversion") is True, f"{candidate_id}: nondeterministic conversion", errors)
            require(SHA256.fullmatch(artifact.get("output_sha256", "")) is not None, f"{candidate_id}: missing output hash", errors)
            require(artifact.get("offsets_valid") is True, f"{candidate_id}: malformed offsets/shapes", errors)
            require(candidate.get("repeated_execution", {}).get("byte_identical") is True, f"{candidate_id}: nondeterministic execution", errors)
            envelope = candidate.get("characterization", {}).get("proposed_logit_max_abs_envelope")
            require(isinstance(envelope, (int, float)) and envelope <= cap and envelope <= margin / 4, f"{candidate_id}: envelope exceeds pre-registered cap", errors)
    selection = record.get("selection", {})
    outcome = selection.get("outcome")
    if outcome == "exactly_one_candidate":
        require(len(admitted) == 1, "selection does not contain exactly one admitted candidate", errors)
        require(selection.get("selected_candidate_id") == admitted[0], "selected candidate does not match admitted candidate", errors)
    elif outcome == "no_candidate_admitted":
        require(not admitted and selection.get("selected_candidate_id") is None, "no_candidate_admitted conflicts with admitted candidate", errors)
    else:
        errors.append("invalid selection outcome")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"invalid record: {exc}", file=sys.stderr)
        return 2
    errors = validate(record)
    if errors:
        print("invalid M6.3-R1.1 admission record:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"valid M6.3-R1.1 admission record: {record['selection']['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
