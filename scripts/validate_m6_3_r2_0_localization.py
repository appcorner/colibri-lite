#!/usr/bin/env python3
"""Validate and classify M6.3-R2.0 localization evidence."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

CONTRACT_SHA256 = "02322820758848b5ebfe3f020db026f1aa5e3dd54ce007fb71202f50e54aceca"
METHOD_SHA256 = "ee2a85a1531956ad790e27c52bfb87cc7014c93dc4140d85576d020e5a45307f"
HARNESS_AMENDMENT_SHA256 = "c3409036b22b85ab176477e662d26c15fa18cede724e57f8ea61ecc3ddc75894"
OBSERVER_V4_SHA256 = "a6cf0d7518e889ddc27234de7a45100a0c8363e31fa0a3eec61dc29feed3a687"
FIXTURES = ("short_english", "short_thai")
VIEWS = {
    "compute_only_preloaded": 25,
    "load_plus_compute": 10,
}
PATHS = ("reference", "candidate")
PAIR_ORDER = (
    ("candidate", "reference"),
    ("reference", "candidate"),
    ("candidate", "reference"),
    ("reference", "candidate"),
    ("candidate", "reference"),
)
HEX = set("0123456789abcdef")

CANDIDATE_LOAD = (
    "seek",
    "packed_value_read",
    "packed_value_decode_validate",
    "scale_read",
    "scale_decode_validate",
)
CANDIDATE_COMPUTE = (
    "gate_packed_projection",
    "up_packed_projection",
    "activation_product",
    "down_packed_projection",
)
REFERENCE_RAW = (
    "cache_lookup_load",
    "f32_payload_decode",
    "f32_gate_up_activation_combined",
    "down_projection",
)
COMMON_RAW = (
    "routing_occurrence_scan",
    "weighted_accumulation",
)
FAMILIES = (
    "load_decode",
    "gate_up_activation",
    "down_projection",
    "routing_accumulation",
    "exclusive_residual",
)


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def is_git_object_id(value: Any) -> bool:
    return isinstance(value, str) and len(value) in (40, 64) and set(value) <= HEX


def validate_event(name: str, event: dict[str, Any]) -> None:
    required = (
        "calls",
        "total_nanos",
        "exclusive_nanos",
        "min_nanos",
        "median_nanos",
        "max_nanos",
        "logical_bytes",
    )
    require(all(key in event for key in required), f"{name}: incomplete timing event")
    require(all(isinstance(event[key], int) and event[key] >= 0 for key in required), f"{name}: invalid timing value")
    require(event["exclusive_nanos"] <= event["total_nanos"], f"{name}: exclusive exceeds total")
    if event["calls"] == 0:
        require(
            event["total_nanos"] == event["exclusive_nanos"] == event["min_nanos"] == event["median_nanos"] == event["max_nanos"] == 0,
            f"{name}: zero-call event has timing",
        )
    else:
        require(event["min_nanos"] <= event["median_nanos"] <= event["max_nanos"], f"{name}: timing distribution order")
        require(event["total_nanos"] >= event["max_nanos"], f"{name}: total below maximum")


def raw_scopes(path: str) -> tuple[str, ...]:
    if path == "candidate":
        return CANDIDATE_LOAD + CANDIDATE_COMPUTE + COMMON_RAW
    return REFERENCE_RAW + COMMON_RAW


def event_for(sample: dict[str, Any], name: str) -> dict[str, int]:
    return sample["stages"].get(
        name,
        {
            "calls": 0,
            "total_nanos": 0,
            "exclusive_nanos": 0,
            "min_nanos": 0,
            "median_nanos": 0,
            "max_nanos": 0,
            "logical_bytes": 0,
        },
    )

def family_sources(path: str) -> dict[str, tuple[str, ...]]:
    if path == "candidate":
        return {
            "load_decode": CANDIDATE_LOAD,
            "gate_up_activation": (
                "gate_packed_projection",
                "up_packed_projection",
                "activation_product",
            ),
            "down_projection": ("down_packed_projection",),
            "routing_accumulation": COMMON_RAW,
        }
    return {
        "load_decode": ("cache_lookup_load", "f32_payload_decode"),
        "gate_up_activation": ("f32_gate_up_activation_combined",),
        "down_projection": ("down_projection",),
        "routing_accumulation": COMMON_RAW,
    }


def interpretable_event_total(sample: dict[str, Any], name: str) -> int:
    event = event_for(sample, name)
    minimum = 20 * sample["timer_noop_median_nanos"]
    if event["calls"] and event["median_nanos"] >= minimum:
        return event["total_nanos"]
    return 0


def effective_families(sample: dict[str, Any]) -> dict[str, int]:
    values = {name: 0 for name in FAMILIES}
    values["exclusive_residual"] = sample["exclusive_residual_nanos"]
    for family, names in family_sources(sample["path"]).items():
        for name in names:
            event = event_for(sample, name)
            interpreted = interpretable_event_total(sample, name)
            if interpreted:
                values[family] += interpreted
            else:
                values["exclusive_residual"] += event["total_nanos"]
    return values


def validate_sample(sample: dict[str, Any], binary: str, host: str) -> None:
    fixture = sample.get("fixture")
    view = sample.get("view")
    path = sample.get("path")
    require(fixture in FIXTURES, "sample fixture")
    require(view in VIEWS, "sample view")
    require(path in PATHS, "sample path")
    require(sample.get("release_binary_sha256") == binary, "sample binary identity")
    require(sample.get("host_id") == host, "sample host identity")
    require(sample.get("timed_iterations") == VIEWS[view], "sample timed iteration count")
    require(sample.get("attempt_ordinal") == 1, "sample automatic retry is prohibited")
    require(is_sha256(sample.get("output_sha256")), "sample output hash")
    require(isinstance(sample.get("timer_noop_median_nanos"), int) and sample["timer_noop_median_nanos"] > 0, "sample timer calibration")
    require(isinstance(sample.get("expert_occurrences"), int) and sample["expert_occurrences"] > 0, "sample expert occurrences")
    require(isinstance(sample.get("unique_expert_loads"), int) and sample["unique_expert_loads"] >= 0, "sample unique expert loads")
    require(isinstance(sample.get("expected_logical_expert_bytes"), int) and sample["expected_logical_expert_bytes"] >= 0, "sample expected logical bytes")
    require(isinstance(sample.get("logical_expert_bytes"), int) and sample["logical_expert_bytes"] >= 0, "sample logical bytes")
    require(sample["logical_expert_bytes"] == sample["expected_logical_expert_bytes"], "sample logical-byte accounting")
    require(isinstance(sample.get("timed_candidate_payload_bytes"), int) and sample["timed_candidate_payload_bytes"] >= 0, "candidate payload bytes")
    require(isinstance(sample.get("timed_f32_expert_load_bytes"), int) and sample["timed_f32_expert_load_bytes"] >= 0, "F32 load bytes")
    if view == "compute_only_preloaded":
        require(sample["unique_expert_loads"] == 0, "compute-only unique expert loads")
        require(sample["timed_candidate_payload_bytes"] == 0, "compute-only candidate payload bytes")
        require(sample["timed_f32_expert_load_bytes"] == 0, "compute-only F32 load bytes")
        require(sample["logical_expert_bytes"] == 0, "compute-only logical expert bytes")
    elif path == "candidate":
        require(sample["unique_expert_loads"] > 0, "candidate load-plus unique expert loads")
        require(sample["logical_expert_bytes"] == sample["timed_candidate_payload_bytes"] > 0, "candidate load-plus bytes")
        require(sample["timed_f32_expert_load_bytes"] == 0, "candidate sample has F32 load bytes")
    else:
        require(sample["unique_expert_loads"] > 0, "reference load-plus unique expert loads")
        require(sample["logical_expert_bytes"] == sample["timed_f32_expert_load_bytes"] > 0, "reference load-plus bytes")
        require(sample["timed_candidate_payload_bytes"] == 0, "reference sample has candidate bytes")

    stages = sample.get("stages")
    require(isinstance(stages, dict), "sample stages")
    allowed = set(raw_scopes(path)) | {"expert_total"}
    require(set(stages) <= allowed, "sample contains unregistered timing scope")
    require(not any("dequant" in name for name in stages), "standalone dequant timing is prohibited")
    require("expert_total" in stages, "sample expert_total stage")
    for name, event in stages.items():
        require(isinstance(event, dict), f"{name}: event object")
        validate_event(name, event)
    parent = stages["expert_total"]
    children = sum(event_for(sample, name)["total_nanos"] for name in raw_scopes(path))
    require(parent["total_nanos"] == parent["exclusive_nanos"] + children, "parent/child timing reconciliation")
    require(sample.get("exclusive_residual_nanos") == parent["exclusive_nanos"], "exclusive residual reporting")

def validate_controls(controls: list[dict[str, Any]], binary: str, host: str) -> dict[str, list[float]]:
    require(len(controls) == 20, "observer controls must contain 20 process pairs")
    grouped: dict[str, list[float]] = {}
    seen: set[tuple[str, str, int]] = set()
    for control in controls:
        fixture = control.get("fixture")
        path = control.get("path")
        pair = control.get("pair")
        require(fixture in FIXTURES and path in PATHS, "observer control identity")
        require(isinstance(pair, int) and 1 <= pair <= 5, "observer control pair")
        key = (fixture, path, pair)
        require(key not in seen, "duplicate observer control pair")
        seen.add(key)
        require(control.get("release_binary_sha256") == binary, "observer binary identity")
        require(control.get("attempt_ordinal") == 1, "observer automatic retry is prohibited")
        require(control.get("host_id") == host, "observer host identity")
        require(control.get("instrumented_output_sha256") == control.get("uninstrumented_output_sha256"), "observer output identity")
        require(is_sha256(control.get("instrumented_output_sha256")), "observer output hash")
        instrumented = control.get("instrumented_total_nanos")
        uninstrumented = control.get("uninstrumented_total_nanos")
        require(isinstance(instrumented, int) and instrumented > 0, "observer instrumented timing")
        require(isinstance(uninstrumented, int) and uninstrumented > 0, "observer uninstrumented timing")
        mini = control.get("mini_pair_overhead_percent")
        require(isinstance(mini, list) and len(mini) == 5, "observer mini-pair coverage")
        require(all(isinstance(value, (int, float)) for value in mini), "observer mini-pair timing")
        overhead = control.get("overhead_percent")
        require(isinstance(overhead, (int, float)), "observer outer-pair overhead")
        require(abs(float(overhead) - statistics.median(mini)) < 1e-9, "observer mini-pair median binding")
        require(float(overhead) <= 10.0, "observer-effect pair exceeds 10 percent")
        grouped.setdefault(f"{fixture}:{path}", []).append(float(overhead))
    require(len(seen) == 20, "observer control matrix")
    for key, values in grouped.items():
        require(len(values) == 5, f"{key}: observer pair count")
        require(statistics.median(values) <= 5.0, f"{key}: observer median exceeds 5 percent")
    return grouped


def sample_index(samples: list[dict[str, Any]]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    index: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for sample in samples:
        key = (sample["fixture"], sample["view"], sample["pair"], sample["path"])
        require(key not in index, "duplicate localization sample")
        index[key] = sample
    return index

def validate_matrix(samples: list[dict[str, Any]], binary: str, host: str) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    require(len(samples) == 40, "localization requires exactly 40 process samples")
    for sample in samples:
        require(isinstance(sample.get("pair"), int) and 1 <= sample["pair"] <= 5, "sample pair")
        require(sample.get("order_index") in (1, 2), "sample order index")
        expected_path = PAIR_ORDER[sample["pair"] - 1][sample["order_index"] - 1]
        require(sample.get("path") == expected_path, "sample frozen pair order")
        validate_sample(sample, binary, host)
    index = sample_index(samples)
    for fixture in FIXTURES:
        for view in VIEWS:
            for pair, order in enumerate(PAIR_ORDER, start=1):
                for path in order:
                    require((fixture, view, pair, path) in index, "incomplete localization matrix")
        for path in PATHS:
            outputs = {
                index[(fixture, view, pair, path)]["output_sha256"]
                for view in VIEWS
                for pair in range(1, 6)
            }
            require(len(outputs) == 1, f"{fixture}/{path}: localization output drift")
    require(len(index) == 40, "localization matrix uniqueness")
    return index


def normalized_total(sample: dict[str, Any]) -> float:
    return sample["stages"]["expert_total"]["total_nanos"] / sample["expert_occurrences"]


def normalized_family(sample: dict[str, Any], family: str) -> float:
    return effective_families(sample)[family] / sample["expert_occurrences"]


def candidate_slower_pairs(index: dict[tuple[str, str, int, str], dict[str, Any]], fixture: str, view: str) -> int:
    count = 0
    for pair in range(1, 6):
        candidate = index[(fixture, view, pair, "candidate")]
        reference = index[(fixture, view, pair, "reference")]
        count += normalized_total(candidate) > normalized_total(reference)
    return count


def median_candidate_share(index: dict[tuple[str, str, int, str], dict[str, Any]], fixture: str, view: str, families: tuple[str, ...]) -> float:
    shares = []
    for pair in range(1, 6):
        sample = index[(fixture, view, pair, "candidate")]
        numerator = sum(effective_families(sample)[family] for family in families)
        denominator = sample["stages"]["expert_total"]["total_nanos"]
        shares.append(100.0 * numerator / denominator)
    return statistics.median(shares)


def median_candidate_packed_projection_share(
    index: dict[tuple[str, str, int, str], dict[str, Any]], fixture: str, view: str
) -> float:
    shares = []
    for pair in range(1, 6):
        sample = index[(fixture, view, pair, "candidate")]
        numerator = sum(
            interpretable_event_total(sample, name)
            for name in (
                "gate_packed_projection",
                "up_packed_projection",
                "down_packed_projection",
            )
        )
        denominator = sample["stages"]["expert_total"]["total_nanos"]
        shares.append(100.0 * numerator / denominator)
    return statistics.median(shares)

def median_family_delta(index: dict[tuple[str, str, int, str], dict[str, Any]], fixture: str, view: str, family: str) -> float:
    deltas = []
    for pair in range(1, 6):
        candidate = index[(fixture, view, pair, "candidate")]
        reference = index[(fixture, view, pair, "reference")]
        deltas.append(normalized_family(candidate, family) - normalized_family(reference, family))
    return statistics.median(deltas)


def classify(index: dict[tuple[str, str, int, str], dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    compute_slower = {
        fixture: candidate_slower_pairs(index, fixture, "compute_only_preloaded")
        for fixture in FIXTURES
    }
    packed_share = {
        fixture: median_candidate_packed_projection_share(
            index, fixture, "compute_only_preloaded"
        )
        for fixture in FIXTURES
    }
    compute_bound = all(compute_slower[fixture] >= 4 and packed_share[fixture] >= 60.0 for fixture in FIXTURES)

    load_slower = {
        fixture: candidate_slower_pairs(index, fixture, "load_plus_compute")
        for fixture in FIXTURES
    }
    load_share = {
        fixture: median_candidate_share(index, fixture, "load_plus_compute", ("load_decode",))
        for fixture in FIXTURES
    }
    load_bound = (not compute_bound) and all(load_slower[fixture] >= 4 and load_share[fixture] >= 50.0 for fixture in FIXTURES)

    routing_share = {
        fixture: median_candidate_share(index, fixture, "compute_only_preloaded", ("routing_accumulation",))
        for fixture in FIXTURES
    }
    routing_delta = {
        fixture: {
            family: median_family_delta(index, fixture, "compute_only_preloaded", family)
            for family in FAMILIES
        }
        for fixture in FIXTURES
    }
    routing_bound = True
    for fixture in FIXTURES:
        positive = max((value for value in routing_delta[fixture].values() if value > 0.0), default=0.0)
        routing = routing_delta[fixture]["routing_accumulation"]
        routing_bound &= routing_share[fixture] >= 30.0 and routing > 0.0 and routing >= positive

    if compute_bound:
        classification = "packed_projection_compute_bound"
    elif load_bound:
        classification = "load_decode_bound"
    elif routing_bound:
        classification = "routing_accumulation_bound"
    else:
        classification = "mixed_or_distributed"
    details = {
        "compute_only_candidate_slower_pairs": compute_slower,
        "packed_projection_share_percent": packed_share,
        "load_plus_candidate_slower_pairs": load_slower,
        "load_decode_share_percent": load_share,
        "routing_accumulation_share_percent": routing_share,
        "compute_only_median_family_delta_nanos_per_occurrence": routing_delta,
    }
    return classification, details


def validate_document(document: dict[str, Any]) -> dict[str, Any]:
    require(document.get("schema") == "m6.3-r2.0-localization-samples-v1", "evidence schema")
    require(document.get("contract_sha256") == CONTRACT_SHA256, "contract identity")
    require(document.get("measurement_method_contract_sha256") == METHOD_SHA256, "measurement method identity")
    require(document.get("localization_harness_amendment_sha256") == HARNESS_AMENDMENT_SHA256, "harness amendment identity")
    require(document.get("prior_observer_v4_controls_sha256") == OBSERVER_V4_SHA256, "observer v4 provenance")
    require(is_sha256(document.get("observer_controls_sha256")), "fresh observer controls identity")
    require(is_sha256(document.get("reference_fixture_record_sha256")), "fixture record hash")
    require(is_sha256(document.get("execution_manifest_sha256")), "execution manifest hash")
    require(is_git_object_id(document.get("instrumented_source_commit")), "instrumented commit")
    binary = document.get("release_binary_sha256")
    require(is_sha256(binary), "release binary hash")
    host = document.get("host_id")
    require(isinstance(host, str) and host, "host identity")
    require(isinstance(document.get("timer_implementation_identity"), str) and document["timer_implementation_identity"], "timer identity")
    require(isinstance(document.get("collector_versions"), dict) and document["collector_versions"], "collector versions")
    require(document.get("observer_controls_passed_before_localization") is True, "observer controls timing boundary")
    controls = document.get("observer_controls")
    require(isinstance(controls, list), "observer controls")
    observer = validate_controls(controls, binary, host)
    samples = document.get("samples")
    require(isinstance(samples, list), "localization samples")
    index = validate_matrix(samples, binary, host)
    classification, details = classify(index)
    return {
        "schema": "m6.3-r2.0-localization-result-v1",
        "status": "valid_localization_set",
        "contract_sha256": CONTRACT_SHA256,
        "sample_count": 40,
        "observer_control_process_samples": 20,
        "classification": classification,
        "classification_details": details,
        "observer_overhead_percent": observer,
        "authorizes_r2_1_hypothesis_design": True,
        "authorizes_r2_1_optimization_implementation": False,
        "authorizes_m6_4": False,
    }


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: validate_m6_3_r2_0_localization.py <samples.json> [result.json]", file=sys.stderr)
        return 2
    try:
        document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        result = validate_document(document)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    if len(sys.argv) == 3:
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        Path(sys.argv[2]).write_bytes(payload.encode("utf-8"))
    print(json.dumps({
        "status": result["status"],
        "classification": result["classification"],
        "authorizes_r2_1_hypothesis_design": True,
        "authorizes_m6_4": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())