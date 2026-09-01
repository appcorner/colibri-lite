import copy
import unittest

import validate_m6_3_r2_0_localization as validator

SHA = "a" * 64
BINARY = "b" * 64
HOST = "r2-test-host"


def event(total: int, logical_bytes: int = 0) -> dict:
    return {
        "calls": 1,
        "total_nanos": total,
        "exclusive_nanos": total,
        "min_nanos": total,
        "median_nanos": total,
        "max_nanos": total,
        "logical_bytes": logical_bytes,
    }


def parent_event(total: int, residual: int) -> dict:
    value = event(total)
    value["exclusive_nanos"] = residual
    return value


def candidate_stage_values(kind: str, view: str) -> dict[str, int]:
    if kind == "compute":
        values = {"gate_packed_projection": 200, "up_packed_projection": 200, "activation_product": 30, "down_packed_projection": 250, "routing_occurrence_scan": 20, "weighted_accumulation": 20}
    elif kind == "load":
        values = {"gate_packed_projection": 70, "up_packed_projection": 70, "activation_product": 30, "down_packed_projection": 80, "routing_occurrence_scan": 25, "weighted_accumulation": 25}
    elif kind == "routing":
        values = {"gate_packed_projection": 70, "up_packed_projection": 70, "activation_product": 30, "down_packed_projection": 80, "routing_occurrence_scan": 130, "weighted_accumulation": 120}
    else:
        values = {"gate_packed_projection": 80, "up_packed_projection": 80, "activation_product": 30, "down_packed_projection": 90, "routing_occurrence_scan": 40, "weighted_accumulation": 40}
    if view == "load_plus_compute":
        load_total = 700 if kind == "load" else 100
        for name in validator.CANDIDATE_LOAD:
            values[name] = load_total // len(validator.CANDIDATE_LOAD)
    return values

def reference_stage_values(kind: str, view: str) -> dict[str, int]:
    if kind == "routing":
        values = {"gate_projection": 70, "up_projection": 70, "activation_product": 30, "down_projection": 80, "routing_occurrence_scan": 25, "weighted_accumulation": 25}
    else:
        values = {"gate_projection": 120, "up_projection": 120, "activation_product": 30, "down_projection": 160, "routing_occurrence_scan": 20, "weighted_accumulation": 20}
    if kind == "mixed":
        values = {"gate_projection": 80, "up_projection": 80, "activation_product": 30, "down_projection": 90, "routing_occurrence_scan": 40, "weighted_accumulation": 40}
    if view == "load_plus_compute":
        values["cache_lookup_load"] = 50
        values["f32_payload_decode"] = 50
    return values


def make_sample(kind: str, fixture: str, view: str, pair: int, order_index: int, path: str) -> dict:
    values = candidate_stage_values(kind, view) if path == "candidate" else reference_stage_values(kind, view)
    residual = 80 if path == "candidate" else 30
    if kind == "routing" and path == "candidate":
        residual = 100
    if kind == "mixed":
        residual = 40
    stages = {name: event(total) for name, total in values.items()}
    total = sum(values.values()) + residual
    stages["expert_total"] = parent_event(total, residual)
    logical = 0
    candidate_bytes = 0
    f32_bytes = 0
    if view == "load_plus_compute":
        if path == "candidate":
            logical = candidate_bytes = 1000
        else:
            logical = f32_bytes = 2000
    return {
        "fixture": fixture,
        "view": view,
        "pair": pair,
        "order_index": order_index,
        "path": path,
        "attempt_ordinal": 1,
        "release_binary_sha256": BINARY,
        "host_id": HOST,
        "timed_iterations": validator.VIEWS[view],
        "output_sha256": ("c" if path == "candidate" else "f") * 64,
        "timer_noop_median_nanos": 1,
        "expert_occurrences": 10,
        "unique_expert_loads": 0 if view == "compute_only_preloaded" else 8,
        "expected_logical_expert_bytes": logical,
        "logical_expert_bytes": logical,
        "timed_candidate_payload_bytes": candidate_bytes,
        "timed_f32_expert_load_bytes": f32_bytes,
        "exclusive_residual_nanos": residual,
        "stages": stages,
    }

def make_controls() -> list[dict]:
    controls = []
    for fixture in validator.FIXTURES:
        for path in validator.PATHS:
            for pair in range(1, 6):
                output = ("c" if path == "candidate" else "f") * 64
                controls.append({
                    "fixture": fixture,
                    "path": path,
                    "pair": pair,
                    "attempt_ordinal": 1,
                    "release_binary_sha256": BINARY,
                    "host_id": HOST,
                    "instrumented_output_sha256": output,
                    "uninstrumented_output_sha256": output,
                    "instrumented_total_nanos": 101,
                    "uninstrumented_total_nanos": 100,
                })
    return controls


def make_document(kind: str = "compute") -> dict:
    samples = []
    for fixture in validator.FIXTURES:
        for view in validator.VIEWS:
            for pair, order in enumerate(validator.PAIR_ORDER, start=1):
                for order_index, path in enumerate(order, start=1):
                    samples.append(make_sample(kind, fixture, view, pair, order_index, path))
    return {
        "schema": "m6.3-r2.0-localization-samples-v1",
        "contract_sha256": validator.CONTRACT_SHA256,
        "reference_fixture_record_sha256": SHA,
        "execution_manifest_sha256": "d" * 64,
        "instrumented_source_commit": "e" * 64,
        "release_binary_sha256": BINARY,
        "host_id": HOST,
        "timer_implementation_identity": "std::time::Instant+r2_localization-v1",
        "collector_versions": {"r2_localization": "v1"},
        "observer_controls_passed_before_localization": True,
        "observer_controls": make_controls(),
        "samples": samples,
    }

class LocalizationValidatorTests(unittest.TestCase):
    def assert_classification(self, kind: str, expected: str) -> None:
        result = validator.validate_document(make_document(kind))
        self.assertEqual(result["classification"], expected)
        self.assertTrue(result["authorizes_r2_1_hypothesis_design"])
        self.assertFalse(result["authorizes_r2_1_optimization_implementation"])
        self.assertFalse(result["authorizes_m6_4"])

    def test_compute_bound_classification(self) -> None:
        self.assert_classification("compute", "packed_projection_compute_bound")

    def test_load_decode_bound_classification(self) -> None:
        self.assert_classification("load", "load_decode_bound")

    def test_routing_accumulation_bound_classification(self) -> None:
        self.assert_classification("routing", "routing_accumulation_bound")

    def test_mixed_fallback_classification(self) -> None:
        self.assert_classification("mixed", "mixed_or_distributed")

    def test_rejects_wrong_contract(self) -> None:
        document = make_document()
        document["contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(validator.ValidationError, "contract identity"):
            validator.validate_document(document)
    def test_rejects_observer_pair_over_ten_percent(self) -> None:
        document = make_document()
        document["observer_controls"][0]["instrumented_total_nanos"] = 111
        with self.assertRaisesRegex(validator.ValidationError, "exceeds 10 percent"):
            validator.validate_document(document)

    def test_rejects_observer_median_over_five_percent(self) -> None:
        document = make_document()
        for control in document["observer_controls"][:5]:
            control["instrumented_total_nanos"] = 106
        with self.assertRaisesRegex(validator.ValidationError, "median exceeds 5 percent"):
            validator.validate_document(document)

    def test_rejects_observer_output_mismatch(self) -> None:
        document = make_document()
        document["observer_controls"][0]["instrumented_output_sha256"] = "1" * 64
        with self.assertRaisesRegex(validator.ValidationError, "observer output identity"):
            validator.validate_document(document)

    def test_rejects_pair_order_change(self) -> None:
        document = make_document()
        document["samples"][0]["path"] = "reference"
        with self.assertRaisesRegex(validator.ValidationError, "frozen pair order"):
            validator.validate_document(document)

    def test_rejects_automatic_retry(self) -> None:
        document = make_document()
        document["samples"][0]["attempt_ordinal"] = 2
        with self.assertRaisesRegex(validator.ValidationError, "automatic retry"):
            validator.validate_document(document)
    def test_rejects_compute_only_payload_read(self) -> None:
        document = make_document()
        sample = document["samples"][0]
        sample["timed_candidate_payload_bytes"] = 1
        sample["logical_expert_bytes"] = 1
        sample["expected_logical_expert_bytes"] = 1
        with self.assertRaisesRegex(validator.ValidationError, "compute-only candidate payload"):
            validator.validate_document(document)

    def test_rejects_parent_child_mismatch(self) -> None:
        document = make_document()
        document["samples"][0]["stages"]["expert_total"]["total_nanos"] += 1
        with self.assertRaisesRegex(validator.ValidationError, "parent/child"):
            validator.validate_document(document)

    def test_rejects_standalone_dequant_scope(self) -> None:
        document = make_document()
        sample = document["samples"][0]
        sample["stages"]["dequant"] = event(20)
        sample["stages"]["expert_total"]["total_nanos"] += 20
        with self.assertRaisesRegex(validator.ValidationError, "unregistered timing scope|dequant"):
            validator.validate_document(document)


if __name__ == "__main__":
    unittest.main()