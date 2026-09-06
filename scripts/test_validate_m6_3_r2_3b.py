import copy
import unittest

from validate_m6_3_r2_3b import (
    CONTRACT_SHA256,
    GROUPS,
    LAYERS,
    REFERENCE_SHA256,
    ROOT_MANIFEST_SHA256,
    SUBSETS,
    logical_expert_bytes,
    packed_artifact_bytes,
    packed_count,
    selected_candidate,
    validate,
)


def candidate(layer: int, group: int, subset: str, error: float = 0.0005) -> dict:
    expert_bytes = logical_expert_bytes(group, subset)
    layer_bytes = expert_bytes * 128
    return {
        "layer": layer,
        "group_size": group,
        "subset_index": SUBSETS.index(subset),
        "subset": subset,
        "packed_count": packed_count(subset),
        "logical_expert_bytes": expert_bytes,
        "logical_layer_bytes": layer_bytes,
        "logical_bytes_saved": 18_874_368 * 128 - layer_bytes,
        "artifact_bytes": packed_artifact_bytes(group),
        "source_sha256": "a" * 64,
        "artifact_sha256": f"{layer:02x}{group:02x}".ljust(64, "b"),
        "same_input_english_max_abs": error,
        "same_input_thai_max_abs": error,
        "same_input_max_abs": error,
        "same_input_pass": error <= 0.001,
        "sequence_english_max_abs": error,
        "sequence_thai_max_abs": error,
        "sequence_max_abs": error,
        "sequence_pass": error <= 0.001,
        "candidate_finite": True,
        "prompt_first_token_exact": True,
        "canonical_f32_control_exact": True,
        "english_trace_sha256": "c" * 64,
        "thai_trace_sha256": "d" * 64,
    }


def valid_document(error: float = 0.0005) -> dict:
    candidates = [candidate(layer, group, subset, error) for layer in LAYERS for group in GROUPS for subset in SUBSETS]
    decisions = []
    for layer in LAYERS:
        rows = [row for row in candidates if row["layer"] == layer]
        selected = selected_candidate(rows)
        decisions.append({
            "layer": layer,
            "selected_policy": "packed_subset",
            "selected_candidate_id": f"layer{layer:02d}-group{selected['group_size']}-{selected['subset']}",
            "group_size": selected["group_size"],
            "subset": selected["subset"],
            "logical_bytes_saved": selected["logical_bytes_saved"],
            "artifact_sha256": selected["artifact_sha256"],
            "candidates_evaluated": 21,
            "candidates_passed": 21,
            "candidates_failed": 0,
            "worst_same_input_max_abs": error,
            "worst_sequence_max_abs": error,
        })
    return {
        "schema": "colibri-m6.3-r2.3b-result-v1",
        "schema_version": 1,
        "identities": {
            "implementation_contract_sha256": CONTRACT_SHA256,
            "r2_3a_reference_sha256": REFERENCE_SHA256,
            "canonical_root_manifest_sha256": ROOT_MANIFEST_SHA256,
            "execution_manifest_sha256": "e" * 64,
        },
        "candidates": candidates,
        "decisions": decisions,
    }


class R23BValidatorTests(unittest.TestCase):
    def test_complete_grid_and_frozen_tie_break_pass(self) -> None:
        document = valid_document()
        self.assertEqual(validate(document), [])
        self.assertEqual(document["decisions"][0]["group_size"], 32)
        self.assertEqual(document["decisions"][0]["subset"], "gate_up_down")

    def test_missing_candidate_is_rejected(self) -> None:
        document = valid_document()
        document["candidates"].pop()
        self.assertTrue(validate(document))

    def test_post_hoc_threshold_flag_is_rejected(self) -> None:
        document = valid_document()
        document["candidates"][0]["same_input_pass"] = False
        self.assertTrue(any("same-input pass mismatch" in error for error in validate(document)))

    def test_nonfinite_candidate_cannot_be_selected(self) -> None:
        document = valid_document()
        document["candidates"][0]["candidate_finite"] = False
        self.assertTrue(any("deterministic selection mismatch" in error for error in validate(document)))

    def test_all_failed_layer_requires_f32_fallback(self) -> None:
        document = valid_document()
        first_layer = LAYERS[0]
        for row in document["candidates"][:21]:
            row["same_input_english_max_abs"] = 0.002
            row["same_input_thai_max_abs"] = 0.002
            row["same_input_max_abs"] = 0.002
            row["same_input_pass"] = False
        document["decisions"][0].update(
            selected_policy="canonical_f32",
            selected_candidate_id=None,
            group_size=None,
            subset=None,
            logical_bytes_saved=0,
            artifact_sha256=None,
            candidates_passed=0,
            candidates_failed=21,
            worst_same_input_max_abs=0.002,
        )
        self.assertEqual(document["decisions"][0]["layer"], first_layer)
        self.assertEqual(validate(document), [])

    def test_malformed_provenance_hash_is_rejected(self) -> None:
        document = copy.deepcopy(valid_document())
        document["candidates"][0]["source_sha256"] = "bad"
        self.assertTrue(any("source hash malformed" in error for error in validate(document)))


if __name__ == "__main__":
    unittest.main()
