"""Schema and evidence checks for M4.3-02 candidate format definition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/qwen3-30b-a3b"


def read_json(name: str) -> dict:
    return json.loads((MODEL / name).read_text(encoding="utf-8"))


class QuantizationDefinitionTests(unittest.TestCase):
    def test_evidence_hashes_and_counts_are_frozen(self) -> None:
        evidence_path = MODEL / "m4.3-02-quantization-evidence-v1.json"
        summary_path = MODEL / "m4.3-02-quantization-matrix-summary-v1.tsv"
        self.assertEqual(
            hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "fe8b7d06d013227952f6387969d705c433913f4c8a95db56f7abda1324f5ddf1",
        )
        self.assertEqual(
            hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "b2bb78d52c96fa8dec4c35cb9d80daabb48576a864b2f8f1689845b77cd2208b",
        )
        evidence = read_json("m4.3-02-quantization-evidence-v1.json")
        self.assertEqual(len(evidence["matrix_metrics"]), 72)
        self.assertEqual(len(evidence["projection_metrics"]), 24)
        self.assertEqual(evidence["representative_cases"].__len__(), 8)
        self.assertEqual(
            {item["format"] for item in evidence["matrix_metrics"]},
            {"int8_per_tensor", "int8_per_output_channel", "int8_per_input_group_128"},
        )
        self.assertEqual(evidence["model"]["canonical_root_manifest_sha256"], "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2")

    def test_selected_format_is_explicit_and_additive(self) -> None:
        spec = read_json("m4.3-02-format-spec-v1.json")
        self.assertEqual(spec["selected_format"], "symmetric_int8_per_output_channel_f32_scale")
        self.assertEqual(spec["quantized_dtype"], "INT8")
        self.assertEqual(spec["scale_dtype"], "F32")
        self.assertEqual(spec["quantization_axis"], 0)
        self.assertIsNone(spec["group_size"])
        self.assertIn("nearest-even", spec["rounding"])
        self.assertEqual(spec["clamping"], "saturate to [-127, 127]; -128 is reserved and never emitted")
        self.assertIn("canonical F32", spec["source"]["artifact"])

    def test_artifact_schema_and_runtime_contract_do_not_change_f32(self) -> None:
        schema = read_json("m4.3-02-quantized-expert-artifact-schema-v1.json")
        runtime = read_json("m4.3-02-runtime-kernel-contract-v1.json")
        self.assertTrue(schema["canonical_f32_artifact_unchanged"])
        self.assertEqual(schema["format_id"], "colibri-qwen3-moe-expert-int8-v1")
        self.assertEqual(schema["file_layout"]["projection_order"], ["gate", "up", "down"])
        self.assertTrue(runtime["initial_operation"].startswith("dequantize_complete_projection"))
        self.assertEqual(runtime["accumulation_dtype"], "F32")
        self.assertIn("SIMD", runtime["forbidden_in_initial_implementation"])

    def test_provisional_gates_are_not_f32_contract(self) -> None:
        gates = read_json("m4.3-02-provisional-correctness-gates-v1.json")
        self.assertEqual(gates["status"], "provisional_future_quantized_contract_not_in_f32_registry")
        self.assertIn("m4.2-tolerance-contract-registry-v1.json", gates["contract_decision"])
        self.assertGreater(gates["representative_error_gates"]["weighted_expert_output_max"], 0.0)
        self.assertTrue(
            any(item.startswith("Tier A generated IDs") for item in gates["semantic_gates"])
        )

    def test_storage_candidates_are_ordered_by_measured_tradeoffs(self) -> None:
        evidence = read_json("m4.3-02-quantization-evidence-v1.json")
        formats = evidence["storage_analysis"]["formats"]
        self.assertLess(
            formats["int8_per_output_channel"]["per_expert_bytes"],
            formats["int8_per_input_group_128"]["per_expert_bytes"],
        )
        self.assertGreater(
            formats["int8_per_input_group_128"]["cache_capacity_by_binary_gib"]["1"],
            0,
        )
        self.assertEqual(formats["int8_per_output_channel"]["cache_capacity_by_binary_gib"]["1"], 226)
        self.assertEqual(formats["int8_per_output_channel"]["cache_capacity_by_binary_gib"]["32"], 7259)


if __name__ == "__main__":
    unittest.main()
