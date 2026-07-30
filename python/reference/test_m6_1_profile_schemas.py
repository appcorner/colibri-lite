"""Contract checks for the tracked M6.1 profile-schema definitions."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = ROOT / "docs/schemas"


class M61ProfileSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hardware = json.loads((SCHEMA_ROOT / "hardware-profile-v1.schema.json").read_text(encoding="utf-8"))
        self.model = json.loads((SCHEMA_ROOT / "model-profile-v1.schema.json").read_text(encoding="utf-8"))

    def test_hardware_schema_has_planner_required_measurement_contracts(self) -> None:
        self.assertEqual(self.hardware["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(self.hardware["properties"]["schema"]["const"], "colibri-lite-hardware-profile-v1")
        self.assertEqual(self.hardware["properties"]["schema_version"]["const"], 1)
        for field in ("runtime", "host", "measurements", "recommendations"):
            self.assertIn(field, self.hardware["required"])
        self.assertEqual(self.hardware["$defs"]["measurement_status"]["enum"], ["measured", "unavailable", "not_run"])
        benchmark = self.hardware["$defs"]["benchmark"]
        self.assertIn("distribution", benchmark["allOf"][0]["then"]["required"])
        self.assertIn("reason", benchmark["allOf"][1]["then"]["required"])
        measurement_properties = self.hardware["$defs"]["measurements"]["properties"]
        for field in ("cpu_kernels", "ram_bandwidth", "storage", "backends"):
            self.assertIn(field, measurement_properties)

    def test_model_schema_pins_artifact_execution_and_quality_inputs(self) -> None:
        self.assertEqual(self.model["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(self.model["properties"]["schema"]["const"], "colibri-lite-model-profile-v1")
        self.assertEqual(self.model["properties"]["schema_version"]["const"], 1)
        for field in ("model", "artifact", "execution", "quality_reference"):
            self.assertIn(field, self.model["required"])
        artifact = self.model["properties"]["artifact"]["properties"]
        self.assertEqual(artifact["experts"]["$ref"], "#/$defs/expert_component")
        self.assertEqual(self.model["$defs"]["expert_component"]["additionalProperties"], False)
        precision_status = self.model["properties"]["execution"]["properties"]["precision_candidates"]["items"]["properties"]["status"]["enum"]
        self.assertEqual(precision_status, ["accepted_reference", "candidate", "rejected", "insufficient_evidence"])
        self.assertEqual(self.model["properties"]["quality_reference"]["properties"]["reference_id"]["const"], "reference-f32-v1")


if __name__ == "__main__":
    unittest.main()
