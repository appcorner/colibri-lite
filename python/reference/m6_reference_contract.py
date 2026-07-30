"""Validation for the immutable M6 F32 reference-contract inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_PATH = Path("models/qwen3-30b-a3b")
REFERENCE_PATH = MODEL_PATH / "reference-f32-v1-manifest.json"
QUALITY_PATH = MODEL_PATH / "m6.0-04-quality-fixtures-v1.json"
TIER_B_PATH = MODEL_PATH / "m4.3-01-tier-b-transformers-f32-v1.json"


class ReferenceContractError(ValueError):
    """Raised when a frozen M6 reference-contract invariant fails."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_reference_contract(root: Path, reference: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    """Validates cross-document identity, integrity, and bilingual outputs."""
    if reference.get("reference_id") != "reference-f32-v1":
        raise ReferenceContractError("reference manifest identity is not reference-f32-v1")
    if reference.get("freeze_policy", {}).get("status") != "frozen":
        raise ReferenceContractError("reference manifest is not frozen")
    if quality.get("reference_id") != reference["reference_id"]:
        raise ReferenceContractError("quality fixture reference ID does not match reference manifest")
    if quality.get("status") != "frozen":
        raise ReferenceContractError("quality fixtures are not frozen")
    if quality.get("model") != {
        "id": reference["model"]["model_id"],
        "revision": reference["model"]["revision"],
    }:
        raise ReferenceContractError("quality fixture model identity does not match reference manifest")

    validate_integrity_records(root, quality["integrity_records"])
    tier_b = load_json(root / TIER_B_PATH)
    tier_b_by_name = {fixture["name"]: fixture for fixture in tier_b["fixtures"]}
    fixtures = quality["fixtures"]
    if [fixture["fixture_id"] for fixture in fixtures] != ["short_english", "short_thai"]:
        raise ReferenceContractError("quality fixture IDs must be short_english then short_thai")
    for fixture in fixtures:
        source = tier_b_by_name[fixture["fixture_id"]]
        expected = fixture["expected"]
        logits = expected["logits"]
        if fixture["input"] != {"text": source["text"], "token_ids": source["token_ids"]}:
            raise ReferenceContractError(f"input mismatch for {fixture['fixture_id']}")
        if fixture["final_position"] != source["final_position"]:
            raise ReferenceContractError(f"final position mismatch for {fixture['fixture_id']}")
        if expected["final_norm_sha256_f32_le"] != source["final_norm"]["sha256_f32_le"]:
            raise ReferenceContractError(f"final-norm digest mismatch for {fixture['fixture_id']}")
        if expected["guard_router_ids"] != source["guard_router_ids"]:
            raise ReferenceContractError(f"router guards mismatch for {fixture['fixture_id']}")
        for field in ("argmax_logit", "argmax_token_id", "fixed_indices", "fixed_logits", "top1_margin", "top20_token_ids", "vocabulary_size"):
            if logits[field] != source["logits"][field]:
                raise ReferenceContractError(f"{field} mismatch for {fixture['fixture_id']}")
        finite_counts = logits["finite_counts"]
        if finite_counts != {
            "nan": source["logits"]["nan_count"],
            "negative_infinity": source["logits"]["negative_infinity_count"],
            "positive_infinity": source["logits"]["positive_infinity_count"],
        }:
            raise ReferenceContractError(f"finite counts mismatch for {fixture['fixture_id']}")
    return {
        "fixture_ids": [fixture["fixture_id"] for fixture in fixtures],
        "model_revision": quality["model"]["revision"],
        "reference_id": reference["reference_id"],
        "status": "passed",
    }


def validate_integrity_records(root: Path, records: list[dict[str, Any]]) -> None:
    """Checks every declared fixture input is present, sized, and hash-locked."""
    roles: set[str] = set()
    for record in records:
        if record["role"] in roles:
            raise ReferenceContractError(f"duplicate integrity role {record['role']}")
        roles.add(record["role"])
        path = root / record["path"]
        if not path.is_file():
            raise ReferenceContractError(f"missing integrity input {record['path']}")
        if path.stat().st_size != record["bytes"]:
            raise ReferenceContractError(f"size mismatch for {record['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ReferenceContractError(f"SHA-256 mismatch for {record['path']}")
