from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_CANDIDATE = "cpu-safe-rust-int8-group32-layer0-r1-1a"
EXPECTED_ARTIFACT_SHA256 = "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2"
EXPECTED_CONTRACT_SHA256 = "571c6b745d1bb809eef270e87647e699e911df1905a2ddd1251b08356cc842d1"
EXPECTED_FIXTURES = {
    "short_english": [0, 358],
    "short_thai": [7360, 91],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(record: dict, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    require(record.get("schema") == "m6.3-r1.2-quality-result-v1", "wrong schema", errors)
    require(record.get("schema_version") == 1, "wrong schema version", errors)
    require(record.get("status") == "passed", "result is not passed", errors)
    candidate = record.get("candidate", {})
    require(candidate.get("candidate_id") == EXPECTED_CANDIDATE, "wrong candidate", errors)
    require(candidate.get("group_size") == 32, "wrong group size", errors)
    require(candidate.get("artifact_sha256") == EXPECTED_ARTIFACT_SHA256, "wrong artifact hash", errors)
    contract = record.get("contract", {})
    require(contract.get("sha256") == EXPECTED_CONTRACT_SHA256, "wrong contract hash", errors)
    fixtures = record.get("fixtures", [])
    require(len(fixtures) == 2, "expected two held-out fixtures", errors)
    seen: set[str] = set()
    for fixture in fixtures:
        fixture_id = fixture.get("fixture_id")
        require(fixture_id in EXPECTED_FIXTURES, f"unexpected fixture: {fixture_id}", errors)
        if fixture_id not in EXPECTED_FIXTURES:
            continue
        require(fixture_id not in seen, f"duplicate fixture: {fixture_id}", errors)
        seen.add(fixture_id)
        require(fixture.get("generated_token_ids") == EXPECTED_FIXTURES[fixture_id], f"wrong generated IDs: {fixture_id}", errors)
        require(fixture.get("prompt_top20_exact") is True, f"top-20 mismatch: {fixture_id}", errors)
        require(fixture.get("router_guards_exact") is True, f"router mismatch: {fixture_id}", errors)
        require(fixture.get("repeatability") == "exact", f"repeatability mismatch: {fixture_id}", errors)
        observed = max(float(fixture.get("prompt_fixed_logit_max_abs", float("inf"))), float(fixture.get("prompt_top20_logit_max_abs", float("inf"))))
        allowed = float(fixture.get("allowed_logit_max_abs", -1.0))
        require(0.0 <= observed <= allowed <= 0.050000001, f"logit gate failed: {fixture_id}", errors)
        stage_error = float(fixture.get("layer0_moe_max_abs", float("inf")))
        require(0.0 < stage_error < float("inf"), f"invalid Layer-0 error: {fixture_id}", errors)
    require(seen == set(EXPECTED_FIXTURES), "held-out fixture set mismatch", errors)
    direct = record.get("direct_consumption", {})
    require(direct.get("packed_candidate_path") is True, "packed candidate path not proven", errors)
    require(direct.get("peak_packed_expert_bytes_asserted") == 5_308_416, "wrong packed peak", errors)
    require(direct.get("complete_f32_weight_materializations") == 0, "F32 materialization detected", errors)
    decision = record.get("decision", {})
    require(decision.get("r1_2") == "go", "R1.2 is not GO", errors)
    require(decision.get("authorizes_r1_3") is True, "R1.3 not authorized", errors)
    require(decision.get("authorizes_m6_4") is False, "M6.4 must remain blocked", errors)
    require(decision.get("authorizes_candidate_reselection") is False, "candidate reselection must remain blocked", errors)
    if root is not None:
        for key in ("contract", "reference", "candidate_evidence"):
            evidence = record.get(key, {})
            path = root / evidence.get("path", "")
            require(path.is_file(), f"missing evidence file: {key}", errors)
            if path.is_file():
                require(sha256(path) == evidence.get("sha256"), f"evidence hash mismatch: {key}", errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    errors = validate(record, args.root.resolve())
    if errors:
        for error in errors:
            print(error)
        return 1
    print(json.dumps({"status": "passed", "r1_2": "go", "authorizes_r1_3": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
