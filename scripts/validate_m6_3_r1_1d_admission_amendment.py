import argparse
import hashlib
import json
from pathlib import Path

GROUP32_ID = "cpu-safe-rust-int8-group32-layer0-r1-1a"
GROUP32_SHA256 = "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2"
HISTORICAL_OUTCOME = "no_candidate_admitted"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(record: dict, repo_root: Path) -> list[str]:
    errors: list[str] = []
    require(record.get("schema") == "m6.3-r1.1d-admission-amendment-v1", "wrong schema", errors)
    require(record.get("schema_version") == 1, "wrong schema version", errors)
    require(record.get("status") == "accepted", "amendment is not accepted", errors)

    historical = record.get("historical_admission", {})
    historical_path = repo_root / historical.get("path", "")
    require(historical.get("outcome") == HISTORICAL_OUTCOME, "historical outcome changed", errors)
    require(historical.get("immutable") is True, "historical admission must remain immutable", errors)
    if historical_path.is_file():
        require(sha256_file(historical_path) == historical.get("sha256"), "historical admission hash mismatch", errors)
        original = json.loads(historical_path.read_text(encoding="utf-8"))
        selection = original.get("selection", {})
        require(selection.get("outcome") == HISTORICAL_OUTCOME, "historical record no longer says no_candidate_admitted", errors)
        require(selection.get("selected_candidate_id") is None, "historical record unexpectedly selects a candidate", errors)
    else:
        errors.append("historical admission record is missing")

    review = record.get("r1_1a_final_review", {})
    review_path = repo_root / review.get("path", "")
    if review_path.is_file():
        require(sha256_file(review_path) == review.get("sha256"), "R1.1a final-review hash mismatch", errors)
    else:
        errors.append("R1.1a final-review record is missing")
    require(review.get("characterization_winner") == GROUP32_ID, "wrong characterization winner", errors)

    candidate = record.get("candidate", {})
    require(candidate.get("candidate_id") == GROUP32_ID, "wrong admitted candidate", errors)
    require(candidate.get("group_size") == 32, "wrong group size", errors)
    require(candidate.get("artifact_sha256") == GROUP32_SHA256, "wrong candidate artifact hash", errors)
    require(candidate.get("valid_characterization_runs") == 3, "three valid characterization runs required", errors)
    require(candidate.get("exact_router_ids") is True, "exact router IDs required", errors)
    require(candidate.get("direct_packed_consumption") is True, "direct packed consumption required", errors)
    require(candidate.get("complete_f32_weight_materializations") == 0, "complete F32 materialization is forbidden", errors)
    require(candidate.get("first_divergence") == "layer0.selected_expert_output", "first divergence mismatch", errors)
    envelope = candidate.get("admission_logit_max_abs_envelope")
    observed = candidate.get("fixed_logits_max_abs")
    require(isinstance(envelope, (int, float)) and 0.0 < envelope <= 0.05, "invalid admission logit envelope", errors)
    require(isinstance(observed, (int, float)) and observed <= envelope, "observed fixed-logit error exceeds envelope", errors)

    ranking = record.get("ranking", {})
    require(ranking.get("winner_decided_by") == "fixed_logits_max_abs_ascending", "ranking criterion changed", errors)
    require(ranking.get("group32_fixed_logits_max_abs", float("inf")) < ranking.get("group64_fixed_logits_max_abs", float("-inf")), "group-32 does not win locked ranking", errors)

    decision = record.get("decision", {})
    require(decision.get("outcome") == "exactly_one_candidate_admitted_for_r1_2", "wrong amendment outcome", errors)
    require(decision.get("selected_candidate_id") == GROUP32_ID, "decision candidate mismatch", errors)
    require(decision.get("authorizes_r1_2") is True, "R1.2 must be explicitly authorized", errors)
    require(decision.get("authorizes_r1_3") is False, "R1.3 must remain blocked", errors)
    require(decision.get("authorizes_m6_4") is False, "M6.4 must remain blocked", errors)
    require(decision.get("historical_r1_1_outcome_revised") is False, "historical R1.1 must not be revised", errors)

    gate = record.get("r1_2_gate", {})
    require(gate.get("held_out_fixture_ids") == ["short_english", "short_thai"], "held-out fixture set changed", errors)
    require(gate.get("must_not_reselect_candidate") is True, "R1.2 must not reselect the candidate", errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    errors = validate(record, args.repo_root.resolve())
    if errors:
        for error in errors:
            print(error)
        return 1
    print(json.dumps({"status": "passed", "selected_candidate_id": GROUP32_ID, "authorizes_r1_2": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
