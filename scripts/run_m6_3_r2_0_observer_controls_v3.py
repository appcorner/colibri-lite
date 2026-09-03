#!/usr/bin/env python3
"""Run preregistered low-observer M6.3-R2.0 controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

CONTRACT_SHA256 = "02322820758848b5ebfe3f020db026f1aa5e3dd54ce007fb71202f50e54aceca"
METHOD_SHA256 = "b2df0dd3ae10028b669e0d31aaf77794eef8015e1411166a0c5a08d9ab263e94"
FIXTURE_SHA256 = "c0490d2a40a214579e7633fcd4706e64186d183a82af6d176eaa1918d7f056e2"
CANDIDATE_SHA256 = "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2"
CANDIDATE_BYTES = 679_477_248
PRIOR_CONTROLS = [
    "140c219c846af500cc99bd352c3d8b3215571048e3cd02271f414bf84d28d64c",
    "dba8ef0aef549f5fecb020ab626c9691fd6ae3ba79fc007f56828edd0558b597",
]
PAIR_ORDER = (
    ("enabled", "disabled"),
    ("disabled", "enabled"),
    ("enabled", "disabled"),
    ("disabled", "enabled"),
    ("enabled", "disabled"),
)
TEST_FILTER = "full_model_validation_tests::r2_benchmark_tests::m6_3_r2_0_observer_pair_sample"


class ControlError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
def validate_inputs(args: argparse.Namespace) -> tuple[dict, dict, str, str]:
    for path, label in [
        (args.binary, "release binary"),
        (args.fixture_record, "fixture record"),
        (args.execution_manifest, "execution manifest"),
        (args.method_contract, "method contract"),
        (args.candidate, "candidate artifact"),
    ]:
        require(path.is_file(), f"{label} is missing")
    require(args.artifact_root.is_dir(), "canonical artifact root is missing")
    fixture_sha = sha256_file(args.fixture_record)
    method_sha = sha256_file(args.method_contract)
    binary_sha = sha256_file(args.binary)
    runner_sha = sha256_file(Path(__file__).resolve())
    manifest = read_json(args.execution_manifest)
    fixtures = read_json(args.fixture_record)
    method = read_json(args.method_contract)
    require(manifest["schema"] == "m6.3-r2.0-execution-manifest-v3", "execution schema")
    require(manifest["contract_sha256"] == CONTRACT_SHA256, "contract binding")
    require(manifest["measurement_method_contract_sha256"] == method_sha == METHOD_SHA256, "method binding")
    require(manifest["reference_fixture_record_sha256"] == fixture_sha == FIXTURE_SHA256, "fixture binding")
    require(manifest["release_binary_sha256"] == binary_sha, "binary binding")
    require(manifest["observer_runner_sha256"] == runner_sha, "runner binding")
    require(manifest["prior_failed_observer_controls_sha256"] == PRIOR_CONTROLS, "prior controls binding")
    require(manifest["candidate"]["sha256"] == CANDIDATE_SHA256, "candidate SHA binding")
    require(manifest["candidate"]["bytes"] == CANDIDATE_BYTES, "candidate byte binding")
    require(args.candidate.stat().st_size == CANDIDATE_BYTES, "candidate byte length")
    require(sha256_file(args.candidate) == CANDIDATE_SHA256, "candidate artifact SHA")
    require(tuple(tuple(pair) for pair in manifest["observer_pair_order"]) == PAIR_ORDER, "pair order")
    require(manifest["observer_controls_seen"] is False, "v3 controls already seen at freeze")
    require(manifest["candidate_localization_results_seen"] is False, "localization already seen")
    require(method["candidate_localization_results_seen"] is False, "method localization boundary")
    require(method["authorizes_one_observer_validation"] is True, "method observer authorization")
    require(fixtures["contract_sha256"] == CONTRACT_SHA256, "fixture contract binding")
    require([row["fixture_id"] for row in fixtures["fixtures"]] == ["short_english", "short_thai"], "fixture order")
    return manifest, fixtures, binary_sha, method_sha


def sample_env(
    fixture: dict, path: str, order: tuple[str, str], args: argparse.Namespace,
    binary_sha: str, method_sha: str, host_id: str, output: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "COLIBRI_ARTIFACT_ROOT": str(args.artifact_root),
        "COLIBRI_R2_FIXTURE": fixture["fixture_id"],
        "COLIBRI_R2_PATH": path,
        "COLIBRI_R2_OBSERVER_ORDER": ",".join(order),
        "COLIBRI_R2_SAMPLE_OUTPUT": str(output),
        "COLIBRI_R2_EXPECT_INPUT_SHA256": fixture["expert_input"]["sha256"],
        "COLIBRI_R2_EXPECT_ROUTER_WEIGHTS_SHA256": fixture["router_weights"]["sha256"],
        "COLIBRI_R2_EXPECT_SELECTED_IDS": ",".join(map(str, fixture["selected_expert_ids"])),
        "COLIBRI_R2_EXPECT_REFERENCE_OUTPUT_SHA256": fixture["reference_routed_output"]["sha256"],
        "COLIBRI_R2_FIXTURE_RECORD_SHA256": FIXTURE_SHA256,
        "COLIBRI_R2_METHOD_CONTRACT_SHA256": method_sha,
        "COLIBRI_R2_RELEASE_BINARY_SHA256": binary_sha,
        "COLIBRI_R2_HOST_ID": host_id,
        "COLIBRI_R2_CANDIDATE_PATH": str(args.candidate),
    })
    return env


def run_pair(
    args: argparse.Namespace, fixture: dict, path: str, pair: int,
    order: tuple[str, str], binary_sha: str, method_sha: str, host_id: str,
) -> dict:
    output = args.work_root / f"{fixture['fixture_id']}-{path}-pair-{pair:02d}.json"
    require(not output.exists(), f"pair output already exists: {output}")
    completed = subprocess.run(
        [str(args.binary), TEST_FILTER, "--exact", "--nocapture"],
        cwd=args.repo_root,
        env=sample_env(fixture, path, order, args, binary_sha, method_sha, host_id, output),
        check=False, text=True, capture_output=True,
    )
    if completed.returncode != 0:
        raise ControlError(
            f"pair failed fixture={fixture['fixture_id']} path={path} pair={pair} "
            f"exit={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    require(output.is_file(), "pair process did not write evidence")
    sample = read_json(output)
    require(sample["schema"] == "m6.3-r2.0-observer-pair-sample-v2", "pair schema")
    require(sample["fixture"] == fixture["fixture_id"], "pair fixture")
    require(sample["path"] == path, "pair path")
    require(tuple(sample["observer_order"]) == order, "pair observer order")
    require(sample["contract_sha256"] == CONTRACT_SHA256, "pair contract")
    require(sample["method_contract_sha256"] == method_sha, "pair method")
    require(sample["release_binary_sha256"] == binary_sha, "pair binary")
    enabled = sample["states"]["enabled"]
    disabled = sample["states"]["disabled"]
    require(enabled["output_sha256"] == disabled["output_sha256"], "pair output identity")
    overhead = 100.0 * (
        enabled["timed_total_nanos"] - disabled["timed_total_nanos"]
    ) / disabled["timed_total_nanos"]
    return {
        "fixture": fixture["fixture_id"], "path": path, "pair": pair,
        "attempt_ordinal": 1, "execution_order": list(order),
        "release_binary_sha256": binary_sha, "host_id": host_id,
        "instrumented_output_sha256": enabled["output_sha256"],
        "uninstrumented_output_sha256": disabled["output_sha256"],
        "instrumented_total_nanos": enabled["timed_total_nanos"],
        "uninstrumented_total_nanos": disabled["timed_total_nanos"],
        "overhead_percent": overhead, "raw_pair_sample": sample,
    }
def validate_controls(records: list[dict]) -> dict[str, dict]:
    require(len(records) == 20, "observer pair count")
    summary: dict[str, dict] = {}
    for fixture in ("short_english", "short_thai"):
        for path in ("reference", "candidate"):
            selected = [row for row in records if row["fixture"] == fixture and row["path"] == path]
            require([row["pair"] for row in selected] == [1, 2, 3, 4, 5], "pair coverage")
            overheads = [row["overhead_percent"] for row in selected]
            every_pair = all(value <= 10.0 for value in overheads)
            median = statistics.median(overheads)
            passed = every_pair and median <= 5.0
            summary[f"{fixture}:{path}"] = {
                "overhead_percent_by_pair": overheads,
                "median_overhead_percent": median,
                "minimum_overhead_percent": min(overheads),
                "maximum_overhead_percent": max(overheads),
                "passed": passed,
            }
            require(every_pair, f"observer pair overhead >10%: {fixture}/{path}")
            require(median <= 5.0, f"observer median overhead >5%: {fixture}/{path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture-record", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--method-contract", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest, fixture_doc, binary_sha, method_sha = validate_inputs(args)
    if args.validate_only:
        print("R2.0 low-observer control preflight: PASS")
        return 0
    require(not args.output.exists(), "observer output must be new")
    require(not args.work_root.exists(), "observer work root must be new")
    args.work_root.mkdir(parents=True)
    host_id = platform.node()
    require(host_id == manifest["host_id"], "host identity drift")
    controls: list[dict] = []
    try:
        for fixture in fixture_doc["fixtures"]:
            for path in ("reference", "candidate"):
                for pair, order in enumerate(PAIR_ORDER, start=1):
                    controls.append(
                        run_pair(args, fixture, path, pair, order, binary_sha, method_sha, host_id)
                    )
        summary = validate_controls(controls)
        document = {
            "schema": "m6.3-r2.0-observer-controls-v3",
            "status": "passed",
            "contract_sha256": CONTRACT_SHA256,
            "measurement_method_contract_sha256": method_sha,
            "execution_manifest_sha256": sha256_file(args.execution_manifest),
            "reference_fixture_record_sha256": FIXTURE_SHA256,
            "release_binary_sha256": binary_sha,
            "host_id": host_id,
            "pair_order": [list(pair) for pair in PAIR_ORDER],
            "observer_control_process_samples": 20,
            "observer_controls": controls,
            "summary": summary,
            "prior_failed_observer_controls_sha256": PRIOR_CONTROLS,
            "candidate_localization_results_seen": False,
            "authorizes_localization_samples": True,
        }
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        print("R2.0 low-observer controls: PASS (20 pairs/processes)")
        return 0
    except Exception as error:
        failure = {
            "schema": "m6.3-r2.0-observer-controls-v3",
            "status": "failed",
            "reason": str(error),
            "completed_pairs": len(controls),
            "candidate_localization_results_seen": False,
            "authorizes_localization_samples": False,
        }
        args.output.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ControlError as error:
        print(f"R2.0 low-observer controls: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
