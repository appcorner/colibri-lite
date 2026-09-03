#!/usr/bin/env python3
"""Run the frozen M6.3-R2.0 40-sample bottleneck-localization matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

CONTRACT_SHA256 = "02322820758848b5ebfe3f020db026f1aa5e3dd54ce007fb71202f50e54aceca"
METHOD_SHA256 = "ee2a85a1531956ad790e27c52bfb87cc7014c93dc4140d85576d020e5a45307f"
HARNESS_AMENDMENT_SHA256 = "c3409036b22b85ab176477e662d26c15fa18cede724e57f8ea61ecc3ddc75894"
PRIOR_OBSERVER_V4_SHA256 = "a6cf0d7518e889ddc27234de7a45100a0c8363e31fa0a3eec61dc29feed3a687"
FIXTURE_SHA256 = "c0490d2a40a214579e7633fcd4706e64186d183a82af6d176eaa1918d7f056e2"
CANDIDATE_SHA256 = "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2"
CANDIDATE_BYTES = 679_477_248
FIXTURES = ("short_english", "short_thai")
VIEWS = ("compute_only_preloaded", "load_plus_compute")
PAIR_ORDER = (
    ("candidate", "reference"),
    ("reference", "candidate"),
    ("candidate", "reference"),
    ("reference", "candidate"),
    ("candidate", "reference"),
)
TEST_FILTER = "full_model_validation_tests::r2_benchmark_tests::m6_3_r2_0_localization_single_sample"


class LocalizationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalizationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

def validate_observer_controls(
    controls: dict, manifest: dict, manifest_sha: str, binary_sha: str, host_id: str
) -> str:
    require(controls.get("schema") == "m6.3-r2.0-observer-controls-v5", "observer schema")
    require(controls.get("status") == "passed", "fresh observer controls did not pass")
    require(controls.get("authorizes_localization_samples") is True, "observer controls do not authorize localization")
    require(controls.get("contract_sha256") == CONTRACT_SHA256, "observer contract binding")
    require(controls.get("measurement_method_contract_sha256") == METHOD_SHA256, "observer method binding")
    require(controls.get("localization_harness_amendment_sha256") == HARNESS_AMENDMENT_SHA256, "observer harness amendment binding")
    require(controls.get("prior_observer_v4_controls_sha256") == PRIOR_OBSERVER_V4_SHA256, "observer v4 provenance")
    require(controls.get("execution_manifest_sha256") == manifest_sha, "observer execution manifest binding")
    require(controls.get("release_binary_sha256") == binary_sha, "observer binary binding")
    require(controls.get("host_id") == host_id, "observer host binding")
    require(controls.get("observer_control_process_samples") == 20, "observer process count")
    summary = controls.get("summary")
    require(isinstance(summary, dict) and len(summary) == 4, "observer summary coverage")
    require(all(row.get("passed") is True for row in summary.values()), "observer summary contains failure")
    rows = controls.get("observer_controls")
    require(isinstance(rows, list) and len(rows) == 20, "observer pair matrix")
    require(all(row.get("attempt_ordinal") == 1 for row in rows), "observer retry detected")
    require(all(row.get("instrumented_output_sha256") == row.get("uninstrumented_output_sha256") for row in rows), "observer output mismatch")
    require(manifest.get("observer_controls_seen") is False, "observer controls already existed at manifest freeze")

def validate_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, str, str, str]:
    required_files = (
        (args.binary, "release binary"),
        (args.fixture_record, "fixture record"),
        (args.execution_manifest, "execution manifest"),
        (args.method_contract, "measurement method contract"),
        (args.harness_amendment, "harness amendment"),
        (args.observer_controls, "fresh observer controls"),
        (args.validator, "localization validator"),
        (args.candidate, "candidate artifact"),
    )
    for path, label in required_files:
        require(path.is_file(), f"{label} is missing")
    require(args.artifact_root.is_dir(), "canonical artifact root is missing")
    manifest = read_json(args.execution_manifest)
    fixtures = read_json(args.fixture_record)
    controls = read_json(args.observer_controls)
    manifest_sha = sha256_file(args.execution_manifest)
    binary_sha = sha256_file(args.binary)
    observer_sha = sha256_file(args.observer_controls)
    runner_sha = sha256_file(Path(__file__).resolve())
    validator_sha = sha256_file(args.validator)
    require(manifest.get("schema") == "m6.3-r2.0-execution-manifest-v5", "execution manifest schema")
    require(manifest.get("contract_sha256") == CONTRACT_SHA256, "manifest contract binding")
    require(manifest.get("measurement_method_contract_sha256") == METHOD_SHA256 == sha256_file(args.method_contract), "manifest method binding")
    require(manifest.get("localization_harness_amendment_sha256") == HARNESS_AMENDMENT_SHA256 == sha256_file(args.harness_amendment), "manifest harness amendment binding")
    require(manifest.get("prior_observer_v4_controls_sha256") == PRIOR_OBSERVER_V4_SHA256, "manifest observer v4 provenance")
    require(manifest.get("reference_fixture_record_sha256") == FIXTURE_SHA256 == sha256_file(args.fixture_record), "manifest fixture binding")
    require(manifest.get("release_binary_sha256") == binary_sha, "manifest binary binding")
    require(manifest.get("localization_runner_sha256") == runner_sha, "localization runner binding")
    require(manifest.get("localization_validator_sha256") == validator_sha, "localization validator binding")
    require(manifest.get("candidate", {}).get("sha256") == CANDIDATE_SHA256, "candidate SHA binding")
    require(manifest.get("candidate", {}).get("bytes") == CANDIDATE_BYTES, "candidate byte binding")
    require(args.candidate.stat().st_size == CANDIDATE_BYTES, "candidate byte length")
    require(sha256_file(args.candidate) == CANDIDATE_SHA256, "candidate artifact SHA")
    require(manifest.get("candidate_localization_results_seen") is False, "localization already existed at freeze")
    require(manifest.get("localization_sample_count_seen") == 0, "localization sample count at freeze")
    require(fixtures.get("contract_sha256") == CONTRACT_SHA256, "fixture contract binding")
    require([row.get("fixture_id") for row in fixtures.get("fixtures", [])] == list(FIXTURES), "fixture order")
    host_id = platform.node()
    require(manifest.get("host_id") == host_id, "host identity drift")
    validate_observer_controls(controls, manifest, manifest_sha, binary_sha, host_id)
    return manifest, fixtures, controls, binary_sha, observer_sha, host_id

def sample_env(
    fixture: dict, view: str, path: str, pair: int, order_index: int,
    args: argparse.Namespace, binary_sha: str, host_id: str, output: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "COLIBRI_ARTIFACT_ROOT": str(args.artifact_root),
        "COLIBRI_R2_FIXTURE": fixture["fixture_id"],
        "COLIBRI_R2_VIEW": view,
        "COLIBRI_R2_PATH": path,
        "COLIBRI_R2_PAIR": str(pair),
        "COLIBRI_R2_ORDER_INDEX": str(order_index),
        "COLIBRI_R2_SAMPLE_OUTPUT": str(output),
        "COLIBRI_R2_EXPECT_INPUT_SHA256": fixture["expert_input"]["sha256"],
        "COLIBRI_R2_EXPECT_ROUTER_WEIGHTS_SHA256": fixture["router_weights"]["sha256"],
        "COLIBRI_R2_EXPECT_SELECTED_IDS": ",".join(map(str, fixture["selected_expert_ids"])),
        "COLIBRI_R2_EXPECT_REFERENCE_OUTPUT_SHA256": fixture["reference_routed_output"]["sha256"],
        "COLIBRI_R2_FIXTURE_RECORD_SHA256": FIXTURE_SHA256,
        "COLIBRI_R2_RELEASE_BINARY_SHA256": binary_sha,
        "COLIBRI_R2_HOST_ID": host_id,
        "COLIBRI_R2_CANDIDATE_PATH": str(args.candidate),
    })
    return env

def run_sample(
    args: argparse.Namespace, fixture: dict, view: str, path: str,
    pair: int, order_index: int, binary_sha: str, host_id: str,
) -> dict:
    output = args.work_root / (
        f"{fixture['fixture_id']}-{view}-pair-{pair:02d}-"
        f"order-{order_index}-{path}.json"
    )
    require(not output.exists(), f"sample output already exists: {output}")
    completed = subprocess.run(
        [str(args.binary), TEST_FILTER, "--exact", "--nocapture"],
        cwd=args.repo_root,
        env=sample_env(fixture, view, path, pair, order_index, args, binary_sha, host_id, output),
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise LocalizationError(
            f"sample failed fixture={fixture['fixture_id']} view={view} "
            f"pair={pair} path={path} exit={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    require(output.is_file(), "sample did not write evidence")
    sample = read_json(output)
    require(sample.get("schema") == "m6.3-r2.0-localization-sample-v1", "sample schema")
    require(sample.get("contract_sha256") == CONTRACT_SHA256, "sample contract binding")
    require(sample.get("fixture") == fixture["fixture_id"], "sample fixture")
    require(sample.get("view") == view, "sample view")
    require(sample.get("path") == path, "sample path")
    require(sample.get("pair") == pair, "sample pair")
    require(sample.get("order_index") == order_index, "sample order index")
    require(sample.get("attempt_ordinal") == 1, "sample retry detected")
    require(sample.get("release_binary_sha256") == binary_sha, "sample binary binding")
    require(sample.get("host_id") == host_id, "sample host binding")
    expected_iterations = 25 if view == "compute_only_preloaded" else 10
    require(sample.get("timed_iterations") == expected_iterations, "sample timed iterations")
    require(isinstance(sample.get("stages"), dict) and "expert_total" in sample["stages"], "sample expert_total")
    output_hash = sample.get("output_sha256")
    require(isinstance(output_hash, str) and len(output_hash) == 64, "sample output hash")
    return sample


def base_document(
    manifest: dict, controls: dict, args: argparse.Namespace,
    binary_sha: str, observer_sha: str, host_id: str, samples: list[dict],
) -> dict:
    return {
        "schema": "m6.3-r2.0-localization-samples-v1",
        "contract_sha256": CONTRACT_SHA256,
        "measurement_method_contract_sha256": METHOD_SHA256,
        "localization_harness_amendment_sha256": HARNESS_AMENDMENT_SHA256,
        "prior_observer_v4_controls_sha256": PRIOR_OBSERVER_V4_SHA256,
        "observer_controls_sha256": observer_sha,
        "reference_fixture_record_sha256": FIXTURE_SHA256,
        "execution_manifest_sha256": sha256_file(args.execution_manifest),
        "instrumented_source_commit": manifest["instrumented_source_commit"],
        "release_binary_sha256": binary_sha,
        "host_id": host_id,
        "timer_implementation_identity": manifest["timer_implementation_identity"],
        "collector_versions": manifest["collector_versions"],
        "observer_controls_passed_before_localization": True,
        "observer_controls": controls["observer_controls"],
        "sample_count": len(samples),
        "samples": samples,
        "authorizes_r2_1_hypothesis_design": False,
        "authorizes_m6_4": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture-record", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--method-contract", type=Path, required=True)
    parser.add_argument("--harness-amendment", type=Path, required=True)
    parser.add_argument("--observer-controls", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest, fixture_doc, controls, binary_sha, observer_sha, host_id = validate_inputs(args)
    if args.validate_only:
        print("R2.0 localization preflight: PASS")
        return 0
    partial = args.output.with_name(args.output.stem + ".partial.json")
    require(not args.output.exists(), "localization output must be new")
    require(not partial.exists(), "localization partial output must be new")
    require(not args.work_root.exists(), "localization work root must be new")
    args.work_root.mkdir(parents=True)
    samples: list[dict] = []
    try:
        for fixture in fixture_doc["fixtures"]:
            for view in VIEWS:
                for pair, order in enumerate(PAIR_ORDER, start=1):
                    for order_index, path in enumerate(order, start=1):
                        sample = run_sample(
                            args, fixture, view, path, pair, order_index,
                            binary_sha, host_id,
                        )
                        samples.append(sample)
                        write_json(
                            partial,
                            base_document(
                                manifest, controls, args, binary_sha,
                                observer_sha, host_id, samples,
                            ),
                        )
        require(len(samples) == 40, "localization matrix did not produce 40 samples")
        document = base_document(
            manifest, controls, args, binary_sha, observer_sha, host_id, samples,
        )
        document["sample_count"] = 40
        write_json(args.output, document)
        completed = subprocess.run(
            [sys.executable, str(args.validator), str(args.output)],
            cwd=args.repo_root,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise LocalizationError(
                f"final localization validation failed\nstdout={completed.stdout}\n"
                f"stderr={completed.stderr}"
            )
        print("R2.0 localization matrix: PASS (40 samples)")
        print(completed.stdout.strip())
        return 0
    except Exception as error:
        failure = {
            "schema": "m6.3-r2.0-localization-run-failure-v1",
            "reason": str(error),
            "completed_samples": len(samples),
            "automatic_retry": False,
            "authorizes_r2_1_hypothesis_design": False,
            "authorizes_m6_4": False,
        }
        write_json(args.output.with_name(args.output.stem + ".failure.json"), failure)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LocalizationError as error:
        print(f"R2.0 localization: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
