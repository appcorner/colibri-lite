#!/usr/bin/env python3
"""Run pre-registered M6.3-R2.0 observer-effect controls."""

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
CANDIDATE_SHA256 = "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2"
CANDIDATE_BYTES = 679_477_248
PAIR_ORDER = (
    ("enabled", "disabled"),
    ("disabled", "enabled"),
    ("enabled", "disabled"),
    ("disabled", "enabled"),
    ("enabled", "disabled"),
)
TEST_FILTER = "full_model_validation_tests::r2_benchmark_tests::m6_3_r2_0_compute_only_single_sample"


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


def validate_inputs(args: argparse.Namespace) -> tuple[dict, dict, str]:
    require(args.binary.is_file(), "release test binary is missing")
    require(args.fixture_record.is_file(), "fixture record is missing")
    require(args.execution_manifest.is_file(), "execution manifest is missing")
    require(args.candidate.is_file(), "candidate artifact is missing")
    require(args.artifact_root.is_dir(), "canonical artifact root is missing")
    fixture_sha = sha256_file(args.fixture_record)
    manifest = read_json(args.execution_manifest)
    fixtures = read_json(args.fixture_record)
    binary_sha = sha256_file(args.binary)
    require(manifest["schema"] == "m6.3-r2.0-execution-manifest-v1", "execution schema")
    require(manifest["contract_sha256"] == CONTRACT_SHA256, "execution contract SHA")
    require(manifest["reference_fixture_record_sha256"] == fixture_sha, "fixture SHA binding")
    require(manifest["release_binary_sha256"] == binary_sha, "binary SHA binding")
    require(manifest["candidate"]["sha256"] == CANDIDATE_SHA256, "candidate SHA binding")
    require(manifest["candidate"]["bytes"] == CANDIDATE_BYTES, "candidate byte binding")
    require(args.candidate.stat().st_size == CANDIDATE_BYTES, "candidate byte length")
    require(sha256_file(args.candidate) == CANDIDATE_SHA256, "candidate artifact SHA")
    require(tuple(tuple(pair) for pair in manifest["observer_pair_order"]) == PAIR_ORDER, "observer pair order")
    require(manifest["observer_controls_seen"] is False, "observer controls already seen at freeze")
    require(manifest["candidate_localization_results_seen"] is False, "localization results already seen")
    require(fixtures["contract_sha256"] == CONTRACT_SHA256, "fixture contract SHA")
    require([item["fixture_id"] for item in fixtures["fixtures"]] == ["short_english", "short_thai"], "fixture order")
    return manifest, fixtures, binary_sha


def sample_env(
    base: os._Environ[str], fixture: dict, path: str, observer: str,
    args: argparse.Namespace, fixture_sha: str, binary_sha: str, host_id: str,
    output: Path,
) -> dict[str, str]:
    env = dict(base)
    env.update({
        "COLIBRI_ARTIFACT_ROOT": str(args.artifact_root),
        "COLIBRI_R2_FIXTURE": fixture["fixture_id"],
        "COLIBRI_R2_PATH": path,
        "COLIBRI_R2_OBSERVER": observer,
        "COLIBRI_R2_SAMPLE_OUTPUT": str(output),
        "COLIBRI_R2_EXPECT_INPUT_SHA256": fixture["expert_input"]["sha256"],
        "COLIBRI_R2_EXPECT_ROUTER_WEIGHTS_SHA256": fixture["router_weights"]["sha256"],
        "COLIBRI_R2_EXPECT_SELECTED_IDS": ",".join(map(str, fixture["selected_expert_ids"])),
        "COLIBRI_R2_EXPECT_REFERENCE_OUTPUT_SHA256": fixture["reference_routed_output"]["sha256"],
        "COLIBRI_R2_FIXTURE_RECORD_SHA256": fixture_sha,
        "COLIBRI_R2_RELEASE_BINARY_SHA256": binary_sha,
        "COLIBRI_R2_HOST_ID": host_id,
        "COLIBRI_R2_CANDIDATE_PATH": str(args.candidate),
    })
    return env


def run_sample(
    args: argparse.Namespace, fixture: dict, path: str, observer: str,
    fixture_sha: str, binary_sha: str, host_id: str, output: Path,
) -> dict:
    require(not output.exists(), f"sample output already exists: {output}")
    command = [str(args.binary), TEST_FILTER, "--exact", "--nocapture"]
    completed = subprocess.run(
        command,
        cwd=args.repo_root,
        env=sample_env(os.environ, fixture, path, observer, args, fixture_sha, binary_sha, host_id, output),
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ControlError(
            f"sample failed fixture={fixture['fixture_id']} path={path} observer={observer} "
            f"exit={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    require(output.is_file(), "sample did not write evidence")
    sample = read_json(output)
    require(sample["observer"] == observer, "sample observer identity")
    require(sample["path"] == path, "sample path identity")
    return sample

def paired_record(
    fixture: str, path: str, pair: int, first: dict, second: dict,
    binary_sha: str, host_id: str,
) -> dict:
    by_observer = {first["observer"]: first, second["observer"]: second}
    require(set(by_observer) == {"enabled", "disabled"}, "observer pair variants")
    instrumented = by_observer["enabled"]
    uninstrumented = by_observer["disabled"]
    require(instrumented["output_sha256"] == uninstrumented["output_sha256"], "observer output identity")
    overhead = 100.0 * (
        instrumented["timed_total_nanos"] - uninstrumented["timed_total_nanos"]
    ) / uninstrumented["timed_total_nanos"]
    return {
        "fixture": fixture,
        "path": path,
        "pair": pair,
        "attempt_ordinal": 1,
        "release_binary_sha256": binary_sha,
        "host_id": host_id,
        "instrumented_output_sha256": instrumented["output_sha256"],
        "uninstrumented_output_sha256": uninstrumented["output_sha256"],
        "instrumented_total_nanos": instrumented["timed_total_nanos"],
        "uninstrumented_total_nanos": uninstrumented["timed_total_nanos"],
        "overhead_percent": overhead,
        "execution_order": [first["observer"], second["observer"]],
    }


def validate_controls(records: list[dict]) -> dict[str, dict]:
    require(len(records) == 20, "observer control pair count")
    summary: dict[str, dict] = {}
    for fixture in ("short_english", "short_thai"):
        for path in ("reference", "candidate"):
            selected = [row for row in records if row["fixture"] == fixture and row["path"] == path]
            require([row["pair"] for row in selected] == [1, 2, 3, 4, 5], "observer pair coverage")
            overheads = [row["overhead_percent"] for row in selected]
            require(all(value <= 10.0 for value in overheads), f"observer pair overhead >10%: {fixture}/{path}")
            median = statistics.median(overheads)
            require(median <= 5.0, f"observer median overhead >5%: {fixture}/{path}")
            summary[f"{fixture}:{path}"] = {
                "overhead_percent_by_pair": overheads,
                "median_overhead_percent": median,
                "passed": True,
            }
    return summary

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture-record", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest, fixture_doc, binary_sha = validate_inputs(args)
    if args.validate_only:
        print("R2.0 observer-control preflight: PASS")
        return 0
    require(not args.output.exists(), "observer-control output must be new")
    require(not args.work_root.exists(), "observer-control work root must be new")
    args.work_root.mkdir(parents=True)
    fixture_sha = manifest["reference_fixture_record_sha256"]
    host_id = platform.node()
    require(host_id == manifest["host_id"], "host identity drift")
    controls: list[dict] = []
    raw_samples: list[dict] = []
    try:
        for fixture in fixture_doc["fixtures"]:
            for path in ("reference", "candidate"):
                for pair, order in enumerate(PAIR_ORDER, start=1):
                    pair_samples = []
                    for ordinal, observer in enumerate(order, start=1):
                        output = args.work_root / (
                            f"{fixture['fixture_id']}-{path}-pair-{pair:02d}-"
                            f"order-{ordinal}-{observer}.json"
                        )
                        sample = run_sample(
                            args, fixture, path, observer, fixture_sha,
                            binary_sha, host_id, output,
                        )
                        raw_samples.append(sample)
                        pair_samples.append(sample)
                    controls.append(
                        paired_record(
                            fixture["fixture_id"], path, pair,
                            pair_samples[0], pair_samples[1], binary_sha, host_id,
                        )
                    )
        summary = validate_controls(controls)
        document = {
            "schema": "m6.3-r2.0-observer-controls-v1",
            "status": "passed",
            "contract_sha256": CONTRACT_SHA256,
            "execution_manifest_sha256": sha256_file(args.execution_manifest),
            "reference_fixture_record_sha256": fixture_sha,
            "release_binary_sha256": binary_sha,
            "host_id": host_id,
            "pair_order": [list(pair) for pair in PAIR_ORDER],
            "observer_controls": controls,
            "summary": summary,
            "raw_process_samples": raw_samples,
            "authorizes_localization_samples": True,
        }
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"R2.0 observer controls: PASS ({len(controls)} pairs)")
        return 0
    except Exception as error:
        failure = {
            "schema": "m6.3-r2.0-observer-controls-v1",
            "status": "failed",
            "reason": str(error),
            "completed_process_samples": len(raw_samples),
            "completed_pairs": len(controls),
            "authorizes_localization_samples": False,
        }
        args.output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ControlError as error:
        print(f"R2.0 observer controls: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
