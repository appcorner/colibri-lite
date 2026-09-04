#!/usr/bin/env python3
"""Run the frozen M6.3-R2.1c 72-sample three-path performance matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

CONTRACT_SHA256 = "76bfc4a850a8d89fb5901eda338568b7226b36acb7207324575568ea21f6cc2a"
FIXTURE_SHA256 = "c0490d2a40a214579e7633fcd4706e64186d183a82af6d176eaa1918d7f056e2"
CANDIDATE_SHA256 = "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2"
CANDIDATE_BYTES = 679_477_248
FIXTURES = ("short_english", "short_thai")
VIEWS = ("compute_only_preloaded", "load_plus_compute")
ORDERS = (
    ("native_avx2_fma_group32", "scalar_group32", "reference_f32"),
    ("native_avx2_fma_group32", "reference_f32", "scalar_group32"),
    ("scalar_group32", "native_avx2_fma_group32", "reference_f32"),
    ("scalar_group32", "reference_f32", "native_avx2_fma_group32"),
    ("reference_f32", "native_avx2_fma_group32", "scalar_group32"),
    ("reference_f32", "scalar_group32", "native_avx2_fma_group32"),
)
TEST_FILTER = "full_model_validation_tests::r2_benchmark_tests::m6_3_r2_1_performance_single_sample"

class PerformanceError(RuntimeError):
    pass

def require(condition: bool, message: str) -> None:
    if not condition:
        raise PerformanceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def validate_inputs(args: argparse.Namespace) -> tuple[dict, str, str]:
    for path, label in ((args.binary, "release binary"), (args.fixture_record, "fixture record"),
                        (args.contract, "R2.1 contract"), (args.candidate, "candidate artifact"),
                        (args.validator, "validator"), (args.execution_manifest, "execution manifest")):
        require(path.is_file(), f"{label} is missing")
    require(args.artifact_root.is_dir(), "canonical artifact root is missing")
    require(sha256_file(args.contract) == CONTRACT_SHA256, "contract SHA mismatch")
    require(sha256_file(args.fixture_record) == FIXTURE_SHA256, "fixture SHA mismatch")
    require(args.candidate.stat().st_size == CANDIDATE_BYTES, "candidate byte length mismatch")
    require(sha256_file(args.candidate) == CANDIDATE_SHA256, "candidate SHA mismatch")
    manifest = read_json(args.execution_manifest)
    binary_sha = sha256_file(args.binary)
    host_id = platform.node()
    require(manifest.get("schema") == "m6.3-r2.1-performance-execution-v1", "execution schema")
    require(manifest.get("contract_sha256") == CONTRACT_SHA256, "execution contract binding")
    require(manifest.get("release_binary_sha256") == binary_sha, "execution binary binding")
    require(manifest.get("runner_sha256") == sha256_file(Path(__file__).resolve()), "execution runner binding")
    require(manifest.get("validator_sha256") == sha256_file(args.validator), "execution validator binding")
    require(manifest.get("host_id") == host_id, "host identity drift")
    require(manifest.get("results_seen") is False, "results existed at freeze")
    fixture_doc = read_json(args.fixture_record)
    require([row.get("fixture_id") for row in fixture_doc.get("fixtures", [])] == list(FIXTURES), "fixture order")
    return fixture_doc, binary_sha, host_id


def sample_env(args: argparse.Namespace, fixture: dict, view: str, path: str,
               triplet: int, order_index: int, binary_sha: str, host_id: str, output: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "COLIBRI_ARTIFACT_ROOT": str(args.artifact_root),
        "COLIBRI_R2_FIXTURE": fixture["fixture_id"],
        "COLIBRI_R2_VIEW": view,
        "COLIBRI_R2_PATH": path,
        "COLIBRI_R2_PAIR": str(triplet),
        "COLIBRI_R2_ORDER_INDEX": str(order_index),
        "COLIBRI_R2_SAMPLE_OUTPUT": str(output),
        "COLIBRI_R2_EXPECT_INPUT_SHA256": fixture["expert_input"]["sha256"],
        "COLIBRI_R2_EXPECT_ROUTER_WEIGHTS_SHA256": fixture["router_weights"]["sha256"],
        "COLIBRI_R2_EXPECT_SELECTED_IDS": ",".join(map(str, fixture["selected_expert_ids"])),
        "COLIBRI_R2_EXPECT_REFERENCE_OUTPUT_SHA256": fixture["reference_routed_output"]["sha256"],
        "COLIBRI_R2_RELEASE_BINARY_SHA256": binary_sha,
        "COLIBRI_R2_HOST_ID": host_id,
        "COLIBRI_R2_CANDIDATE_PATH": str(args.candidate),
    })
    return env

def run_sample(args: argparse.Namespace, fixture: dict, view: str, path: str,
               triplet: int, order_index: int, binary_sha: str, host_id: str) -> dict:
    output = args.work_root / (
        f"{fixture['fixture_id']}-{view}-triplet-{triplet:02d}-order-{order_index}-{path}.json"
    )
    require(not output.exists(), f"sample output already exists: {output}")
    completed = subprocess.run(
        [str(args.binary), TEST_FILTER, "--exact", "--nocapture"],
        cwd=args.repo_root,
        env=sample_env(args, fixture, view, path, triplet, order_index, binary_sha, host_id, output),
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise PerformanceError(
            f"sample failed fixture={fixture['fixture_id']} view={view} triplet={triplet} "
            f"path={path} exit={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    require(output.is_file(), "sample did not write evidence")
    sample = read_json(output)
    require(sample.get("schema") == "m6.3-r2.1-performance-sample-v1", "sample schema")
    require(sample.get("contract_sha256") == CONTRACT_SHA256, "sample contract binding")
    require(sample.get("fixture") == fixture["fixture_id"], "sample fixture")
    require(sample.get("view") == view and sample.get("path") == path, "sample view/path")
    require(sample.get("triplet") == triplet and sample.get("order_index") == order_index, "sample order")
    require(sample.get("attempt_ordinal") == 1, "sample retry detected")
    require(sample.get("release_binary_sha256") == binary_sha, "sample binary binding")
    require(sample.get("host_id") == host_id, "sample host binding")
    return sample

def base_document(args: argparse.Namespace, binary_sha: str, host_id: str, samples: list[dict]) -> dict:
    return {
        "schema": "m6.3-r2.1-performance-samples-v1",
        "contract_sha256": CONTRACT_SHA256,
        "execution_manifest_sha256": sha256_file(args.execution_manifest),
        "release_binary_sha256": binary_sha,
        "host_id": host_id,
        "sample_count": len(samples),
        "automatic_retry": False,
        "triplet_orders": [list(order) for order in ORDERS],
        "samples": samples,
        "authorizes_r2_2_design": False,
        "authorizes_r2_2_implementation": False,
        "authorizes_m6_4": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture-record", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    fixture_doc, binary_sha, host_id = validate_inputs(args)
    if args.validate_only:
        print("R2.1 performance preflight: PASS")
        return 0
    partial = args.output.with_name(args.output.stem + ".partial.json")
    require(not args.output.exists(), "performance output must be new")
    require(not partial.exists(), "performance partial output must be new")
    require(not args.work_root.exists(), "performance work root must be new")
    args.work_root.mkdir(parents=True)
    samples: list[dict] = []
    try:
        for fixture in fixture_doc["fixtures"]:
            for view in VIEWS:
                for triplet, order in enumerate(ORDERS, start=1):
                    for order_index, path in enumerate(order, start=1):
                        samples.append(run_sample(
                            args, fixture, view, path, triplet, order_index, binary_sha, host_id
                        ))
                        write_json(partial, base_document(args, binary_sha, host_id, samples))
        require(len(samples) == 72, "performance matrix did not produce 72 samples")
        document = base_document(args, binary_sha, host_id, samples)
        write_json(args.output, document)
        completed = subprocess.run(
            [sys.executable, str(args.validator), str(args.output)],
            cwd=args.repo_root, check=False, text=True, capture_output=True,
        )
        if completed.returncode != 0:
            raise PerformanceError(
                f"final performance validation failed\nstdout={completed.stdout}\nstderr={completed.stderr}"
            )
        print("R2.1 performance matrix: COMPLETE (72 samples)")
        print(completed.stdout.strip())
        return 0
    except Exception as error:
        write_json(args.output.with_name(args.output.stem + ".failure.json"), {
            "schema": "m6.3-r2.1-performance-run-failure-v1",
            "reason": str(error), "completed_samples": len(samples), "automatic_retry": False,
        })
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PerformanceError as error:
        print(f"R2.1 performance: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
