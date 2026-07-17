"""Capture the deterministic M5.2-01 representative expert-trace corpus.

The full model is executed one fixture at a time. Each newly captured trace is
regenerated twice before the first copy is atomically promoted to the tracked
corpus directory. This script records no timings or machine-local paths in
canonical trace content and never simulates a cache policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


EXPECTED_ARTIFACT_ROOT = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
EXPECTED_TRACE_SCHEMA = "colibri-qwen3-moe-m5.2-01-ordered-expert-trace-v2"
CONTROL_TRACE = "models/qwen3-30b-a3b/m5.1-00-ordered-expert-trace-v1.json"
CONTROL_MANIFEST = "models/qwen3-30b-a3b/m5.1-00-trace-manifest-v1.json"
PAYLOAD_BYTES = 18_874_368
EXPERT_COUNT = 128
LAYER_COUNT = 48


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_trace(trace: dict[str, Any], fixture: dict[str, Any]) -> None:
    assert trace["schema"] == EXPECTED_TRACE_SCHEMA
    assert trace["schema_version"] == 2
    assert trace["fixture_id"] == fixture["fixture_id"]
    assert trace["workload_class"] == fixture["workload_class"]
    assert trace["canonical_artifact_root_sha256"] == EXPECTED_ARTIFACT_ROOT
    assert trace["input_token_ids"] == fixture["token_ids"]
    assert trace["requested_generation_length"] == fixture["requested_generation_length"]
    assert trace["cache_configuration"]["budget_bytes"] == 18_874_368
    assert trace["cache_configuration"]["policy"] == "strict_global_lru"
    assert trace["runtime_configuration"]["compute_dtype"] == "F32"
    assert trace["runtime_configuration"]["mmap"] is False
    assert trace["runtime_configuration"]["prefetch"] is False
    assert trace["runtime_configuration"]["simd"] is False
    assert trace["runtime_configuration"]["threading"] is False
    assert trace["runtime_configuration"]["quantization"] is False
    assert trace["runtime_configuration"]["gpu"] is False
    records = trace["records"]
    assert len(records) == trace["counters"]["requested_trace_count"]
    assert len(records) == (len(fixture["token_ids"]) + fixture["requested_generation_length"] - 1) * LAYER_COUNT * 8
    assert trace["kv_cache"]["final_sequence_length"] == len(fixture["token_ids"]) + fixture["requested_generation_length"] - 1
    assert trace["kv_cache"]["capacity"] == fixture["kv_cache_capacity"]
    assert trace["expected_generated_token_ids"]
    for ordinal, record in enumerate(records):
        assert record["global_ordinal"] == ordinal
        assert record["fixture_id"] == fixture["fixture_id"]
        assert 0 <= record["layer_index"] < LAYER_COUNT
        assert 0 <= record["expert_id"] < EXPERT_COUNT
        assert record["layer_expert_key"] == f"layer.{record['layer_index']}.expert.{record['expert_id']}"
        assert record["payload_bytes"] == PAYLOAD_BYTES
        assert record["input_token_id"] < 151_936
        assert record["phase"] in ("prefill", "decode")
        assert isinstance(record["cache_hit"], bool)
        assert isinstance(record["loaded"], bool)
        assert record["evictions_caused"] >= 0
    counters = trace["counters"]
    assert counters["cache_hits"] + counters["cache_misses"] == len(records)
    assert counters["loads"] == counters["cache_misses"]
    assert counters["cache_hits"] == sum(record["cache_hit"] for record in records)
    assert counters["loads"] == sum(record["loaded"] for record in records)
    assert counters["evictions"] == sum(record["evictions_caused"] for record in records)
    assert counters["expert_bytes_read"] == counters["loads"] * PAYLOAD_BYTES
    assert counters["expert_payload_bytes_requested"] == len(records) * PAYLOAD_BYTES
    assert counters["cache_hits"] >= 0
    assert counters["cache_misses"] >= 0
    assert counters["loads"] >= 0
    assert counters["evictions"] >= 0


def capture_fixture(
    root: Path,
    executable: Path,
    artifact_root: Path,
    run_dir: Path,
    fixture: dict[str, Any],
    instrumentation_commit: str,
) -> dict[str, Any]:
    fixture_id = fixture["fixture_id"]
    first_bytes: bytes | None = None
    first_generated: list[int] | None = None
    repeat_hashes: list[str] = []
    for repeat in (1, 2):
        trace_path = run_dir / f"{fixture_id}-repeat-{repeat}.json"
        environment = os.environ.copy()
        environment.update(
            {
                "COLIBRI_ARTIFACT_ROOT": str(artifact_root),
                "COLIBRI_EXPERT_CACHE_BUDGET_BYTES": "18874368",
                "COLIBRI_EXPERT_TRACE_OUTPUT": str(trace_path),
                "COLIBRI_TRACE_FIXTURE_ID": fixture_id,
                "COLIBRI_TRACE_WORKLOAD_CLASS": fixture["workload_class"],
                "COLIBRI_TRACE_INPUT_TOKEN_IDS": ",".join(str(value) for value in fixture["token_ids"]),
                "COLIBRI_TRACE_REQUESTED_GENERATION_LENGTH": str(fixture["requested_generation_length"]),
                "COLIBRI_TRACE_SEED": str(fixture["seed"]),
                "COLIBRI_TRACE_DECODING_MODE": fixture["decoding_mode"],
                "COLIBRI_TRACE_KV_CACHE_CAPACITY": str(fixture["kv_cache_capacity"]),
                "COLIBRI_TRACE_INSTRUMENTATION_COMMIT": instrumentation_commit,
            }
        )
        command = [
            str(executable),
            "full_model_validation_tests::m5_2_trace_capture::representative_trace_capture",
            "--exact",
            "--nocapture",
        ]
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"fixture {fixture_id} repeat {repeat} failed with exit code {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        if not trace_path.exists():
            raise RuntimeError(f"fixture {fixture_id} repeat {repeat} did not produce a trace")
        payload = trace_path.read_bytes()
        trace = json.loads(payload.decode("utf-8"))
        validate_trace(trace, fixture)
        digest = sha256_bytes(payload)
        repeat_hashes.append(digest)
        generated = trace["expected_generated_token_ids"]
        if first_bytes is None:
            first_bytes = payload
            first_generated = generated
        else:
            assert payload == first_bytes, f"fixture {fixture_id} traces differ between repeats"
            assert generated == first_generated, f"fixture {fixture_id} generated IDs differ between repeats"

    assert first_bytes is not None and first_generated is not None
    expected = fixture.get("expected_generated_token_ids")
    if expected is not None:
        assert first_generated == expected, f"fixture {fixture_id} generated IDs differ from manifest"
    final_path = root / "models/qwen3-30b-a3b/m5.2-01-traces" / f"{fixture_id}.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        assert final_path.read_bytes() == first_bytes, f"existing canonical trace differs: {final_path}"
    else:
        os.replace(run_dir / f"{fixture_id}-repeat-1.json", final_path)
    return {
        "fixture_id": fixture_id,
        "trace_path": final_path.relative_to(root).as_posix(),
        "trace_sha256": sha256_bytes(first_bytes),
        "repeat_sha256": repeat_hashes,
        "byte_identical_repeat": repeat_hashes[0] == repeat_hashes[1],
        "generated_token_ids": first_generated,
        "record_count": len(json.loads(first_bytes.decode("utf-8")).get("records", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fixture-manifest", type=Path, default=Path("models/qwen3-30b-a3b/m5.2-01-representative-fixture-manifest-v1.json"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--test-executable", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path(r"D:\tmp\colibri-lite-runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--instrumentation-commit", required=True)
    parser.add_argument("--repeat-report", type=Path, default=Path("models/qwen3-30b-a3b/m5.2-01-repeatability-v1.json"))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--retain-failed", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    fixture_manifest_path = (root / args.fixture_manifest).resolve()
    artifact_root = args.artifact_root.resolve()
    executable = args.test_executable.resolve()
    if not artifact_root.is_dir():
        raise SystemExit(f"artifact root does not exist: {artifact_root}")
    if not executable.is_file():
        raise SystemExit(f"test executable does not exist: {executable}")
    manifest = load_json(fixture_manifest_path)
    fixtures = manifest["fixtures"]
    selected = [fixture for fixture in fixtures if not args.only or fixture["fixture_id"] in args.only]
    missing = sorted(set(args.only) - {fixture["fixture_id"] for fixture in selected})
    if missing:
        raise SystemExit(f"unknown fixture IDs: {', '.join(missing)}")

    run_parent = args.run_root.resolve()
    run_dir = run_parent / f"m5.2-01-{args.run_id}"
    if run_dir.exists():
        raise SystemExit(f"run directory already exists; run IDs are not reusable: {run_dir}")
    expected_new_bytes = max(1, len(selected) * 2 * 1_500_000)
    safety_reserve = max(1024**3, expected_new_bytes // 20)
    free_bytes = shutil.disk_usage(run_parent.anchor or run_parent).free
    if free_bytes < expected_new_bytes + safety_reserve:
        raise SystemExit(
            f"disk preflight failed: free={free_bytes} required={expected_new_bytes + safety_reserve}"
        )
    run_dir.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    try:
        for fixture in selected:
            if fixture["fixture_id"] == "tier_a_control":
                trace_path = root / CONTROL_TRACE
                trace_manifest = load_json(root / CONTROL_MANIFEST)
                trace_bytes = trace_path.read_bytes()
                assert sha256_bytes(trace_bytes) == trace_manifest["trace"]["sha256"]
                assert trace_manifest["repeat"]["byte_identical"] is True
                assert trace_manifest["repeat"]["sha256"] == sha256_bytes(trace_bytes)
                trace = json.loads(trace_bytes.decode("utf-8"))
                assert trace["expected_generated_token_ids"] == fixture["expected_generated_token_ids"]
                results.append(
                    {
                        "fixture_id": fixture["fixture_id"],
                        "trace_path": CONTROL_TRACE,
                        "trace_sha256": sha256_bytes(trace_bytes),
                        "repeat_sha256": [trace_manifest["repeat"]["sha256"]] * 2,
                        "repeat_count": 2,
                        "byte_identical_repeat": True,
                        "generated_token_ids": trace["expected_generated_token_ids"],
                        "record_count": len(trace["records"]),
                    }
                )
            else:
                results.append(capture_fixture(root, executable, artifact_root, run_dir, fixture, args.instrumentation_commit))
            results[-1].setdefault("repeat_count", 2)
            print(json.dumps(results[-1], sort_keys=True))
    finally:
        if run_dir.exists() and (args.retain_failed and len(results) != len(selected)):
            print(f"retained failed run directory: {run_dir}")
        elif run_dir.exists():
            shutil.rmtree(run_dir)
    repeat_report_path = (root / args.repeat_report).resolve()
    prior_results: dict[str, dict[str, Any]] = {}
    if repeat_report_path.exists():
        prior_report = load_json(repeat_report_path)
        prior_results = {item["fixture_id"]: item for item in prior_report.get("fixtures", [])}
    prior_results.update({item["fixture_id"]: item for item in results})
    ordered_results = [
        prior_results[fixture["fixture_id"]]
        for fixture in fixtures
        if fixture["fixture_id"] in prior_results
    ]
    repeat_report = {
        "schema": "colibri-qwen3-moe-m5.2-01-repeatability-v1",
        "schema_version": 1,
        "corpus_id": "qwen3-30b-a3b-m5.2-01-representative-expert-traces-v1",
        "trace_schema": EXPECTED_TRACE_SCHEMA,
        "canonical_artifact_root_sha256": EXPECTED_ARTIFACT_ROOT,
        "fixtures": ordered_results,
        "requirements": {
            "capture_count_per_fixture": 2,
            "canonical_content_excludes": ["timestamp", "process_id", "local_filesystem_path", "timing"],
            "ordered_records_compared": True,
            "generated_ids_compared": True,
            "non_timing_counters_compared": True,
        },
        "serialization": "UTF-8 JSON, sorted keys, compact separators, trailing newline; no run ID or timing",
    }
    repeat_report_path.parent.mkdir(parents=True, exist_ok=True)
    repeat_report_path.write_text(json.dumps(repeat_report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(repeat_report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
