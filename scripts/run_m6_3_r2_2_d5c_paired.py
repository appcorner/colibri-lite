"""Run the frozen M6.3-R2.2-D5c paired performance/resource matrix."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

CONTRACT_SHA256 = "34b5a5a032bf6251b7d724c115bad9db44fc11f08092d235155e32c304d8acd3"
D5B_RESULT_SHA256 = "0c742147f967583a735a5a0d8a9dcc2482878204fcf257507532839d40a3872e"
BINARY_SHA256 = "452ab02c0e99ae46034bf431b126f07ee7b003a267506db88d335545b42d71ae"
HARNESS_SHA256 = "6fcbc065d4cf6d16aa2999cb50f9720a151eff3687dfcec012df403f806ccc3c"
LAYER0_SHA256 = "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2"
LAYER24_SHA256 = "d94d12cbea648e2f2911573c893f564254cbde526d60132c600ed24e88317ac2"
LAYER0_BYTES = 679_477_248
LAYER24_BYTES = 1_912_602_624
PARSER_SHA256 = "33ca6008dd39aee452646a4b31285eb1863f71fae6b6dc366f0a7a38586b7ff2"
COLLECTOR_SHA256 = "3282b91e18ce64bb0b69bf9fd2beb12a6f621bf502ab12867f76255126b2aef2"
PROVIDER_SHA256 = "1445d29341f5d9c1765f92fe17a70ab9bcaea6ad8f98c12fcf26a854f2ccf505"
MODEL_MANIFEST_SHA256 = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
REFERENCE_SHA256 = "fa8f1a4f6b30624b78cf8c004f379fdb0633a08a65c102791acc57d0205d54ff"
PAIR_ORDER = [
    ["production_hybrid", "reference_f32"],
    ["reference_f32", "production_hybrid"],
    ["production_hybrid", "reference_f32"],
    ["reference_f32", "production_hybrid"],
    ["production_hybrid", "reference_f32"],
]
FIXTURES = ["short_english", "short_thai"]
CACHE_LABELS = ["first_process_touch", "likely_warm"]
EXPECTED_GENERATED = {"short_english": "0,358", "short_thai": "7360,91"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def number(row: dict[str, str], name: str, integer: bool = False):
    value = int(row[name]) if integer else float(row[name])
    require(math.isfinite(float(value)) and float(value) >= 0.0, f"invalid metric {name}")
    return value


def timed_artifacts(artifact_root: Path, layer0: Path, layer24: Path, mode: str) -> list[str]:
    dense = artifact_root / "dense" / "dense-f32.bin"
    experts = [
        artifact_root / "experts" / f"experts-layer-{layer:05d}-of-00048.bin"
        for layer in range(48)
    ]
    if mode == "reference_f32":
        paths = [dense, *experts]
    else:
        require(mode == "production_hybrid", "invalid D5c mode")
        paths = [dense, *[path for index, path in enumerate(experts) if index not in (0, 24)], layer0, layer24]
    for path in paths:
        require(path.is_file(), f"missing timed artifact: {path}")
    return [str(path.resolve()) for path in paths]


def artifact_identities(repo: Path, artifact_root: Path, layer0: Path, layer24: Path, mode: str) -> list[dict]:
    manifest_path = repo / "models" / "qwen3-30b-a3b" / "model-manifest-v1.json"
    require(sha256(manifest_path) == MODEL_MANIFEST_SHA256, "model manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [manifest["components"]["dense"]["payload"], *manifest["components"]["experts"]["shards"]]
    by_path = {
        str((artifact_root / entry["path"]).resolve()).lower(): {
            "path": str((artifact_root / entry["path"]).resolve()),
            "bytes": int(entry["bytes"]),
            "sha256": entry["sha256"],
        }
        for entry in entries
    }
    by_path[str(layer0.resolve()).lower()] = {
        "path": str(layer0.resolve()), "bytes": LAYER0_BYTES, "sha256": LAYER0_SHA256
    }
    by_path[str(layer24.resolve()).lower()] = {
        "path": str(layer24.resolve()), "bytes": LAYER24_BYTES, "sha256": LAYER24_SHA256
    }
    result = []
    for path_text in timed_artifacts(artifact_root, layer0, layer24, mode):
        record = by_path.get(path_text.lower())
        require(record is not None, f"missing frozen identity for timed artifact: {path_text}")
        require(Path(record["path"]).stat().st_size == record["bytes"], f"timed artifact byte length mismatch: {path_text}")
        result.append(record)
    return result


def sample_config(
    repo: Path,
    binary: Path,
    artifact_root: Path,
    layer0: Path,
    layer24: Path,
    reference: Path,
    sample_root: Path,
    mode: str,
    fixture: str,
    cache_label: str,
) -> Path:
    captures = sample_root / "captures"
    captures.mkdir()
    environment = {
        "COLIBRI_ARTIFACT_ROOT": str(artifact_root),
        "COLIBRI_EXPERT_CACHE_BUDGET_BYTES": "18874368",
        "COLIBRI_R2_2_D5C_MODE": mode,
        "COLIBRI_R2_2_D5C_FIXTURE": fixture,
        "COLIBRI_R2_2_D5C_CACHE_LABEL": cache_label,
        "COLIBRI_R1_2_REFERENCE_PATH": str(reference),
        "COLIBRI_R2_2_D5C_READY": "{run_dir}\\ready.txt",
        "COLIBRI_R2_2_D5C_GO": "{run_dir}\\go.txt",
        "COLIBRI_R2_2_D5C_METRICS_OUTPUT": "{run_dir}\\metrics.tsv",
    }
    if mode == "production_hybrid":
        environment["COLIBRI_R2_2_D5C_LAYER0_PATH"] = str(layer0)
        environment["COLIBRI_R2_2_D5C_LAYER24_PATH"] = str(layer24)
    config = {
        "executable": str(binary),
        "argument_string": "full_model_validation_tests::r2_2_d5c_benchmark_tests::m6_3_r2_2_d5c_paired_sample --exact --nocapture",
        "working_directory": str(repo),
        "artifacts": timed_artifacts(artifact_root, layer0, layer24, mode),
        "artifact_identities": artifact_identities(repo, artifact_root, layer0, layer24, mode),
        "output_directory": str(captures),
        "use_output_directory_as_run": False,
        "wait_for_ready_before_trace": True,
        "ready_marker": "{run_dir}\\ready.txt",
        "go_marker": "{run_dir}\\go.txt",
        "environment": environment,
        "required_environment": list(environment),
    }
    path = sample_root / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def collect_sample(run_dir: Path, fixture: str, cache_label: str, pair: int, order_index: int, mode: str) -> dict:
    capture = json.loads((run_dir / "capture.json").read_text(encoding="utf-8-sig"))
    require(capture["exit_code"] == 0, "sample process failed")
    require(capture["samples"] >= 1, "sample has no memory observations")
    require(capture["etw"]["status"] == "correlated", "sample ETW is not correlated")
    require(capture.get("handshake", {}).get("etw_after_ready") is True, "sample handshake missing")
    with (run_dir / "metrics.tsv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    require(len(rows) == 1, "sample metrics row count")
    row = rows[0]
    require(row["mode"] == mode and row["fixture"] == fixture and row["cache_label"] == cache_label, "sample identity mismatch")
    require(row["generated_ids"] == EXPECTED_GENERATED[fixture], "sample generation guard mismatch")
    return {
        "fixture": fixture,
        "cache_label": cache_label,
        "pair": pair,
        "order_index": order_index,
        "mode": mode,
        "run_directory": str(run_dir),
        "exit_code": capture["exit_code"],
        "memory_samples": capture["samples"],
        "process_wall_seconds": float(capture["process_wall_seconds"]),
        "peak_working_set_bytes": int(capture["memory"]["peak_working_set_bytes"]),
        "peak_private_bytes": int(capture["memory"]["private_bytes_peak"]),
        "etw_status": capture["etw"]["status"],
        "trace_sha256": capture["etw"]["trace_sha256"],
        "setup_seconds": number(row, "setup_seconds"),
        "wall_seconds": number(row, "wall_seconds"),
        "prefill_tokens_per_second": number(row, "prefill_tokens_per_second"),
        "decode_tokens_per_second": number(row, "decode_tokens_per_second"),
        "generated_ids": row["generated_ids"],
        "dense_logical_bytes": number(row, "dense_logical_bytes", True),
        "f32_expert_logical_bytes": number(row, "f32_expert_logical_bytes", True),
        "layer0_candidate_logical_bytes": number(row, "layer0_candidate_logical_bytes", True),
        "layer24_candidate_logical_bytes": number(row, "layer24_candidate_logical_bytes", True),
        "logical_artifact_bytes_read": number(row, "logical_artifact_bytes_read", True),
        "expert_payload_bytes_read": number(row, "expert_payload_bytes_read", True),
        "candidate_artifact_bytes_read": number(row, "candidate_artifact_bytes_read", True),
        "cache_hits": number(row, "cache_hits", True),
        "cache_misses": number(row, "cache_misses", True),
        "cache_loads": number(row, "cache_loads", True),
        "cache_evictions": number(row, "cache_evictions", True),
        "f32_cache_peak_resident_bytes": number(row, "f32_cache_peak_resident_bytes", True),
        "layer0_peak_packed_expert_bytes": number(row, "layer0_peak_packed_expert_bytes", True),
        "layer24_peak_hybrid_expert_bytes": number(row, "layer24_peak_hybrid_expert_bytes", True),
        "kv_cache_bytes": number(row, "kv_cache_bytes", True),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expected_samples() -> list[tuple[str, str, int, int, str]]:
    return [
        (fixture, cache_label, pair_index, order_index, mode)
        for fixture in FIXTURES
        for cache_label in CACHE_LABELS
        for pair_index, modes in enumerate(PAIR_ORDER, start=1)
        for order_index, mode in enumerate(modes, start=1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--layer0", type=Path, required=True)
    parser.add_argument("--layer24", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--d5b-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    artifact_root = args.artifact_root.resolve()
    layer0 = args.layer0.resolve()
    layer24 = args.layer24.resolve()
    reference = args.reference.resolve()
    binary = args.binary.resolve()
    contract = args.contract.resolve()
    d5b_result = args.d5b_result.resolve()
    output_root = args.output_root.resolve()
    require(sys.platform == "win32", "D5c official runner requires Windows")
    if not args.validate_only:
        require(
            bool(ctypes.windll.shell32.IsUserAnAdmin()),
            "D5c official runner requires Administrator",
        )
    require(sha256(contract) == CONTRACT_SHA256, "D5c contract hash mismatch")
    require(sha256(d5b_result) == D5B_RESULT_SHA256, "D5b GO result hash mismatch")
    require(sha256(reference) == REFERENCE_SHA256, "D5c reference hash mismatch")
    require(sha256(binary) == BINARY_SHA256, "D5c release binary hash mismatch")
    require(sha256(repo / "crates" / "clr-qwen3-moe" / "src" / "r2_2_d5c_benchmark_tests.rs") == HARNESS_SHA256, "D5c harness hash mismatch")
    require(layer0.stat().st_size == LAYER0_BYTES and sha256(layer0) == LAYER0_SHA256, "D5c Layer0 identity mismatch")
    require(layer24.stat().st_size == LAYER24_BYTES and sha256(layer24) == LAYER24_SHA256, "D5c Layer24 identity mismatch")
    require(sha256(repo / "scripts" / "parse_m6_3_r1_etw.py") == PARSER_SHA256, "D5c ETW parser hash mismatch")
    require(sha256(repo / "scripts" / "capture_m6_3_r1_etw.ps1") == COLLECTOR_SHA256, "D5c collector hash mismatch")
    require(sha256(repo / "scripts" / "m6_3_r1_etw-providers.txt") == PROVIDER_SHA256, "D5c provider hash mismatch")
    timed_artifacts(artifact_root, layer0, layer24, "reference_f32")
    timed_artifacts(artifact_root, layer0, layer24, "production_hybrid")
    artifact_identities(repo, artifact_root, layer0, layer24, "reference_f32")
    artifact_identities(repo, artifact_root, layer0, layer24, "production_hybrid")
    if args.validate_only:
        print(json.dumps({"status": "passed", "contract_sha256": CONTRACT_SHA256, "binary_sha256": BINARY_SHA256, "expected_samples": 40}, sort_keys=True))
        return 0

    expected = expected_samples()
    require(len(expected) == 40, "D5c frozen sample count")
    samples: list[dict] = []
    if args.resume:
        require(output_root.is_dir(), "resume output root must exist")
        partial_path = output_root / "samples.partial.json"
        require(partial_path.is_file(), "resume requires samples.partial.json")
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        require(partial.get("contract_sha256") == CONTRACT_SHA256, "resume contract hash mismatch")
        samples = list(partial.get("samples", []))
        require(partial.get("completed_samples") == len(samples), "resume partial sample count mismatch")
        require(len(samples) < len(expected), "resume has no remaining samples")
        for sample, identity in zip(samples, expected, strict=False):
            fixture, cache_label, pair_index, order_index, mode = identity
            require(
                (sample.get("fixture"), sample.get("cache_label"), sample.get("pair"), sample.get("order_index"), sample.get("mode"))
                == (fixture, cache_label, pair_index, order_index, mode),
                "resume partial samples are not an exact expected prefix",
            )
    else:
        require(not output_root.exists(), "official D5c output root must not already exist")
        output_root.mkdir(parents=True)

    capture_script = repo / "scripts" / "capture_m6_3_r1_etw.ps1"
    completed_prefix = len(samples)
    for expected_index, (fixture, cache_label, pair_index, order_index, mode) in enumerate(expected):
        if expected_index < completed_prefix:
            continue
        sample_id = f"{fixture}-{cache_label}-pair-{pair_index:02d}-order-{order_index}-{mode}"
        sample_root = output_root / sample_id
        require(not sample_root.exists(), f"sample directory already exists: {sample_id}")
        sample_root.mkdir()
        config = sample_config(repo, binary, artifact_root, layer0, layer24, reference, sample_root, mode, fixture, cache_label)
        print(f"D5c START {sample_id}", flush=True)
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(capture_script), "-Config", str(config)],
            cwd=repo,
            check=False,
        )
        require(completed.returncode == 0, f"invalid sample {sample_id}: collector exit {completed.returncode}; no automatic retry")
        latest = (sample_root / "captures" / "latest-run.txt").read_text(encoding="ascii").strip()
        run_dir = Path(latest)
        require((run_dir / "capture.json").is_file() and (run_dir / "metrics.tsv").is_file(), f"incomplete sample {sample_id}")
        sample = collect_sample(run_dir, fixture, cache_label, pair_index, order_index, mode)
        samples.append(sample)
        partial = {
            "schema": "m6.3-r2.2-d5c-paired-samples-partial-v1",
            "contract_sha256": CONTRACT_SHA256,
            "completed_samples": len(samples),
            "samples": samples,
        }
        write_json(output_root / "samples.partial.json", partial)
        print(f"D5c PASS {sample_id} wall={sample['wall_seconds']:.6f}s ws={sample['peak_working_set_bytes']}", flush=True)

    require(len(samples) == 40, "official D5c sample count")
    final = {
        "schema": "m6.3-r2.2-d5c-paired-samples-v1",
        "contract": str(contract),
        "contract_sha256": CONTRACT_SHA256,
        "d5b_result": str(d5b_result),
        "d5b_result_sha256": D5B_RESULT_SHA256,
        "binary": str(binary),
        "binary_sha256": BINARY_SHA256,
        "harness_sha256": HARNESS_SHA256,
        "layer0": str(layer0),
        "layer0_sha256": LAYER0_SHA256,
        "layer24": str(layer24),
        "layer24_sha256": LAYER24_SHA256,
        "parser_sha256": PARSER_SHA256,
        "collector_sha256": COLLECTOR_SHA256,
        "provider_sha256": PROVIDER_SHA256,
        "performance_results_seen": True,
        "samples": samples,
    }
    final_path = output_root / "m6.3-r2-2-d5c-paired-samples-v1.json"
    write_json(final_path, final)
    print(f"D5c COMPLETE {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
