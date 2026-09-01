"""Run the preregistered M6.3-R1.3 paired release-process measurement."""
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

CONTRACT_SHA256 = "57a02d697231350977a4391a3625d2808b97cb394078290db398e2996195584f"
BINARY_SHA256 = "69109985e658918e8b0accf3dacb737daba53b6d6f05fb6535b256379805d541"
CANDIDATE_SHA256 = "777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2"
PARSER_SHA256 = "33ca6008dd39aee452646a4b31285eb1863f71fae6b6dc366f0a7a38586b7ff2"
COLLECTOR_SHA256 = "3282b91e18ce64bb0b69bf9fd2beb12a6f621bf502ab12867f76255126b2aef2"
PROVIDER_SHA256 = "1445d29341f5d9c1765f92fe17a70ab9bcaea6ad8f98c12fcf26a854f2ccf505"
MODEL_MANIFEST_SHA256 = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
PAIR_ORDER = [
    ["candidate", "reference"],
    ["reference", "candidate"],
    ["candidate", "reference"],
    ["reference", "candidate"],
    ["candidate", "reference"],
]
CONDITIONS = ["runtime_cache_cold", "runtime_cache_warm"]

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


def artifacts(artifact_root: Path, candidate: Path, mode: str) -> list[str]:
    dense = artifact_root / "dense" / "dense-f32.bin"
    experts = [artifact_root / "experts" / f"experts-layer-{layer:05d}-of-00048.bin" for layer in range(48)]
    paths = [dense, *experts] if mode == "reference" else [dense, *experts[1:], candidate]
    for path in paths:
        require(path.is_file(), f"missing timed artifact: {path}")
    return [str(path.resolve()) for path in paths]


def artifact_identities(repo: Path, artifact_root: Path, candidate: Path, mode: str) -> list[dict]:
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
    by_path[str(candidate.resolve()).lower()] = {
        "path": str(candidate.resolve()),
        "bytes": 679_477_248,
        "sha256": CANDIDATE_SHA256,
    }
    result = []
    for path_text in artifacts(artifact_root, candidate, mode):
        record = by_path.get(path_text.lower())
        require(record is not None, f"missing frozen identity for timed artifact: {path_text}")
        require(Path(record["path"]).stat().st_size == record["bytes"], f"timed artifact byte length mismatch: {path_text}")
        result.append(record)
    return result


def sample_config(repo: Path, binary: Path, artifact_root: Path, candidate: Path, sample_root: Path, mode: str, condition: str) -> Path:
    captures = sample_root / "captures"
    captures.mkdir()
    environment = {
        "COLIBRI_ARTIFACT_ROOT": str(artifact_root),
        "COLIBRI_EXPERT_CACHE_BUDGET_BYTES": "18874368",
        "COLIBRI_R1_3_MODE": mode,
        "COLIBRI_R1_3_CONDITION": condition,
        "COLIBRI_R1_3_READY": "{run_dir}\\ready.txt",
        "COLIBRI_R1_3_GO": "{run_dir}\\go.txt",
        "COLIBRI_R1_3_METRICS_OUTPUT": "{run_dir}\\metrics.tsv",
    }
    if mode == "candidate":
        environment["COLIBRI_R1_3_CANDIDATE_PATH"] = str(candidate)
    config = {
        "executable": str(binary),
        "argument_string": "m6_3_r1_3_paired_sample --nocapture",
        "working_directory": str(repo),
        "artifacts": artifacts(artifact_root, candidate, mode),
        "artifact_identities": artifact_identities(repo, artifact_root, candidate, mode),
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

def collect_sample(run_dir: Path, condition: str, pair: int, order_index: int, mode: str) -> dict:
    capture = json.loads((run_dir / "capture.json").read_text(encoding="utf-8-sig"))
    require(capture["exit_code"] == 0, "sample process failed")
    require(capture["samples"] >= 1, "sample has no memory observations")
    require(capture["etw"]["status"] == "correlated", "sample ETW is not correlated")
    require(capture.get("handshake", {}).get("etw_after_ready") is True, "sample handshake missing")
    with (run_dir / "metrics.tsv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    require(len(rows) == 1, "sample metrics row count")
    row = rows[0]
    require(row["mode"] == mode and row["condition"] == condition, "sample identity mismatch")
    require(int(row["final_argmax"]) == 0, "sample final argmax mismatch")
    physical = capture["etw"]["physical_read_bytes"]
    require(isinstance(physical, int) and physical >= 0, "physical bytes missing")
    return {
        "condition": condition,
        "pair": pair,
        "order_index": order_index,
        "mode": mode,
        "run_directory": str(run_dir),
        "exit_code": capture["exit_code"],
        "memory_samples": capture["samples"],
        "process_wall_seconds": float(capture["process_wall_seconds"]),
        "working_set_peak_bytes": int(capture["memory"]["peak_working_set_bytes"]),
        "private_bytes_peak": int(capture["memory"]["private_bytes_peak"]),
        "physical_read_bytes": physical,
        "physical_reads_per_token": physical / 2.0,
        "trace_sha256": capture["etw"]["trace_sha256"],
        "etw_status": capture["etw"]["status"],
        "setup_seconds": number(row, "setup_seconds"),
        "timed_wall_seconds": number(row, "timed_wall_seconds"),
        "ttft_seconds": number(row, "ttft_seconds"),
        "prefill_tokens_per_second": number(row, "prefill_tokens_per_second"),
        "decode_tokens_per_second": number(row, "decode_tokens_per_second"),
        "final_argmax": int(row["final_argmax"]),
        "dense_logical_bytes": number(row, "dense_logical_bytes", True),
        "f32_expert_logical_bytes": number(row, "f32_expert_logical_bytes", True),
        "candidate_logical_bytes": number(row, "candidate_logical_bytes", True),
        "total_logical_bytes": number(row, "total_logical_bytes", True),
        "logical_bytes_per_token": number(row, "logical_bytes_per_token"),
        "cache_hits": number(row, "cache_hits", True),
        "cache_misses": number(row, "cache_misses", True),
        "cache_loads": number(row, "cache_loads", True),
        "cache_evictions": number(row, "cache_evictions", True),
        "f32_cache_peak_resident_bytes": number(row, "f32_cache_peak_resident_bytes", True),
        "candidate_peak_packed_expert_bytes": number(row, "candidate_peak_packed_expert_bytes", True),
        "candidate_verification_bytes": number(row, "candidate_verification_bytes", True),
        "kv_cache_bytes": number(row, "kv_cache_bytes", True),
        "vram_bytes": number(row, "vram_bytes", True),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    artifact_root = args.artifact_root.resolve()
    candidate = args.candidate.resolve()
    binary = args.binary.resolve()
    contract = args.contract.resolve()
    output_root = args.output_root.resolve()
    require(sys.platform == "win32", "R1.3 official runner requires Windows")
    require(bool(ctypes.windll.shell32.IsUserAnAdmin()), "R1.3 official runner requires Administrator")
    require(sha256(contract) == CONTRACT_SHA256, "R1.3 contract hash mismatch")
    require(sha256(binary) == BINARY_SHA256, "R1.3 release binary hash mismatch")
    require(candidate.stat().st_size == 679_477_248 and sha256(candidate) == CANDIDATE_SHA256, "R1.3 candidate identity mismatch")
    require(sha256(repo / "scripts" / "parse_m6_3_r1_etw.py") == PARSER_SHA256, "R1.3 parser hash mismatch")
    require(sha256(repo / "scripts" / "capture_m6_3_r1_etw.ps1") == COLLECTOR_SHA256, "R1.3 collector hash mismatch")
    require(sha256(repo / "scripts" / "m6_3_r1_etw-providers.txt") == PROVIDER_SHA256, "R1.3 provider hash mismatch")
    artifacts(artifact_root, candidate, "reference")
    artifacts(artifact_root, candidate, "candidate")
    artifact_identities(repo, artifact_root, candidate, "reference")
    artifact_identities(repo, artifact_root, candidate, "candidate")
    if args.validate_only:
        print(json.dumps({"status": "passed", "contract_sha256": CONTRACT_SHA256, "binary_sha256": BINARY_SHA256}, sort_keys=True))
        return 0

    require(not output_root.exists(), "official output root must not already exist")
    output_root.mkdir(parents=True)
    capture_script = repo / "scripts" / "capture_m6_3_r1_etw.ps1"
    samples: list[dict] = []
    for condition in CONDITIONS:
        for pair_index, modes in enumerate(PAIR_ORDER, start=1):
            for order_index, mode in enumerate(modes, start=1):
                sample_id = f"{condition}-pair-{pair_index:02d}-order-{order_index}-{mode}"
                sample_root = output_root / sample_id
                sample_root.mkdir()
                config = sample_config(repo, binary, artifact_root, candidate, sample_root, mode, condition)
                print(f"R1.3 START {sample_id}", flush=True)
                completed = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(capture_script), "-Config", str(config)],
                    cwd=repo,
                    check=False,
                )
                require(completed.returncode == 0, f"invalid sample {sample_id}: collector exit {completed.returncode}; no automatic retry")
                latest = (sample_root / "captures" / "latest-run.txt").read_text(encoding="ascii").strip()
                run_dir = Path(latest)
                require((run_dir / "capture.json").is_file() and (run_dir / "metrics.tsv").is_file(), f"incomplete sample {sample_id}")
                sample = collect_sample(run_dir, condition, pair_index, order_index, mode)
                samples.append(sample)
                partial = {
                    "schema": "m6.3-r1.3-paired-samples-partial-v1",
                    "contract_sha256": CONTRACT_SHA256,
                    "completed_samples": len(samples),
                    "samples": samples,
                }
                write_json(output_root / "samples.partial.json", partial)
                print(f"R1.3 PASS {sample_id} timed={sample['timed_wall_seconds']:.6f}s physical={sample['physical_read_bytes']}", flush=True)

    require(len(samples) == 20, "official R1.3 sample count")
    final = {
        "schema": "m6.3-r1.3-paired-samples-v1",
        "contract": str(contract),
        "contract_sha256": CONTRACT_SHA256,
        "binary": str(binary),
        "binary_sha256": BINARY_SHA256,
        "candidate": str(candidate),
        "candidate_sha256": CANDIDATE_SHA256,
        "parser_sha256": PARSER_SHA256,
        "collector_sha256": COLLECTOR_SHA256,
        "provider_sha256": PROVIDER_SHA256,
        "candidate_performance_results_seen": True,
        "samples": samples,
    }
    final_path = output_root / "m6.3-r1-3-paired-samples-v1.json"
    write_json(final_path, final)
    print(f"R1.3 COMPLETE {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
