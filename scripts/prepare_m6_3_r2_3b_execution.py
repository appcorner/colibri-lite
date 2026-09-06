#!/usr/bin/env python3
"""Freeze the committed R2.3b machinery and one release test binary identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

FILES = (
    "crates/clr-qwen3-moe/src/r2_3b_characterization_tests.rs",
    "scripts/build_m6_3_r2_3b_artifact.py",
    "scripts/run_m6_3_r2_3b.py",
    "scripts/validate_m6_3_r2_3b.py",
)
CONTRACT = "models/qwen3-30b-a3b/m6.3-r2-3-implementation-contract-v1.json"
REFERENCE = "models/qwen3-30b-a3b/m6.3-r2-3a-f32-four-token-reference-v1.tsv"
CONTRACT_SHA256 = "97dd33b5cdb576d1ebcf9b3b7661f5462c912be40c7485c7e6bd7f8773d7771b"
REFERENCE_SHA256 = "9f8de6841ff883c062c2ed8387dba93631c9de0565943f1de2bb1a27f0fada1e"
ROOT_MANIFEST_SHA256 = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    binary = args.binary.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("execution manifest output must be new")
    if not binary.is_file():
        raise SystemExit("release test binary is missing")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise SystemExit("worktree must be clean before freezing execution identity")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    file_records = []
    for relative in FILES:
        path = repo / relative
        if not path.is_file():
            raise SystemExit(f"required machinery file is missing: {relative}")
        file_records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    if sha256(repo / CONTRACT) != CONTRACT_SHA256:
        raise SystemExit("implementation contract SHA-256 mismatch")
    if sha256(repo / REFERENCE) != REFERENCE_SHA256:
        raise SystemExit("R2.3a reference SHA-256 mismatch")
    document = {
        "schema": "colibri-m6.3-r2.3b-execution-manifest-v1",
        "schema_version": 1,
        "machinery_commit": commit,
        "files": file_records,
        "release_test_binary": {
            "name": binary.name,
            "bytes": binary.stat().st_size,
            "sha256": sha256(binary),
        },
        "identities": {
            "implementation_contract_path": CONTRACT,
            "implementation_contract_sha256": CONTRACT_SHA256,
            "r2_3a_reference_path": REFERENCE,
            "r2_3a_reference_sha256": REFERENCE_SHA256,
            "canonical_root_manifest_sha256": ROOT_MANIFEST_SHA256,
        },
        "grid": {
            "layers": list(range(1, 24)) + list(range(25, 47)),
            "groups": [32, 16, 8],
            "subsets": ["gate_up_down", "gate_up", "gate_down", "up_down", "gate", "up", "down"],
            "candidate_count": 945,
        },
        "gates": {
            "same_input_max_abs": 0.001,
            "sequence_aware_bilingual_max_abs": 0.001,
            "generated_tokens_propagated": 1,
            "automatic_retry": False,
        },
        "test_filter": "full_model_validation_tests::r2_3b_characterization_tests::m6_3_r2_3b_characterize_layer_group_candidates",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "frozen", "manifest": str(output), "binary_sha256": document["release_test_binary"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
