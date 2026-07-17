#!/usr/bin/env python3
"""Generate and validate the canonical Qwen3-30B-A3B model artifact manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Iterable


MODEL_ID = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
ARCHITECTURE = "Qwen3MoeForCausalLM"
MODEL_TYPE = "qwen3_moe"
LICENSE = "Apache-2.0"
ROOT_MANIFEST_NAME = "model-manifest-v1.json"

EXPECTED_DIMENSIONS = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attention_heads": 32,
    "decoder_sparse_step": 1,
    "experts": 128,
    "experts_per_token": 8,
    "head_dimension": 128,
    "hidden_activation": "silu",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "key_value_heads": 4,
    "key_value_projection_width": 512,
    "layers": 48,
    "mlp_only_layer_count": 0,
    "moe_intermediate_size": 768,
    "normalize_topk_probabilities": True,
    "query_projection_width": 4096,
    "rms_norm_epsilon": 0.000001,
    "rope_scaling": None,
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "vocabulary_size": 151936,
}

EXPECTED_CONTEXT_LIMITS = {
    "model_max_positions": 40960,
    "runtime_session_capacity": None,
    "runtime_session_capacity_policy": (
        "caller-configured per session and validated separately; no model or "
        "tokenizer limit is silently selected as a universal capacity"
    ),
    "tokenizer_declared_max_length": 131072,
}

EXPECTED_RUNTIME = {
    "artifact_reader_format_version": 1,
    "contract_version": 1,
    "expert_payload_contract": "packed F32 gate/up/down selected by layer and expert ID",
    "primary_host": "x86_64-windows",
    "requires_explicit_head_dimension": True,
    "requires_safe_rust": True,
    "runtime": "colibri-lite-rs",
}


class ManifestError(RuntimeError):
    """A deterministic artifact contract or integrity check failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def strict_keys(value: Any, expected: set[str], context: str) -> None:
    require(isinstance(value, dict), f"{context} must be an object")
    actual = set(value)
    require(actual == expected, f"{context} fields differ: expected {sorted(expected)}, got {sorted(actual)}")


def canonical_json(document: Any) -> bytes:
    return (json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read JSON '{path.name}': {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ManifestError(f"cannot hash '{path.name}': {error}") from error
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_relative_path(value: str) -> str:
    require(isinstance(value, str) and value != "", "artifact path must be a non-empty string")
    require("\\" not in value, f"artifact path is not canonical POSIX form: {value}")
    path = PurePosixPath(value)
    require(not path.is_absolute(), f"artifact path must be relative: {value}")
    require(path.parts and all(part not in ("", ".", "..") for part in path.parts), f"artifact path is unsafe: {value}")
    require(path.as_posix() == value, f"artifact path is not canonical: {value}")
    return value


def validate_hash(value: Any, context: str) -> str:
    require(isinstance(value, str) and len(value) == 64, f"{context} must be a 64-digit SHA-256")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ManifestError(f"{context} is not hexadecimal") from error
    require(value == value.lower(), f"{context} must use lowercase hexadecimal")
    return value


def file_record(path: str, byte_length: int, sha256: str) -> dict[str, Any]:
    return {
        "bytes": byte_length,
        "path": canonical_relative_path(path),
        "sha256": validate_hash(sha256, f"{path} hash"),
    }


def actual_small_file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / Path(relative)
    require(path.is_file(), f"required file is missing: {relative}")
    return file_record(relative, path.stat().st_size, sha256_file(path))


def declared_payload_record(root: Path, relative: str, byte_length: int, sha256: str) -> dict[str, Any]:
    path = root / Path(relative)
    require(path.is_file(), f"required payload is missing: {relative}")
    require(path.stat().st_size == byte_length, f"payload size mismatch: {relative}")
    return file_record(relative, byte_length, sha256)


def generate_document(root: Path) -> dict[str, Any]:
    source_path = root / "provenance" / "source-manifest-v1.json"
    dense_path = root / "dense" / "dense-manifest-v1.json"
    expert_path = root / "experts" / "expert-manifest-v1.json"
    tokenizer_path = root / "tokenizer" / "tokenizer-artifact-manifest-v1.json"
    source = read_json(source_path)
    dense = read_json(dense_path)
    experts = read_json(expert_path)
    tokenizer = read_json(tokenizer_path)

    require(source.get("schema_version") == 1, "unsupported source manifest version")
    require(source.get("model", {}).get("id") == MODEL_ID, "source model ID is incompatible")
    require(source["model"].get("revision") == REVISION, "source revision is incompatible")
    require(source["model"].get("architecture") == ARCHITECTURE, "source architecture is incompatible")
    require(source["model"].get("model_type") == MODEL_TYPE, "source model type is incompatible")
    require(source["model"].get("license") == LICENSE, "source license is incompatible")
    require(dense.get("format_version") == 1, "unsupported dense artifact version")
    require(experts.get("format_version") == 1, "unsupported expert artifact version")
    require(tokenizer.get("format_version") == 1, "unsupported tokenizer artifact version")

    config = source.get("config", {})
    dimensions = {
        "attention_bias": config.get("attention_bias"),
        "attention_dropout": config.get("attention_dropout"),
        "attention_heads": config.get("num_attention_heads"),
        "decoder_sparse_step": config.get("decoder_sparse_step"),
        "experts": config.get("num_experts"),
        "experts_per_token": config.get("num_experts_per_tok"),
        "head_dimension": config.get("head_dim"),
        "hidden_activation": config.get("hidden_act"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "key_value_heads": config.get("num_key_value_heads"),
        "key_value_projection_width": config.get("num_key_value_heads") * config.get("head_dim"),
        "layers": config.get("num_hidden_layers"),
        "mlp_only_layer_count": len(config.get("mlp_only_layers", [])),
        "moe_intermediate_size": config.get("moe_intermediate_size"),
        "normalize_topk_probabilities": config.get("norm_topk_prob"),
        "query_projection_width": config.get("num_attention_heads") * config.get("head_dim"),
        "rms_norm_epsilon": config.get("rms_norm_eps"),
        "rope_scaling": config.get("rope_scaling"),
        "rope_theta": config.get("rope_theta"),
        "sliding_window": config.get("sliding_window"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "vocabulary_size": config.get("vocab_size"),
    }
    require(dimensions == EXPECTED_DIMENSIONS, "validated model dimensions changed")

    require(dense.get("model_id") == MODEL_ID and dense.get("model_revision") == REVISION, "dense provenance changed")
    require(experts.get("model_id") == MODEL_ID and experts.get("model_revision") == REVISION, "expert provenance changed")
    require(tokenizer.get("model_id") == MODEL_ID and tokenizer.get("revision") == REVISION, "tokenizer provenance changed")
    require(dense.get("source_dtype") == "BF16" and dense.get("artifact_dtype") == "F32", "dense dtype changed")
    require(experts.get("source_dtype") == "BF16" and experts.get("artifact_dtype") == "F32", "expert dtype changed")
    require(dense.get("endianness") == "little" and experts.get("endianness") == "little", "artifact endianness changed")

    source_record = actual_small_file_record(root, "provenance/source-manifest-v1.json")
    dense_manifest_record = actual_small_file_record(root, "dense/dense-manifest-v1.json")
    dense_payload = dense["artifact"]
    dense_payload_record = declared_payload_record(
        root,
        "dense/" + dense_payload["path"],
        dense_payload["byte_length"],
        dense_payload["sha256"],
    )
    expert_manifest_record = actual_small_file_record(root, "experts/expert-manifest-v1.json")
    expert_shards = []
    for shard in experts["shards"]:
        record = declared_payload_record(
            root,
            "experts/" + shard["path"],
            shard["byte_length"],
            shard["sha256"],
        )
        expert_shards.append({"shard_id": shard["shard_id"], **record})
    ordered_shard_hash = sha256_bytes(b"".join(bytes.fromhex(record["sha256"]) for record in expert_shards))

    tokenizer_manifest_record = actual_small_file_record(root, "tokenizer/tokenizer-artifact-manifest-v1.json")
    tokenizer_files = []
    for record in tokenizer["files"]:
        actual = actual_small_file_record(root, "tokenizer/" + record["path"])
        require(actual["bytes"] == record["bytes"] and actual["sha256"] == record["sha256"], f"tokenizer asset changed: {record['path']}")
        tokenizer_files.append(actual)

    dense_bytes = dense_manifest_record["bytes"] + dense_payload_record["bytes"]
    expert_bytes = expert_manifest_record["bytes"] + sum(record["bytes"] for record in expert_shards)
    tokenizer_bytes = tokenizer_manifest_record["bytes"] + sum(record["bytes"] for record in tokenizer_files)
    component_bytes = source_record["bytes"] + dense_bytes + expert_bytes + tokenizer_bytes
    required_file_count = 1 + 2 + 1 + len(expert_shards) + 1 + len(tokenizer_files)

    document = {
        "architecture": ARCHITECTURE,
        "artifact_format": "colibri-lite-model",
        "components": {
            "dense": {
                "bytes": dense_bytes,
                "format_version": dense["format_version"],
                "manifest": dense_manifest_record,
                "payload": dense_payload_record,
                "tensor_count": len(dense["tensors"]),
            },
            "experts": {
                "bytes": expert_bytes,
                "format_version": experts["format_version"],
                "logical_expert_count": len(experts["experts"]),
                "manifest": expert_manifest_record,
                "ordered_shard_set_sha256": ordered_shard_hash,
                "shard_count": len(expert_shards),
                "shard_policy": experts["shard_policy"],
                "shards": expert_shards,
                "source_tensor_count": len(experts["experts"]) * 3,
            },
            "tokenizer": {
                "added_token_count": tokenizer["tokenizer"]["added_token_count"],
                "base_vocabulary_size": tokenizer["tokenizer"]["base_vocabulary_size"],
                "bytes": tokenizer_bytes,
                "chat_template": tokenizer["chat_template"],
                "file_count": len(tokenizer_files),
                "files": tokenizer_files,
                "format_version": tokenizer["format_version"],
                "manifest": tokenizer_manifest_record,
                "tokenizer_class": tokenizer["tokenizer"]["class"],
                "tokenizer_size_with_added_tokens": tokenizer["tokenizer"]["tokenizer_size_with_added_tokens"],
            },
        },
        "context_limits": EXPECTED_CONTEXT_LIMITS,
        "dimensions": dimensions,
        "dtypes": {"compute": "F32", "source": "BF16", "storage": "F32"},
        "endianness": "little",
        "format_version": 1,
        "inventory": {
            "component_bytes": component_bytes,
            "dense_tensor_count": len(dense["tensors"]),
            "expert_shard_count": len(expert_shards),
            "expert_source_tensor_count": len(experts["experts"]) * 3,
            "logical_expert_count": len(experts["experts"]),
            "required_file_count": required_file_count,
            "tokenizer_file_count": len(tokenizer_files),
        },
        "license": LICENSE,
        "model_id": MODEL_ID,
        "model_type": MODEL_TYPE,
        "revision": REVISION,
        "runtime_compatibility": EXPECTED_RUNTIME,
        "source_contract": source_record,
    }
    validate_root_document(document)
    return document


ROOT_KEYS = {
    "architecture",
    "artifact_format",
    "components",
    "context_limits",
    "dimensions",
    "dtypes",
    "endianness",
    "format_version",
    "inventory",
    "license",
    "model_id",
    "model_type",
    "revision",
    "runtime_compatibility",
    "source_contract",
}
FILE_KEYS = {"bytes", "path", "sha256"}


def validate_file_record(record: Any, context: str) -> dict[str, Any]:
    strict_keys(record, FILE_KEYS, context)
    canonical_relative_path(record["path"])
    require(isinstance(record["bytes"], int) and record["bytes"] >= 0, f"{context} byte length is invalid")
    validate_hash(record["sha256"], f"{context} hash")
    return record


def validate_root_document(document: Any) -> list[dict[str, Any]]:
    strict_keys(document, ROOT_KEYS, "root manifest")
    require(document["artifact_format"] == "colibri-lite-model", "unsupported artifact format")
    require(document["format_version"] == 1, "unsupported root artifact version")
    require(document["architecture"] == ARCHITECTURE, "incompatible architecture")
    require(document["model_type"] == MODEL_TYPE, "incompatible model type")
    require(document["model_id"] == MODEL_ID, "incompatible model ID")
    require(document["revision"] == REVISION, "incompatible model revision")
    require(document["license"] == LICENSE, "incompatible license metadata")
    require(document["dtypes"] == {"compute": "F32", "source": "BF16", "storage": "F32"}, "unsupported dtype contract")
    require(document["endianness"] == "little", "unsupported endianness")
    require(document["dimensions"] == EXPECTED_DIMENSIONS, "incompatible model dimensions")
    require(document["context_limits"] == EXPECTED_CONTEXT_LIMITS, "incompatible context-limit metadata")
    require(document["runtime_compatibility"] == EXPECTED_RUNTIME, "incompatible runtime requirements")

    source = validate_file_record(document["source_contract"], "source contract")
    components = document["components"]
    strict_keys(components, {"dense", "experts", "tokenizer"}, "components")
    dense = components["dense"]
    strict_keys(dense, {"bytes", "format_version", "manifest", "payload", "tensor_count"}, "dense component")
    require(dense["format_version"] == 1, "unsupported dense artifact version")
    require(dense["tensor_count"] == 435, "dense tensor inventory is incomplete")
    dense_manifest = validate_file_record(dense["manifest"], "dense manifest")
    dense_payload = validate_file_record(dense["payload"], "dense payload")
    require(dense["bytes"] == dense_manifest["bytes"] + dense_payload["bytes"], "dense component bytes mismatch")

    experts = components["experts"]
    strict_keys(
        experts,
        {
            "bytes",
            "format_version",
            "logical_expert_count",
            "manifest",
            "ordered_shard_set_sha256",
            "shard_count",
            "shard_policy",
            "shards",
            "source_tensor_count",
        },
        "expert component",
    )
    require(experts["format_version"] == 1, "unsupported expert artifact version")
    require(experts["logical_expert_count"] == 6144, "expert inventory is incomplete")
    require(experts["source_tensor_count"] == 18432, "expert tensor inventory is incomplete")
    require(experts["shard_count"] == 48 and len(experts["shards"]) == 48, "expert shard inventory is incomplete")
    require(experts["shard_policy"] == "one container per selected layer", "expert shard policy changed")
    expert_manifest = validate_file_record(experts["manifest"], "expert manifest")
    shard_records = []
    shard_ids = []
    for index, shard in enumerate(experts["shards"]):
        strict_keys(shard, FILE_KEYS | {"shard_id"}, f"expert shard {index}")
        require(isinstance(shard["shard_id"], int), f"expert shard {index} ID is invalid")
        shard_ids.append(shard["shard_id"])
        base_record = {key: shard[key] for key in FILE_KEYS}
        validate_file_record(base_record, f"expert shard {index}")
        shard_records.append(shard)
    require(shard_ids == list(range(48)), "expert shard IDs are missing, duplicated, or out of order")
    validate_hash(experts["ordered_shard_set_sha256"], "ordered expert shard-set hash")
    expected_set_hash = sha256_bytes(b"".join(bytes.fromhex(record["sha256"]) for record in shard_records))
    require(experts["ordered_shard_set_sha256"] == expected_set_hash, "ordered expert shard-set hash mismatch")
    require(experts["bytes"] == expert_manifest["bytes"] + sum(record["bytes"] for record in shard_records), "expert component bytes mismatch")

    tokenizer = components["tokenizer"]
    strict_keys(
        tokenizer,
        {
            "added_token_count",
            "base_vocabulary_size",
            "bytes",
            "chat_template",
            "file_count",
            "files",
            "format_version",
            "manifest",
            "tokenizer_class",
            "tokenizer_size_with_added_tokens",
        },
        "tokenizer component",
    )
    require(tokenizer["format_version"] == 1, "unsupported tokenizer artifact version")
    require(tokenizer["tokenizer_class"] == "Qwen2Tokenizer", "incompatible tokenizer class")
    require(tokenizer["base_vocabulary_size"] == 151643, "tokenizer base vocabulary changed")
    require(tokenizer["added_token_count"] == 26, "tokenizer added-token inventory changed")
    require(tokenizer["tokenizer_size_with_added_tokens"] == 151669, "tokenizer size changed")
    require(tokenizer["file_count"] == 4 and len(tokenizer["files"]) == 4, "tokenizer file inventory is incomplete")
    strict_keys(tokenizer["chat_template"], {"bytes", "characters", "preserved", "rendering_implemented", "sha256", "source_path"}, "chat-template metadata")
    require(tokenizer["chat_template"]["preserved"] is True, "chat template is not preserved")
    require(tokenizer["chat_template"]["rendering_implemented"] is False, "chat rendering is outside M4.1")
    validate_hash(tokenizer["chat_template"]["sha256"], "chat-template hash")
    tokenizer_manifest = validate_file_record(tokenizer["manifest"], "tokenizer manifest")
    tokenizer_files = [validate_file_record(record, f"tokenizer file {index}") for index, record in enumerate(tokenizer["files"])]
    require(tokenizer["bytes"] == tokenizer_manifest["bytes"] + sum(record["bytes"] for record in tokenizer_files), "tokenizer component bytes mismatch")

    records = [source, dense_manifest, dense_payload, expert_manifest, *shard_records, tokenizer_manifest, *tokenizer_files]
    paths = [record["path"] for record in records]
    require(len(paths) == len(set(paths)), "required artifact paths are duplicated")
    inventory = document["inventory"]
    strict_keys(
        inventory,
        {
            "component_bytes",
            "dense_tensor_count",
            "expert_shard_count",
            "expert_source_tensor_count",
            "logical_expert_count",
            "required_file_count",
            "tokenizer_file_count",
        },
        "artifact inventory",
    )
    require(inventory["required_file_count"] == len(records), "required file count mismatch")
    require(inventory["component_bytes"] == sum(record["bytes"] for record in records), "component byte total mismatch")
    require(inventory["dense_tensor_count"] == 435, "dense inventory count mismatch")
    require(inventory["expert_shard_count"] == 48, "expert shard count mismatch")
    require(inventory["logical_expert_count"] == 6144, "logical expert count mismatch")
    require(inventory["expert_source_tensor_count"] == 18432, "expert tensor count mismatch")
    require(inventory["tokenizer_file_count"] == 4, "tokenizer file count mismatch")
    return records


def cross_validate_components(root: Path, document: dict[str, Any]) -> None:
    dense = read_json(root / document["components"]["dense"]["manifest"]["path"])
    strict_keys(dense, {"artifact", "artifact_dtype", "endianness", "format_version", "model_id", "model_revision", "source_dtype", "tensors"}, "dense component manifest")
    require(dense["format_version"] == 1, "unsupported dense component version")
    require(dense["model_id"] == MODEL_ID and dense["model_revision"] == REVISION, "dense component provenance mismatch")
    require(len(dense["tensors"]) == 435, "dense component tensor count mismatch")
    root_dense = document["components"]["dense"]["payload"]
    require(root_dense["path"] == "dense/" + dense["artifact"]["path"], "dense payload path mismatch")
    require(root_dense["bytes"] == dense["artifact"]["byte_length"] and root_dense["sha256"] == dense["artifact"]["sha256"], "dense payload metadata mismatch")

    experts = read_json(root / document["components"]["experts"]["manifest"]["path"])
    strict_keys(experts, {"artifact_dtype", "endianness", "experts", "format_version", "model_id", "model_revision", "shard_policy", "shards", "source_dtype"}, "expert component manifest")
    require(experts["format_version"] == 1, "unsupported expert component version")
    require(experts["model_id"] == MODEL_ID and experts["model_revision"] == REVISION, "expert component provenance mismatch")
    require(len(experts["experts"]) == 6144, "expert component inventory mismatch")
    expert_keys = {(record["layer"], record["expert"]) for record in experts["experts"]}
    require(len(expert_keys) == 6144, "expert component contains duplicate logical experts")
    require(expert_keys == {(layer, expert) for layer in range(48) for expert in range(128)}, "expert component coverage mismatch")
    root_shards = document["components"]["experts"]["shards"]
    require(len(experts["shards"]) == len(root_shards), "expert component shard count mismatch")
    for source, root_record in zip(experts["shards"], root_shards, strict=True):
        require(root_record["shard_id"] == source["shard_id"], "expert shard ID mismatch")
        require(root_record["path"] == "experts/" + source["path"], "expert shard path mismatch")
        require(root_record["bytes"] == source["byte_length"] and root_record["sha256"] == source["sha256"], "expert shard metadata mismatch")

    tokenizer = read_json(root / document["components"]["tokenizer"]["manifest"]["path"])
    require(tokenizer.get("format_version") == 1, "unsupported tokenizer component version")
    require(tokenizer.get("model_id") == MODEL_ID and tokenizer.get("revision") == REVISION, "tokenizer component provenance mismatch")
    root_files = document["components"]["tokenizer"]["files"]
    require(len(tokenizer["files"]) == len(root_files), "tokenizer component file count mismatch")
    for source, root_record in zip(tokenizer["files"], root_files, strict=True):
        require(root_record["path"] == "tokenizer/" + source["path"], "tokenizer file path mismatch")
        require(root_record["bytes"] == source["bytes"] and root_record["sha256"] == source["sha256"], "tokenizer file metadata mismatch")

    source = read_json(root / document["source_contract"]["path"])
    require(source.get("schema_version") == 1, "unsupported source component version")
    require(source.get("model", {}).get("id") == MODEL_ID, "source component model ID mismatch")
    require(source["model"].get("revision") == REVISION, "source component revision mismatch")
    require(source["model"].get("architecture") == ARCHITECTURE, "source component architecture mismatch")
    require(source["model"].get("model_type") == MODEL_TYPE, "source component model type mismatch")


def validate_artifact(root: Path, full_integrity: bool) -> dict[str, Any]:
    manifest_path = root / ROOT_MANIFEST_NAME
    try:
        raw_manifest = manifest_path.read_bytes()
    except OSError as error:
        raise ManifestError(f"root manifest is missing: {error}") from error
    document = read_json(manifest_path)
    require(raw_manifest == canonical_json(document), "root manifest is not canonical JSON")
    records = validate_root_document(document)
    incomplete = [path for path in root.rglob("*") if path.is_file() and "incomplete" in path.name.lower()]
    require(not incomplete, f"incomplete temporary output exists: {incomplete[0].name}" if incomplete else "")

    metadata_paths = {
        document["source_contract"]["path"],
        document["components"]["dense"]["manifest"]["path"],
        document["components"]["experts"]["manifest"]["path"],
        document["components"]["tokenizer"]["manifest"]["path"],
        *(record["path"] for record in document["components"]["tokenizer"]["files"]),
    }
    hashed_bytes = len(raw_manifest)
    hashed_files = 1
    for record in records:
        path = root / Path(record["path"])
        require(path.is_file(), f"required artifact file is missing or renamed: {record['path']}")
        require(path.stat().st_size == record["bytes"], f"artifact size mismatch: {record['path']}")
        if full_integrity or record["path"] in metadata_paths:
            require(sha256_file(path) == record["sha256"], f"artifact hash mismatch: {record['path']}")
            hashed_bytes += record["bytes"]
            hashed_files += 1
    cross_validate_components(root, document)
    return {
        "artifact_format": document["artifact_format"],
        "component_bytes": document["inventory"]["component_bytes"],
        "full_integrity": full_integrity,
        "hashed_bytes": hashed_bytes,
        "hashed_files": hashed_files,
        "required_files": document["inventory"]["required_file_count"],
        "root_manifest_bytes": len(raw_manifest),
        "root_manifest_sha256": sha256_bytes(raw_manifest),
        "status": "passed",
    }


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    require(not path.exists(), f"output already exists: {path.name}")
    require(not temporary.exists(), f"incomplete output already exists: {temporary.name}")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ManifestError(f"cannot commit root manifest: {error}") from error


def command_generate(root: Path, output: Path) -> dict[str, Any]:
    document = generate_document(root)
    payload = canonical_json(document)
    write_atomic(output, payload)
    return {
        "bytes": len(payload),
        "output": output.name,
        "sha256": sha256_bytes(payload),
        "status": "generated",
    }


def usage() -> str:
    return (
        "usage:\n"
        "  model_artifact_manifest.py generate <artifact-root> <output-manifest>\n"
        "  model_artifact_manifest.py validate-metadata <artifact-root>\n"
        "  model_artifact_manifest.py validate-full <artifact-root>"
    )


def main(arguments: Iterable[str] | None = None) -> int:
    args = list(sys.argv[1:] if arguments is None else arguments)
    try:
        if len(args) == 3 and args[0] == "generate":
            result = command_generate(Path(args[1]).resolve(), Path(args[2]).resolve())
        elif len(args) == 2 and args[0] in ("validate-metadata", "validate-full"):
            result = validate_artifact(Path(args[1]).resolve(), args[0] == "validate-full")
        else:
            print(usage(), file=sys.stderr)
            return 2
    except ManifestError as error:
        print(f"artifact manifest error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
