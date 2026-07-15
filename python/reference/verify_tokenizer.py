#!/usr/bin/env python3
"""Verify the pinned Qwen3 tokenizer entirely from committed local assets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import transformers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "models" / "qwen3-30b-a3b"
MANIFEST_NAME = "tokenizer-artifact-manifest-v1.json"
REFERENCE_NAME = "tokenizer-reference-v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def verify(artifact_root: Path) -> dict[str, Any]:
    manifest = load_json(artifact_root / MANIFEST_NAME)
    reference = load_json(artifact_root / REFERENCE_NAME)

    require(manifest["format_version"] == 1, "unsupported tokenizer manifest version")
    require(reference["schema_version"] == 1, "unsupported tokenizer reference version")
    require(
        manifest["model_id"] == reference["model_id"]
        and manifest["revision"] == reference["revision"],
        "reference provenance does not match tokenizer manifest",
    )

    total_asset_bytes = 0
    for record in manifest["files"]:
        path = artifact_root / record["path"]
        require(path.is_file(), f"missing tokenizer file: {record['path']}")
        require(path.stat().st_size == record["bytes"], f"wrong size: {record['path']}")
        require(sha256(path) == record["sha256"], f"wrong SHA-256: {record['path']}")
        total_asset_bytes += record["bytes"]
    require(total_asset_bytes == manifest["canonical_asset_bytes"], "artifact size changed")

    tokenizer_json = load_json(artifact_root / "tokenizer.json")
    tokenizer_config = load_json(artifact_root / "tokenizer_config.json")
    vocab = load_json(artifact_root / "vocab.json")
    merges = (artifact_root / "merges.txt").read_text(encoding="utf-8").splitlines()
    contract = manifest["tokenizer"]

    require(tokenizer_json["model"]["type"] == "BPE", "source model is not BPE")
    require(tokenizer_json["normalizer"] == {"type": "NFC"}, "normalizer changed")
    require(len(tokenizer_json["model"]["vocab"]) == contract["base_vocabulary_size"], "tokenizer.json vocabulary changed")
    require(len(tokenizer_json["model"]["merges"]) == contract["merge_count"], "tokenizer.json merge count changed")
    require(len(vocab) == contract["base_vocabulary_size"], "vocabulary count changed")
    require(min(vocab.values()) == 0, "vocabulary no longer starts at zero")
    require(
        max(vocab.values()) == contract["base_vocabulary_size"] - 1,
        "base vocabulary IDs are not contiguous",
    )
    require(merges[0] == "#version: 0.2", "merge file header changed")
    require(len([line for line in merges[1:] if line]) == contract["merge_count"], "merge count changed")
    require(
        tokenizer_json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        == contract["pretokenizer_regex"],
        "pretokenizer regex changed",
    )
    byte_level = tokenizer_json["pre_tokenizer"]["pretokenizers"][1]
    require(byte_level["type"] == "ByteLevel", "byte-level pretokenizer changed")
    require(byte_level["add_prefix_space"] is False, "prefix-space policy changed")
    require(byte_level["trim_offsets"] is False, "offset trimming policy changed")
    require(tokenizer_json["decoder"]["type"] == "ByteLevel", "decoder changed")
    source_added = [
        {"id": record["id"], "content": record["content"], "special": record["special"]}
        for record in tokenizer_json["added_tokens"]
    ]
    require(source_added == contract["added_tokens"], "added-token metadata changed")

    require(transformers.__version__ == reference["reference"]["transformers"], "Transformers version changed")
    tokenizer = AutoTokenizer.from_pretrained(
        artifact_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    require(type(tokenizer).__name__ == contract["class"], "tokenizer class changed")
    require(tokenizer.is_fast is True, "expected the Rust tokenizers backend")
    require(tokenizer.vocab_size == contract["base_vocabulary_size"], "runtime base vocabulary changed")
    require(len(tokenizer) == contract["tokenizer_size_with_added_tokens"], "runtime tokenizer size changed")
    require(tokenizer.model_max_length == manifest["limits"]["tokenizer_declared_max_length"], "tokenizer limit changed")

    expected_added = {record["content"]: record["id"] for record in contract["added_tokens"]}
    require(tokenizer.get_added_vocab() == expected_added, "added-token mapping changed")
    require(tokenizer.all_special_ids == contract["all_special_token_ids"], "special-token IDs changed")
    special_ids = manifest["special_token_ids"]
    require(tokenizer.bos_token_id == special_ids["tokenizer_bos"], "tokenizer BOS changed")
    require(tokenizer.eos_token_id == special_ids["tokenizer_eos"], "tokenizer EOS changed")
    require(tokenizer.pad_token_id == special_ids["tokenizer_pad"], "tokenizer PAD changed")
    require(tokenizer.unk_token_id == special_ids["tokenizer_unk"], "tokenizer UNK changed")

    template = tokenizer_config["chat_template"]
    template_bytes = template.encode("utf-8")
    chat = manifest["chat_template"]
    require(len(template) == chat["characters"], "chat-template character count changed")
    require(len(template_bytes) == chat["bytes"], "chat-template byte count changed")
    require(hashlib.sha256(template_bytes).hexdigest() == chat["sha256"], "chat-template hash changed")

    encode_matches = 0
    decode_matches = 0
    round_trip_matches = 0
    for case in reference["cases"]:
        token_ids = tokenizer.encode(case["text"], add_special_tokens=False)
        require(token_ids == case["token_ids"], f"token IDs changed for {case['name']}")
        encode_matches += 1
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        require(decoded == case["decoded"], f"decoded text changed for {case['name']}")
        decode_matches += 1
        require((decoded == case["text"]) == case["exact_round_trip"], f"round-trip policy changed for {case['name']}")
        round_trip_matches += int(decoded == case["text"])

    return {
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "transformers": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "offline": True,
        "file_hashes_verified": len(manifest["files"]),
        "reference_cases": len(reference["cases"]),
        "exact_encode_matches": encode_matches,
        "exact_decode_matches": decode_matches,
        "exact_round_trips": round_trip_matches,
    }


def main() -> int:
    artifact_root = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else DEFAULT_ARTIFACT_ROOT
    try:
        result = verify(artifact_root)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"tokenizer verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
