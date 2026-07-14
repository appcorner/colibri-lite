# ADR 0011: Qwen3-30B-A3B Source Contract

- Status: Proposed - review required
- Date: 2026-07-14
- Milestone: M4.1
- Task: M4.1-01

## Context

M4 begins with an immutable upstream source contract. Full weights must not be
downloaded or converted until model identity, license, configuration,
tokenizer, shard inventory, source hashes, and disk requirements are recorded.

The source contract must also be checked against the frozen tiny-model mapping
before implementation. A conflict that changes public configuration or tensor
layout is a mandatory review stop.

## Decision

Pin the following upstream source:

- Model ID: `Qwen/Qwen3-30B-A3B`
- Revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Immutable tree: <https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39>
- Architecture: `Qwen3MoeForCausalLM`
- Model type: `qwen3_moe`
- License: Apache License 2.0 (`Apache-2.0`)

The machine-readable contract is
`models/qwen3-30b-a3b/source-manifest-v1.json`. The revision is a 40-character
commit SHA, not a mutable branch or tag.

## Verified source inventory

The pinned tree contains 26 files:

- 16 Safetensors shards with official Git LFS SHA-256 object IDs;
- `model.safetensors.index.json` with 18,867 tensor mappings across all 16
  shards;
- `config.json` and `generation_config.json`;
- tokenizer files `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and
  `merges.txt`;
- `LICENSE`, `README.md`, and `.gitattributes`.

The index declares 61,064,245,248 bytes of BF16 tensor payload for
30,532,122,624 parameters. The shard files occupy 61,066,575,648 bytes including
Safetensors headers. Index names include 18,432 routed-expert tensors and 48
router tensors; no shared-expert tensor names are present.

All non-weight files were fetched in memory at the pinned revision and hashed
with SHA-256. The 16 weight SHA-256 values come from the official Hugging Face
Git LFS metadata and were not downloaded for local re-hashing. The tokenizer
LFS hash was locally verified against its downloaded bytes. Exact methods and
hashes are recorded per file in the source manifest.

## Disk requirement

A single materialized source snapshot requires exactly 61,084,187,391 bytes
(56.889 GiB). Before a direct source-only download, require at least
70,000,000,000 free bytes (65.193 GiB).

A workflow retaining both a download cache and a materialized copy may require
approximately twice the snapshot size. Require at least 130,000,000,000 free
bytes (121.072 GiB) for that mode. Conversion output and temporary space are not
included and must be calculated after the artifact layout is reviewed.

No full weight shard was downloaded during M4.1-01.

## Configuration review stop

The source contract conflicts with the frozen tiny-model mapping:

- Upstream explicitly sets `head_dim = 128`.
- Upstream has `hidden_size = 2048` and `num_attention_heads = 32`.
- The current public `Qwen3MoeConfig::head_dimension` derives
  `hidden_size / attention_head_count`, which produces 64.

Using the existing derivation would create incorrect Q/K/V projection and
KV-cache dimensions. Resolving this requires a reviewed public configuration
contract change; M4 implementation must not hide the upstream field or alter
the tiny-model oracle.

Two additional mapping decisions are recorded but not resolved here:

- upstream source tensors are BF16 while the correctness runtime computes F32;
- `max_position_embeddings` is 40,960 while tokenizer `model_max_length` is
  131,072, so the model limit remains authoritative pending a RoPE policy.

This ADR does not approve a redesign. Work stops for review before M4.1-02.

## Excluded work

- No full weight download or conversion.
- No GGUF, quantization, memory mapping, SIMD, native kernel, or optimization.
- No artifact format or public Rust contract change.
- No mutable upstream revision.

## Evidence

- Official Hugging Face model API at the immutable revision returned the same
  commit SHA, architecture, model type, Apache-2.0 card metadata, file sizes,
  Git blobs, and LFS hashes.
- The pinned `LICENSE` bytes begin with `Apache License, Version 2.0` and hash
  to `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`.
- Pinned config, tokenizer, index, model card, and repository metadata hashes
  are recorded in the manifest.
- JSON parsing and manifest arithmetic are verified locally before commit.
