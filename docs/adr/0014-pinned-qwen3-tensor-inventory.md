# ADR 0014: Pinned Qwen3-MoE Tensor Inventory

- Status: Accepted
- Date: 2026-07-14
- Milestone: M4.1
- Task: M4.1-04
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

M4.1-04 must account for every tensor in the pinned Qwen3-30B-A3B
Safetensors index before any weight shard is downloaded or any payload is
decoded. The inventory contract must reject names or shapes the existing
Qwen3-MoE correctness path cannot interpret.

The sources of truth are pinned `config.json` and
`model.safetensors.index.json`, whose SHA-256 hashes are recorded in source
manifest v1. Safetensors metadata headers were inspected with bounded byte
ranges to verify dtype and shape; no tensor payload byte was requested.

## Decision

Add a Qwen-specific, metadata-only tensor inventory validator. It first maps
the source configuration through ADR 0013, then uses checked arithmetic to
derive the complete expected name set and shapes. It rejects duplicate,
missing, unknown, out-of-range, wrong-dtype, wrong-rank, and wrong-shape
entries. Shard indices must be in `0..16`.

All source tensors are BF16 metadata. This decision adds no BF16 computation;
the first correctness conversion may decode source BF16 to runtime F32.

### Naming grammar

Top-level tensors:

```text
model.embed_tokens.weight
model.norm.weight
lm_head.weight
```

For each layer `L` in `0..48`:

```text
model.layers.L.input_layernorm.weight
model.layers.L.post_attention_layernorm.weight
model.layers.L.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
model.layers.L.self_attn.{q_norm,k_norm}.weight
model.layers.L.mlp.gate.weight
```

For each layer `L` and expert `E` in `0..128`:

```text
model.layers.L.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight
```

Names outside this grammar are reported as unknown. Numeric layer and expert
components are parsed before range validation, so out-of-range indices retain
their own structured error categories.

### Shape formulas

Validated pinned dimensions are vocabulary 151,936, hidden width 2,048,
48 layers, 32 attention heads, 4 KV heads, explicit head dimension 128,
128 experts, and expert intermediate width 768. ADR 0012's checked helpers
produce query width 4,096 and KV width 512.

| Role | Count | Expected source shape |
| --- | ---: | --- |
| Token embedding | 1 | `[151936, 2048]` |
| Final norm | 1 | `[2048]` |
| Language-model head | 1 | `[151936, 2048]` |
| Input norm | 48 | `[2048]` |
| Post-attention norm | 48 | `[2048]` |
| Query projection | 48 | `[heads * head_dim, hidden] = [4096, 2048]` |
| Key projection | 48 | `[kv_heads * head_dim, hidden] = [512, 2048]` |
| Value projection | 48 | `[kv_heads * head_dim, hidden] = [512, 2048]` |
| Output projection | 48 | `[hidden, heads * head_dim] = [2048, 4096]` |
| Query norm | 48 | `[head_dim] = [128]` |
| Key norm | 48 | `[head_dim] = [128]` |
| Router | 48 | `[experts, hidden] = [128, 2048]` |
| Expert gate projection | 6,144 | `[moe_intermediate, hidden] = [768, 2048]` |
| Expert up projection | 6,144 | `[moe_intermediate, hidden] = [768, 2048]` |
| Expert down projection | 6,144 | `[hidden, moe_intermediate] = [2048, 768]` |

The checked count formula is:

```text
3 + layers * (9 + experts * 3)
= 3 + 48 * (9 + 128 * 3)
= 18,867
```

## Tied-weight decision

Pinned `tie_word_embeddings` is `false`. Both
`model.embed_tokens.weight` and `lm_head.weight` occur in the source index and
are independently required. The validator does not synthesize or alias either
tensor.

## Architecture review

All 18,867 indexed names are classified; unknown and unsupported counts are
zero. No shared-expert tensor, attention bias, or other architecture-specific
tensor is present. Separate expert `gate_proj` and `up_proj` weights implement
the two existing inputs to the M1 gated expert MLP and require no new numerical
operation or public core contract.

## Public API surface

The following Qwen-specific items are exported from `clr-qwen3-moe`:

- `Qwen3MoeTensorMetadata`
- `Qwen3MoeTensorRole`
- `Qwen3MoeMappedTensor`
- `Qwen3MoeTensorInventory`
- `Qwen3MoeTensorInventoryError`
- `validate_qwen3_moe_tensor_inventory`
- `PINNED_QWEN3_30B_A3B_SHARD_COUNT`

No `clr-core` or `clr-storage` public contract changes. The validator consumes
the already approved `Qwen3MoeSourceConfig` mapping from ADR 0013.

## Evidence

`models/qwen3-30b-a3b/tensor-inventory-summary-v1.json` records:

- pinned config and index hashes;
- exact role counts and shape formulas;
- 18,867 index, header, expected, and classified tensors;
- zero unknown or unsupported tensors;
- 2,330,272 total inspected metadata-header bytes and zero payload bytes;
- per-shard metadata-header sizes, tensor counts, and SHA-256 hashes;
- canonical classified-inventory SHA-256
  `1252b1b9073edc0d414b8424eba388352be6591864f961d20d898e41063a2bd2`.

Offline tests synthesize the complete canonical grammar and verify the valid
inventory plus missing, duplicate, unknown, wrong-rank, wrong-shape,
out-of-range layer, out-of-range expert, wrong-dtype, out-of-range shard, and
checked-count-overflow failures.

## Excluded work

- No weight shard or tensor payload download.
- No tensor payload decoding or artifact conversion.
- No BF16 computation, quantization, GGUF, memory mapping, SIMD, native
  kernels, tokenizer parsing, or performance optimization.
