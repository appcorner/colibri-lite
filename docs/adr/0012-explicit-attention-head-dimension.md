# ADR 0012: Explicit Attention Head Dimension

- Status: Accepted
- Date: 2026-07-14
- Milestone: M4.1
- Prerequisite: M4.1-03

## Context

The frozen tiny model has `hidden_size = attention_heads * head_dim`, so the
initial generic contract derived head dimension as `hidden_size /
attention_heads`. Pinned Qwen3-30B-A3B does not satisfy that relationship:
`hidden_size = 2048`, `attention_heads = 32`, and explicit `head_dim = 128`.

Deriving 64 would invalidate query/key/value projection shapes, rotary
embeddings, attention output layout, and KV-cache accounting. The public model
configuration must represent the upstream dimension without imposing an
architecture relationship that is not generally true.

## Decision

`ModelConfigSpec` requires an explicit `head_dimension`. `ModelConfig` stores
that value and precomputes two checked widths:

- query projection width = `attention_head_count * head_dimension`;
- KV projection width = `key_value_head_count * head_dimension`.

Both helpers return validated `usize` values. Construction rejects zero
required dimensions, query heads not divisible by KV heads, and either checked
projection-width overflow. It does not require hidden size to equal query
projection width or to be divisible by the query-head count.

`Qwen3MoeConfig::head_dimension` delegates to the explicit generic value.
Attention reshapes concatenated head output by query projection width before
the output projection. Weight-shape validation uses the checked helpers. KV
cache construction uses the explicit head dimension and retains checked byte
accounting.

The frozen tiny fixture now records `head_dim = 4` explicitly in its source
configuration and generated Rust constants. Tensor weights, checkpoints,
inputs, and numerical tolerances are unchanged.

## Public contract changes

- Add `ModelConfigSpec::head_dimension`.
- Add `ModelConfig::head_dimension`.
- Add `ModelConfig::query_projection_width`.
- Add `ModelConfig::key_value_projection_width`.
- Remove the invariant that hidden size must be divisible by attention heads.

This is an intentional breaking constructor change approved before M4.1-03.

## Precision and context boundaries

Pinned `torch_dtype = bfloat16` remains source storage metadata. This decision
adds no BF16 compute kernel, quantization, or tolerance change; the first
correctness artifact may decode BF16 values to F32.

Model maximum positions, tokenizer model length, and runtime session capacity
remain separate values. No universal context limit is selected here.

## Evidence

- Frozen M1 full-decoder logits and expert IDs still match the oracle at the
  existing tolerance.
- Frozen M3 greedy and seeded generation sequences remain unchanged.
- Qwen3-30B-A3B dimensions are accepted with query width 4,096 and KV width
  512.
- Zero head dimension and both projection overflow paths return structured
  errors.
- Explicit head dimension drives checked KV-cache byte accounting.
- All standard workspace verification commands pass before commit.

## Excluded work

- No full-model config parser or artifact conversion in this contract patch.
- No BF16 kernel, GGUF, quantization, memory mapping, SIMD, or native kernel.
- No change to frozen tensor data or expected numerical outputs.
