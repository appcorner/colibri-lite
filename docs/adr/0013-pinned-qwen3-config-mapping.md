# ADR 0013: Pinned Qwen3-MoE Configuration Mapping

- Status: Accepted
- Date: 2026-07-14
- Milestone: M4.1
- Task: M4.1-03
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

After ADR 0012 made attention head dimension explicit, M4.1-03 must map the
required Hugging Face Qwen3-30B-A3B fields into validated runtime contracts.
The mapping must not imply support for source features that the current
correctness implementation does not execute.

M4.1-03 defines a typed source-to-runtime bridge. JSON parsing and artifact
conversion remain later tasks and do not justify a serialization dependency
here.

## Decision

Add `Qwen3MoeSourceConfig` as the required upstream field set and
`Qwen3MoeConfigMapping` as the validated mapping result. The exact pinned
values are frozen in `PINNED_QWEN3_30B_A3B_CONFIG` and correspond to source
manifest v1.

Required field mapping:

| Hugging Face field | Runtime contract |
| --- | --- |
| `vocab_size` | `ModelConfig::vocabulary_size` |
| `hidden_size` | `ModelConfig::hidden_size` |
| `intermediate_size` | `ModelConfig::intermediate_size` |
| `num_hidden_layers` | `ModelConfig::layer_count` |
| `num_attention_heads` | `ModelConfig::attention_head_count` |
| `num_key_value_heads` | `ModelConfig::key_value_head_count` |
| `head_dim` | `ModelConfig::head_dimension` directly |
| `max_position_embeddings` | model maximum positions |
| `rms_norm_eps` | `Qwen3MoeConfig::rms_norm_epsilon` |
| `rope_theta` | `Qwen3MoeConfig::rope_theta` |
| `num_experts` | `Qwen3MoeConfig::expert_count` |
| `num_experts_per_tok` | `Qwen3MoeConfig::experts_per_token` |
| `moe_intermediate_size` | `Qwen3MoeConfig::moe_intermediate_size` |
| `norm_topk_prob` | routing-weight normalization policy |

`torch_dtype = bfloat16` maps only to `source_data_type`. The correctness
runtime remains F32, permitting a later artifact reader to decode BF16 source
values to F32 without adding BF16 kernels or weakening the oracle.

The mapping validates these pinned execution assumptions:

- sole architecture `Qwen3MoeForCausalLM` and model type `qwen3_moe`;
- SiLU activation, no attention bias, and zero inference dropout;
- no RoPE scaling or sliding-window attention;
- every layer is sparse and no dense-only MLP layers exist;
- token embeddings and LM head are not tied;
- source storage type is BF16.

Unsupported values fail with `RuntimeError::InvalidModelConfig`. Dimension,
head divisibility, and projection overflow errors retain the generic structured
categories from ADR 0012.

## Deliberately separate fields

`max_position_embeddings = 40,960` is retained as the model position limit.
Tokenizer `model_max_length = 131,072` is tokenizer metadata and is not mapped
into `ModelConfig`. Runtime session capacity remains a per-session value bounded
by the model configuration. These three concepts are not collapsed.

Training or presentation fields such as initializer range, auxiliary router
loss coefficient, output-router-logit preference, token IDs, generation
sampling defaults, and chat template are recorded in the source manifest or
tokenizer/generation metadata but do not change inference dimensions.

## Public API surface

- `Qwen3MoeSourceConfig`
- `Qwen3MoeConfigMapping`
- `Qwen3MoeSourceConfig::map_to_f32_runtime`
- `Qwen3MoeConfigMapping::{runtime_config, source_data_type,
  model_max_position_count}`
- `PINNED_QWEN3_30B_A3B_CONFIG`

## Evidence

- The pinned mapping accepts `hidden_size=2048`, heads 32/4, and explicit
  `head_dim=128`.
- Query projection width is 4,096 and KV projection width is 512.
- Source BF16 and runtime F32 types remain independently observable.
- Unsupported source feature tests return named structured errors.
- Zero and overflow source dimensions preserve ADR 0012 error categories.
- Frozen M1/M3 and all workspace regression tests pass unchanged.

## Excluded work

- No JSON parser, full weight download, converter, or tensor-name mapping.
- No BF16 compute kernel, quantization, GGUF, memory mapping, SIMD, or native
  kernel.
