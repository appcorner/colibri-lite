# ADR 0024: Selected Expert Intermediate Validation Budgets

- Status: Accepted provisionally for M4.2-03
- Date: 2026-07-16
- Milestone: M4.2
- Task: M4.2-03
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

M4.2-02 proved the genuine storage-aware path through the Layer-47 router.
M4.2-03 must show that representative matching expert and block outputs are
produced by matching gate, up, activation, product, down, weighting, and
aggregation computations rather than compensating errors.

The compact deterministic selection is:

| Layer | Token | Highest-rank expert | Lowest top-k expert |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 62 | 91 |
| 1 | 0 | 68 | 127 |
| 24 | 1 | 85 | 8 |
| 47 | 0 | 54 | 36 |

Layer 24 uses token 1 because BF16 and F32 have identical complete top-k
membership there. The selection spans expert IDs 8 through 127, source shards
0, 7, 14, and 15, both routing-rank extremes, and distinct routing weights.

The initial reference recomputation stopped at Layer 0 because evaluating one
expert occurrence in isolation changed the Transformers matrix-operation batch
shape. A same-input diagnostic proved separate and concatenated gate/up calls
were bit-identical. Replaying all occurrences for each selected expert restored
the genuine established batch shape and reproduced the frozen Layer 0, 1, and
24 MoE and block tensors exactly.

## Decision

Add a test-only scalar trace beside `expert_mlp`. The normal expert output is
still computed by the unchanged runtime function. A trace is computed only for
the eight selected occurrences, and its down projection must be bit-identical
to that normal output before cross-runtime comparison.

Freeze per-layer, per-checkpoint budgets from one characterization. Except for
routing weights, each budget is three times the maximum observed F32-vs-Rust
error in that layer plus an operation-specific absolute guard:

| Checkpoint | Guard |
| --- | ---: |
| Expert input, activation | `5e-7` |
| Gate/up projection | `2e-6` |
| Product, down, weighted output, aggregation, residual | `1e-6` |

Routing weights retain the already approved propagated budgets from ADR 0023.
This accounts separately for incoming post-normalization drift, projection
accumulation variance, activation sensitivity, multiplication, down-projection
amplification, routing-weight variance, ascending-expert aggregation, and the
final residual add.

| Layer | Input | Gate | Up | Activation | Product | Down |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `7.1161e-6` | `7.0068e-6` | `7.7220e-6` | `3.0928e-6` | `6.7220e-6` | `2.2964e-6` |
| 1 | `8.3678e-6` | `3.6332e-5` | `2.4888e-5` | `3.4832e-5` | `8.2497e-4` | `1.6489e-3` |
| 24 | `1.2659e-5` | `1.9881e-5` | `1.5590e-5` | `9.4407e-6` | `9.2254e-6` | `4.0175e-6` |
| 47 | `2.2938e-4` | `1.1930e-4` | `7.6387e-5` | `1.1780e-4` | `6.6047e-4` | `1.4086e-3` |

| Layer | Routing | Weighted | Aggregate | Residual/final block |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `5.8220e-6` | `1.6258e-6` | `2.1176e-6` | `3.5034e-6` |
| 1 | `6.2989e-6` | `1.0539e-3` | `1.0539e-3` | `1.0539e-3` |
| 24 | `1.8697e-5` | `2.7881e-6` | `3.1458e-6` | `3.3902e-5` |
| 47 | `1.5120e-5` | `2.9855e-4` | `1.0310e-3` | `2.1067e-3` |

## Consequences

- Every selected projection identity, tensor shape, orientation, source range,
  artifact range, routing association, and aggregation membership is exact.
- The normal `ExpertStore` path executes complete Layers 0-47 and loads only
  selected experts. Layer 47 stops after its final block output.
- Final model RMSNorm, LM head, vocabulary logits, sampling, and generation
  remain unreachable.
- RMSNorm, router ordering, activation semantics, expert layout, F32 dtype,
  cache policy, and public APIs are unchanged.
- These budgets apply only to the frozen M4.2-03 cases and are not general
  runtime tolerances.

## Evidence

- `models/qwen3-30b-a3b/m4.2-03-intermediate-structure-v1.tsv`
- `models/qwen3-30b-a3b/m4.2-03-transformers-bf16-intermediate-v1.safetensors`
- `models/qwen3-30b-a3b/m4.2-03-transformers-f32-intermediate-v1.safetensors`
- `models/qwen3-30b-a3b/m4.2-03-rust-intermediate-evidence-v1.tsv`
