# ADR 0025: Short Cached Generation Validation Budgets

- Status: Accepted provisionally for M4.2-04
- Date: 2026-07-16
- Milestone: M4.2
- Task: M4.2-04
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

M4.2-03 validated complete Layer-47 expert and block execution. M4.2-04 must
extend the same storage-aware F32 path through final RMSNorm, the untied LM
head, greedy token selection, and genuine fixed-capacity KV-cache reuse.

The frozen input and generated sequence is:

```text
prompt:    9707, 11, 1879, 0
generated: 1096, 374
processed: 9707, 11, 1879, 0, 1096, 374
```

Transformers BF16 and same-weight F32 use independent `DynamicCache`
instances. Rust uses the existing `KvCache`, cached-attention kernel,
router/expert path, and lower-token-ID greedy tie rule. The primary Rust path
does not perform full-sequence recomputation.

Full vocabulary rows are required transiently to measure all-logit error and
top-20 rank agreement. They are hashed, compared, and removed after validation;
only compact checkpoints and summaries remain.

## Decision

Use separate per-step budgets for Layer-47 expert output, routing weights,
combined MoE output, final block, final RMSNorm, and LM-head logits.

For each step, freeze:

```text
expert/MoE/block budget = 3 * observed error + 1e-6
routing budget          = 3 * observed error + 1e-7
final RMSNorm budget    = 3 * observed incoming block error + 5e-7
LM-head logit budget    = 3 * observed all-logit error + 2e-6
```

The formulas preserve separate allowance for incoming hidden-state drift,
expert projection/aggregation variance, residual propagation, RMSNorm
reduction order, and LM-head matrix accumulation. Token selection additionally
requires the F32 top-1 margin to exceed twice the measured all-logit error.

| Step | Expert | Routing | MoE | Block | Final norm | Logits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `1.9808e-3` | `2.7599e-6` | `1.3285e-3` | `1.5574e-3` | `1.5569e-3` | `2.0513e-4` |
| 1 | `5.4173e-4` | `2.4246e-6` | `2.6421e-4` | `3.4432e-4` | `3.4382e-4` | `1.5650e-4` |
| 2 | `2.0699e-4` | `1.7093e-6` | `6.6804e-5` | `5.7320e-4` | `5.7270e-4` | `1.3361e-4` |
| 3 | `3.6864e-4` | `1.8434e-6` | `2.6421e-4` | `5.0454e-4` | `5.0404e-4` | `1.0214e-4` |
| 4 | `2.5277e-4` | `1.2846e-6` | `8.6473e-5` | `3.9010e-4` | `3.8960e-4` | `1.1644e-4` |
| 5 | `2.9855e-4` | `1.5752e-6` | `1.3833e-4` | `1.8411e-4` | `1.8361e-4` | `1.3647e-4` |

## Cache Contract

- Allocate 48 layers at fixed capacity 6 with `[capacity, 4, 128]` keys and
  values.
- Require `cache.len()` to equal the token position before execution and
  position plus one after append.
- Compare every initialized key/value prefix byte-for-byte after each append.
- Require all 96 `Vec` capacities and total `1,179,648` payload bytes to remain
  unchanged.
- Compare cached F32 argmax with Transformers incremental and full-recompute
  argmax at sequence lengths 4, 5, and 6.

## Consequences

- Full Qwen3-30B-A3B inference now reaches deterministic vocabulary token IDs
  through the unquantized correctness path.
- The LM head is read in bounded 256-row buffered chunks; scalar dot order is
  unchanged and no mmap, SIMD, threading, quantization, GPU, or FFI is added.
- Canonical artifacts and pinned source remain read-only.
- These budgets apply only to this frozen six-position validation path and do
  not authorize general tolerance changes or performance claims.

## Evidence

- `models/qwen3-30b-a3b/m4.2-04-transformers-bf16-generation-evidence-v1.json`
- `models/qwen3-30b-a3b/m4.2-04-transformers-f32-generation-evidence-v1.json`
- `models/qwen3-30b-a3b/m4.2-04-rust-short-generation-evidence-v1.tsv`
