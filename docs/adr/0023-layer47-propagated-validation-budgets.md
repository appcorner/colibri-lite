# ADR 0023: Layer-47 Propagated Validation Budgets

- Status: Accepted provisionally for M4.2 Layer 47
- Date: 2026-07-16
- Milestone: M4.2
- Task: M4.2-02
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

Layer 47 receives the genuine Rust output of 47 complete storage-aware blocks.
Layers 0-46 execute attention, routing, only selected expert MLPs, weighted MoE
aggregation, and the final residual. Layer 47 executes only through routing.

The approved Layer-24 budgets cannot define the additional Layers 24-47. The
Layer-47 characterization therefore retained new BF16, F32, and Rust evidence
for every input, RMSNorm, attention, residual, router, routing-weight, combined
MoE, and completed-block checkpoint.

No new checkpoint exceeded its characterized recurrence. No isolated same-input
diagnostic was required. The accepted Layer-1 incoming-drift diagnosis remains
unchanged and no local implementation defect appeared.

## Decision

Freeze the newly measured per-layer components through Layer 47 and apply the
same component model, with new values rather than reused Layer-24 budgets:

```text
input:
  L = 0: 0
  L > 0: 3 * measured_input_error[L] + 5e-7

input RMSNorm:
  3 * measured_input_error[L] + 5e-7

attention:
  3 * measured_input_rmsnorm_error[L]
    + measured_attention_variance[L]

residual:
  measured_input_error[L] + attention_budget[L]

post-attention RMSNorm:
  3 * measured_residual_error[L] + 5e-7

router logits:
  3 * measured_post_rmsnorm_error[L] + 1.430511474609375e-5

routing weights:
  0.5 * measured_router_logit_error[L] + 1e-7

combined MoE, L < 47:
  3 * measured_post_rmsnorm_error[L] + measured_moe_variance[L]

final block, L < 47:
  measured_residual_error[L] + moe_budget[L]
    + f32::EPSILON * max(abs(reference_block[L]))
```

The formulas retain the approved interpretation of RMSNorm reduction variance,
softmax sensitivity, router projection allowance, expert-stage propagation,
and residual-add rounding. The frozen arrays for Layers 25-47 and expert stages
24-46 are new Layer-47 evidence and are not aliases of ADR 0022.

## Propagation Shape

- Largest absolute block drift: `2.3193359375e-3`, first reached at Layer 3.
- Largest local block-error increase: Layer 3, `1.52587890625e-3`.
- Layers 4-45 retain the same maximum block error; there is no late positive
  jump.
- Layer 46 has a `1.3427734375e-3` combined-MoE error but partially cancels the
  incoming maximum, reducing the block error to `9.765625e-4`.
- Layer-47 attention further reduces the maximum residual error to
  `9.1552734375e-4` before post-attention RMSNorm.

This is bounded early stepwise growth followed by a stable plateau and a late
decrease. It is not gradual accumulation and contains no unexplained anomalous
late increase.

## Layer-47 Budgets

| Checkpoint | Observed maximum | Absolute budget |
| --- | ---: | ---: |
| Input | `9.765625e-4` | `2.9301873874e-3` |
| Input RMSNorm | `3.0517578125e-5` | `2.9301873874e-3` |
| Attention | `1.3732910156e-4` | `2.2888183594e-4` |
| Residual | `9.1552734375e-4` | `1.2054443359e-3` |
| Post-attention RMSNorm | `7.6293945313e-5` | `2.7470819186e-3` |
| Router logits | `3.0040740967e-5` | `2.4318695068e-4` |
| Routing weights | `8.9406967163e-7` | `1.5120370335e-5` |

## Consequences

- Every checkpoint from embedding through Layer-47 routing is guarded before
  the hard stop.
- Layer-47 experts, Layer-47 block output, final model RMSNorm, LM head, logits,
  sampling, and generation remain unreachable.
- RMSNorm arithmetic, router ordering, cache behavior, public APIs, artifact
  format, and global numerical contracts are unchanged.
- These budgets close M4.2-02 only. They do not authorize M4.2-03 execution or
  define general runtime tolerances.

## Evidence

- `models/qwen3-30b-a3b/m4.2-02-rust-layer47-layer-evidence-v1.tsv`
- `models/qwen3-30b-a3b/m4.2-02-rust-layer47-checkpoint-evidence-v1.tsv`
- `models/qwen3-30b-a3b/m4.2-02-rust-layer47-router-evidence-v1.tsv`
- `models/qwen3-30b-a3b/m4.2-02-layer47-transformers-f32-evidence-v1.json`
- `models/qwen3-30b-a3b/m4.2-02-layer47-transformers-bf16-evidence-v1.json`
