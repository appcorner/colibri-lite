# ADR 0022: Layer-24 Propagated Validation Budgets

- Status: Accepted provisionally for M4.2 Layer 24
- Date: 2026-07-16
- Milestone: M4.2
- Task: M4.2-02
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

The genuine Layer-24 input requires complete Rust execution of Layers 0 through
23. The path streams only router-selected experts through the normal
`ExpertStore`; it does not inject a Transformers hidden state. Consequently,
local reduction differences and expert-stage differences propagate through 24
residual blocks.

Characterization retained compact F32 and BF16 checkpoints for every layer's
input, both RMSNorms, attention, residual, router, routing weights, combined
MoE output, and final block output. The first attempted scalar-only combined
MoE guard stopped at Layer 1, token 0, element 475: error
`1.2293457984924316e-6` versus `1.190322336697136e-6`.

The required frozen-input diagnostic used the exact Rust Layer-1 expert input,
IDs, and routing weights. The same-input PyTorch F32 combined MoE maximum was
`8.392333984375e-5` and all 8,192 outputs passed the existing scalar rule. The
original-path maximum was `3.509521484375e-4` with one scalar-rule failure.
This classifies the failure as accumulated incoming drift, not a local Rust
expert implementation mismatch. No runtime arithmetic or global tolerance was
changed.

## Decision

Freeze the observed per-layer components in the feature-gated validation and
derive a separate absolute budget for each layer and checkpoint.

For layer `L`, the recurrence is:

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

combined MoE, L < 24:
  3 * measured_post_rmsnorm_error[L] + measured_moe_variance[L]

final block, L < 24:
  measured_residual_error[L] + moe_budget[L]
    + f32::EPSILON * max(abs(reference_block[L]))
```

The RMSNorm allowance and factor 3 retain ADR 0020. The router-local allowance
retains the Layer-0 same-input measurement. The routing bound follows the
maximum row-sum derivative of softmax for an unchanged selected set. The final
term is a one-ULP magnitude-dependent guard for the residual addition; it was
required at a Layer-2 reference value near `800` and is not a blanket absolute
tolerance.

The per-layer attention and combined-MoE terms are measured stage variances for
this frozen path. They are not general kernel tolerances. The Layer-1 isolated
diagnostic remains the evidence that the expert-stage propagation does not hide
a local scalar-contract failure.

Router IDs retain ADR 0019. Exact F32 IDs require a positive boundary margin
greater than twice the measured per-token F32-versus-Rust logit error. BF16 is
classified independently.

## Layer-24 Budgets

| Checkpoint | Observed maximum | Absolute budget |
| --- | ---: | ---: |
| Input | `2.3193359375e-3` | `6.9585079327e-3` |
| Input RMSNorm | `3.4570693970e-6` | `6.9585079327e-3` |
| Attention | `1.8477439880e-6` | `1.2218952179e-5` |
| Residual | `2.3193359375e-3` | `2.3315548897e-3` |
| Post-attention RMSNorm | `1.4877319336e-4` | `6.9585079327e-3` |
| Router logits | `3.7193298340e-5` | `4.6062469482e-4` |
| Routing weights | `8.6426734924e-7` | `1.8696649931e-5` |

## Consequences

- Every completed layer and Layer-24 pre-router checkpoint must pass its own
  frozen budget before later work can execute.
- Layer-24 experts, Layer 25, final norm, LM head, and generation remain
  unreachable in this validation.
- The scalar Rust operation order, RMSNorm, router ordering, artifact format,
  cache policy, and public APIs are unchanged.
- Layer 47 requires a new propagated characterization and cannot reuse these
  budgets.

## Evidence

- `models/qwen3-30b-a3b/m4.2-02-layer1-moe-isolated-diagnostic-v1.json`
- `models/qwen3-30b-a3b/m4.2-02-rust-layer24-layer-evidence-v1.tsv`
- `models/qwen3-30b-a3b/m4.2-02-rust-layer24-checkpoint-evidence-v1.tsv`
- `models/qwen3-30b-a3b/m4.2-02-rust-layer24-router-evidence-v1.tsv`
