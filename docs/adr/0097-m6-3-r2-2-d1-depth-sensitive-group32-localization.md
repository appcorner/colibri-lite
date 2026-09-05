# ADR 0097: M6.3-R2.2-D1 depth-sensitive group32 localization

- Status: Accepted
- Date: 2026-09-05
- Milestone: M6.3-R2.2-D1

## Context

ADR 0095 closed three-sentinel R2.2 quality `NO-GO`. ADR 0096 then froze a
scalar-only, prompt-only `short_thai` diagnostic over the F32 control and all
seven subsets of Layers 0/24/47. The durable D1 execution completed with exit
code `0` in `436.49s`; evidence SHA-256 is
`d6d05c0f5201fde047237082bb01af684215d3c17c5dd242c9357fc23c6bf473`.

Across all eight states, the prompt argmax remains `7360`, the top-20 token set
is unchanged, and Layers 0/24/47 router guards remain exact. The observed drift
is therefore a ranking sensitivity downstream of expert approximation, not a
router selection change.

## Decision

D1 classifies the unchanged group32 representation as **depth-sensitive**.
Layer 47 alone is sufficient to swap the frozen ranks 18/19 (`52388` and
`129075`). Layers 0 and 24 each preserve exact top-20 ordering in isolation,
but their `[0,24]` combination swaps both rank pairs, establishing a separate
multi-layer interaction/accumulation effect.
Local scalar-group32 versus same-input/router F32 routed-output max-abs is
`8.7054e-4` at Layer 0, `1.1682e-2` at Layer 24, and `6.8835e-2` at Layer 47.
Against the original R2.2 local limit `0.001`, Layers 24 and 47 are approximately
`11.68x` and `68.83x` over budget while Layer 0 remains within budget.

The scalar-only diagnostic reproduces the quality failure, so no native-kernel
cause is required by current evidence. The R2.1 group32 representation cannot
be assumed to generalize uniformly across model depth.

D1 result SHA-256:
`602614c45accb4393cfa1cec28801d9f906002943b1a0523b7c1d9319cf3a9f0`.

## Consequences

D1 is closed as diagnostic evidence, not a quality re-entry PASS. R2.2c/R2.2d,
all-layer rollout, and M6.4 remain blocked. The next authorized work is a
prospectively frozen depth-sensitive precision characterization focused on
Layers 24 and 47. Candidate precision/granularity options must be verified as
supported and frozen before their results are observed; no post-result
requantization or performance claim is authorized by this ADR.
