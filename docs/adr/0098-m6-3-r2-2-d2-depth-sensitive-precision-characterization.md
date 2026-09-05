# ADR 0098: M6.3-R2.2-D2 depth-sensitive precision characterization

- Status: Accepted
- Date: 2026-09-05
- Milestone: M6.3-R2.2-D2

## Context

ADR 0097 shows the admitted group32 representation is not depth-invariant.
Layer 0 remains within the original local `<=0.001` budget, while Layers 24 and
47 exceed it by about `11.68x` and `68.83x`. Layer 47 alone changes prompt
ranking, and Layers 0+24 show a separate interaction.

The existing packed layout accepts only group32/group64. Group64 is coarser and
cannot serve as a precision-recovery candidate. The scalar projection math is
otherwise group-size generic, and the native path already falls back to scalar
when group size is not 32.

## Decision

D2 may add characterization-only group16 and group8 layout support. It must not
change the group32 AVX2/FMA kernel, R2.1/R2.2 artifacts, router, attention,
activation, cache, or default runtime policy. D2 is scalar-only and untimed.
Create/freeze group16 and group8 artifacts for Layers 24 and 47 only, preserving
the same symmetric int8 + little-endian F32 scale grammar and rounding rule as
group32. Compare group32/group16/group8 against same-input/router canonical F32
routed output in two frozen upstream contexts per layer.

For Layer 24 the contexts are canonical F32 prefix and Layer-0-group32 prefix.
For Layer 47 they are canonical F32 prefix and the original R2.2 failing prefix
with Layers 0+24 group32. Measure the maximum local routed-output absolute error
over the complete `short_thai` prompt. No performance timing is permitted.

The prospective selection rule is fixed: choose the coarsest representation in
`group32 -> group16 -> group8` that satisfies local max-abs `<=0.001` in every
frozen context for that layer. If none passes, select canonical F32 fallback for
that layer. Selection only proposes a later hybrid quality-reentry candidate;
it does not itself pass quality.

## Consequences

All new artifact identities must be frozen before D2 measurements. No
results-based requantization, extra group sizes, context reselection, or retries
are allowed. R2.2c/R2.2d, all-layer rollout, and M6.4 remain blocked. A separate
quality-reentry contract is required after D2 selection.
