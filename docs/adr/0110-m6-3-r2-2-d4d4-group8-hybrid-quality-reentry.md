# ADR 0110: M6.3-R2.2-D4D4 Group8 Hybrid Held-Out Quality Re-entry

## Status

Pre-registered before D4D4 implementation or execution.

Contract SHA-256:
`3f8cb0c72180b46b8e6fe29d1b27ea50251119cc9f5ce9d3356d0e0eb5e1d9ed`.

## Context

ADR 0109 selects Layer24 `down` group8 as the first precision that stays within the frozen `0.001` local budget across every measured propagated position in both held-out fixtures. This changes the D4D2 candidate and therefore requires a new held-out quality re-entry.

The current native projection kernel supports group32 only. D4D4 must not imply native group8 support.

## Decision

Evaluate the exact policy: Layer0 group32 gate/up/down; Layer24 canonical F32 gate/up plus group8 down; Layer47 and all other layers canonical F32.

For each held-out fixture run scalar hybrid once and mixed-backend hybrid twice. Scalar uses Layer0 group32 scalar and Layer24 group8-down scalar. Mixed-backend differs only at Layer0, where group32 uses AVX2+FMA; Layer24 remains scalar group8.

Require exact two-token generation, prompt top-20 ordering, argmax, Layers0/24/47 router guards, finite logits, the existing F32 logit envelope, scalar-candidate/F32 local max-abs `<=0.001` at Layers0/24, mixed/scalar Layer0 max-abs `<=0.001`, mixed/scalar prompt-logit max-abs `<=0.002`, and exact repeatability of the two mixed runs.

This quality seam may decode a complete F32 expert at Layer24 and therefore cannot support resource/I/O/performance claims.

PASS authorizes only D5 hybrid artifact/layout and backend/performance-contract design. It does not authorize historical R2.2c, performance timing, native group8 implementation, all-layer rollout, or M6.4.
