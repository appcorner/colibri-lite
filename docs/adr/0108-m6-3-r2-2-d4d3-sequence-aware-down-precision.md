# ADR 0108: M6.3-R2.2-D4D3 Sequence-Aware Layer24 Down Precision

## Status

Pre-registered before D4D3 implementation or execution.

Contract SHA-256:
`dddc8b4ec320835686e5c5286b420e5b39bb4f3bbb3ef95e0f6b82bab473470a`.

## Context

D4D1 selected Layer24 down group16 from prompt-only cross-fixture evidence.
D4D2 then failed on `short_thai` during the full autoregressive quality path:
Layer24 scalar-hybrid/F32 local max-abs reached `0.0012040138`, above `0.001`.
The missing variable is propagated sequence state.

## Decision

Characterize Layer24 down at the already-frozen group32/group16/group8
precisions over the full two-token reference trajectory. Layer0 remains group32
scalar and Layers1-23 remain F32. Candidate hidden state is propagated across
positions, but the token trajectory is frozen to the reference so candidate
branching cannot confound the local precision comparison.

For each fixture and group, record same-input/router Layer24 F32-comparator
error at every processed position, router IDs, finiteness, and frozen greedy
agreement. Apply the unchanged `0.001` coarsest-passing rule in 32->16->8 order.

No new artifact, group size, higher bit width, native change, timing, retry,
quality-pass claim, or threshold change is authorized.