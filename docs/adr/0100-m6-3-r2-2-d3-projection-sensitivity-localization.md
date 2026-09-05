# ADR 0100: M6.3-R2.2-D3 projection-sensitivity localization

- Status: Accepted
- Date: 2026-09-05
- Milestone: M6.3-R2.2-D3-PR

## Context

D1 proved that fixed group32 error is depth-sensitive. D2 then proved that
reducing grouped-int8 granularity from 32 to 16 or 8 does not recover the
frozen `0.001` local routed-output budget at Layers 24/47. Layer 24 even worsens
with finer grouping, while Layer 47 improves but remains far above the limit.

The next question is therefore not another group size. It is whether the local
error is concentrated in one or more expert projections (`gate`, `up`, `down`)
so a projection-level mixed representation could retain useful byte savings.

## Decision

Run scalar-only projection ablation on the frozen `short_thai` prompt at Layers
24 and 47. Reuse the exact group32 artifacts and canonical F32 expert payloads;
no new quantized artifact, native kernel, timing, or quality claim is allowed.
The frozen variant order is:

1. `gate+up+down`
2. `gate+up`
3. `gate+down`
4. `up+down`
5. `gate`
6. `up`
7. `down`

A named projection is group32; every omitted projection is canonical F32. Each
variant is evaluated against an all-F32 comparator from the same post-norm input
and router. Target-layer variants are not propagated downstream; the context
prefix is fixed independently of the variant.

Use the same four D2 contexts: Layer 24 with F32 and Layer0-group32 prefixes;
Layer 47 with F32 and original Layer0+24-group32 prefixes. Record per-position
and prompt-maximum routed-output error, router IDs, finiteness, and modeled
bytes. The local limit remains `0.001`.

Selection is prospective: inspect the fixed order above and choose the first
variant passing every frozen context for that layer. This maximizes packed
projection count with deterministic tie order. If none passes, select F32.

D3 may only propose a projection-level hybrid policy for a separate quality
re-entry review. R2.2c, all-layer rollout, and M6.4 remain blocked.