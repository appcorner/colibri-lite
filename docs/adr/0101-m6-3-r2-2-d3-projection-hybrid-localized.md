# ADR 0101: M6.3-R2.2-D3 Projection-Hybrid Localization

## Status

Accepted. D3 is closed.

## Context

ADR 0099 showed that reducing grouped-int8 granularity from 32 to 16 to 8
cannot recover the frozen `0.001` local routed-output budget at Layers 24 or 47.
ADR 0100 therefore pre-registered projection sensitivity across every non-empty
subset of `gate`, `up`, and `down`, using the original group32 representation
and canonical F32 for omitted projections.

The frozen D3 execution completed 28/28 rows in one durable run. Native process
exit was `0`; stderr was empty; runtime was `269.30s`. The all-packed seam was
bit-exact with the existing scalar group32 path in every row, router guards
remained exact, and every candidate output was finite.

## Decision

Apply ADR 0100's prospective selection rule without modification.

For Layer 24, select `down` as the only packed projection. It passes both frozen
contexts with worst local max-abs `4.88266348838806152e-4`, below `0.001`.
`gate` and `up` remain canonical F32.
For Layer 47, no packed subset passes. The lowest-error observed variant is
`up` alone, but its worst-context max-abs is `1.63116455078125000e-2`, over
16x the frozen limit. Layer 47 therefore falls back to canonical F32.

The resulting three-sentinel representation proposal is:

- Layer 0: existing all-group32 candidate;
- Layer 24: F32 `gate` + F32 `up` + group32 `down`;
- Layer 47: canonical F32.

Its modeled three-layer payload is `4,932,501,504` bytes versus
`7,247,757,312` bytes all-F32, a modeled saving of `2,315,255,808` bytes
(`31.944444%`). This is a representation-size model only, not measured I/O,
working-set, latency, or tokens/s evidence.

## Consequences

D3 authorizes only a separately pre-registered held-out quality re-entry design
for this exact hybrid policy. It does not authorize R2.2c performance samples,
a production artifact layout, all-layer rollout, or M6.4.

No post-result projection reselection, new bit width, or new group size is
permitted under D3. Any Layer-47 representation beyond F32 requires a new
prospective hypothesis.
