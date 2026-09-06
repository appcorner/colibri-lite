# ADR 0109: M6.3-R2.2-D4D3 Select Layer24 Down Group8

## Status

Accepted. D4D3 closes with deterministic selection `group8`.

Result SHA-256:
`6bda021e44212caf40117c6a7a80a8dbe451bfd6812968aaf315cae422d07a43`.

## Context

D4D2 showed that prompt-only group16 characterization did not generalize to the propagated autoregressive state: `short_thai` Layer24 local max-abs reached `0.0012040138`, above the frozen `0.001` limit.

ADR 0108 therefore froze a sequence-aware characterization using only the already-existing group32/group16/group8 Layer24 artifacts. Layer0 remains group32, Layer24 gate/up remain F32, each candidate trajectory propagates its own hidden/KV state, and token IDs remain on the frozen reference trajectory.

## Decision

Apply the frozen coarsest-passing order `32 -> 16 -> 8 -> F32`.

Worst Layer24 local max-abs across both fixtures and all measured positions:
- group32: `0.00117570161819458` — fail;
- group16: `0.0012040138244628906` — fail;
- group8: `0.0006819292902946472` — pass.

Select Layer24 F32 gate/up plus group8 down. Layer0 remains all-group32 and Layer47 remains canonical F32.

All candidate outputs were finite, prompt Layer24 router guards remained exact, and frozen greedy outputs matched at all measured generation points. These are diagnostic guards, not a held-out quality PASS.

The modeled Layer24 representation is `20.833333%` smaller than all-F32. The modeled three-sentinel policy is `30.902778%` smaller than all-F32. No runtime, I/O, or throughput claim is made.

## Consequences

The selected group8 policy authorizes only a separately pre-registered D4D4 held-out quality re-entry. D5, historical R2.2c/R2.2d, all-layer rollout, and M6.4 remain blocked.
