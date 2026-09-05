# ADR 0102: M6.3-R2.2-D4 Hybrid Held-Out Quality Re-entry

## Status

Pre-registered before D4 implementation or execution.

Contract SHA-256:
`5f2b27b623a0456ad15e14b542816e4d5ad17bf32dacd91099184adfc86fb4fe`.

## Context

ADR 0101 closes D3 with one exact semantic policy derived from frozen evidence:
Layer 0 uses all group32, Layer 24 uses group32 only for `down` while `gate/up`
remain canonical F32, and Layer 47 remains canonical F32. D3 was diagnostic and
made no held-out quality claim.

The original R2.2c timing contract belongs to the failed all-three-sentinel
group32 candidate. It must not be reused for the new hybrid policy.

## Decision

Run a new quality-only re-entry on the frozen `short_english` and `short_thai`
fixtures. Execute one scalar hybrid run and two native hybrid runs per fixture.
Native applies only to group32 projections: all three at Layer 0 and `down` at
Layer 24. Layer 47 remains F32.

Require exact two-token generation, exact prompt top-20 order, exact argmax,
exact Layers 0/24/47 router guards, finite logits, and the existing F32 compact
logit envelope `min(0.05, fixture.margin/4)`.
Additionally require scalar-hybrid versus same-input/router F32 routed-output
max-abs `<=0.001` at Layers 0 and 24; native versus scalar hybrid max-abs
`<=0.001` at those layers; and native versus scalar prompt-logit max-abs
`<=0.002`. Native repeatability must be exact for generation, top-20, argmax,
router guards, prompt-logit SHA-256, final-norm SHA-256, and local error maps.

No automatic retry is permitted. Evidence is written only after both fixtures
complete all gates.

## Consequences

A D4 PASS authorizes only D5 hybrid artifact/layout and performance-contract
design. It does not authorize the historical R2.2c samples, all-layer rollout,
or M6.4.

A D4 failure closes this exact hybrid quality re-entry. Thresholds, projection
selection, fixtures, or artifact identities may not be changed after observing
results under this contract.
