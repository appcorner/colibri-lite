# ADR 0111: M6.3-R2.2-D4D4 Group8 Hybrid Held-Out Quality GO

## Status

Accepted. D4D4 closes `GO`.

Result SHA-256:
`ab77024116847b59fe53ce8571794b6584139d9ff173a338f485e44d567dc4cc`.

## Context

ADR 0110 pre-registered held-out quality for the exact D4D3-selected policy: Layer0 all-group32, Layer24 F32 gate/up plus group8 down, Layer47 and all other layers canonical F32.

The frozen durable execution completed with exit code `0`, runtime `767.61s`, six quality executions, zero stderr, and evidence SHA-256 `a330462130baa7b06b52122ce5851ffe412ff6496ade909b418c370d04767a6b`.

## Decision

The candidate passes every frozen D4D4 gate on both held-out fixtures.

English preserves generated IDs `[0,358]`, exact prompt top-20/argmax/router guards, Layer24 scalar-hybrid/F32 max-abs `0.0006817728`, Layer0 mixed/scalar max-abs `2.3841858e-7`, and prompt-logit mixed/scalar max-abs `2.2888184e-5`.

Thai preserves generated IDs `[7360,91]`, exact prompt top-20/argmax/router guards, Layer24 scalar-hybrid/F32 max-abs `0.0006819293`, Layer0 mixed/scalar max-abs `2.3841858e-7`, and prompt-logit mixed/scalar max-abs `3.3378601e-5`.

Both mixed runs are exactly repeatable and all logits are finite.

D4D4 therefore authorizes only D5 hybrid artifact/layout and backend/performance-contract design. D5 implementation, historical R2.2c reuse, all-layer rollout, and M6.4 remain blocked.
