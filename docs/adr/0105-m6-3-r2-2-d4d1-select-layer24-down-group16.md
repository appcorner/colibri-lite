# ADR 0105: M6.3-R2.2-D4D1 Select Layer24 Down Group16

## Status

Accepted. D4D1 closes with deterministic selection `group16`.

Result SHA-256:
`486679e583c1881ef97d7d35ff771cab53f0ba66dd2eac84a5da88929c22bd43`.

## Context

D4 failed because Layer24 `down` group32 narrowly exceeded the held-out `short_english` local budget. ADR 0104 pre-registered a cross-fixture comparison of the already-frozen group32/group16/group8 Layer24 artifacts, with F32 gate/up and both canonical/L0-group32 prefixes.

The frozen D4D1 execution completed successfully with 12 evidence rows and no stderr.

## Decision

Apply the frozen coarsest-passing rule. Group32 fails `short_english`; group16 and group8 pass every fixture/prefix context. Select group16.

Maximum local errors across all contexts:
- group32: `0.0010076723992824554` — fail;
- group16: `0.0008960836566984653` — pass;
- group8: `0.0006817728281021118` — pass.

The proposed sentinel policy becomes Layer0 group32 gate/up/down, Layer24 F32 gate/up plus group16 down, and Layer47 F32. Layer24 modeled bytes remain `22.916667%` below all-F32 while costing only `1.369863%` more than the failed group32 hybrid.

This selection authorizes only a separately pre-registered held-out quality re-entry. D5, historical R2.2c/R2.2d, all-layer rollout, and M6.4 remain blocked.
