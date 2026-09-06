# ADR 0118: M6.3-R2.3 All-Layer Plan Review Contract

## Status

Pre-registered before R2.3 plan review.

Contract SHA-256:
`4c4f37356e01d57b247664d0c2537b2f928ad329863778f98a9de0b54624ba2c`.

## Context

ADR 0117 closes D5c `GO` for the production-like Layer0 group32 + Layer24 F32 gate/up with group8 down + Layer47 F32 policy. D5 proves a measured end-to-end benefit for that validated slice, but it does not prove that the same precision policy generalizes to the other 45 MoE layers.

Earlier R2.2 evidence already showed that depth sensitivity is material: full group32 passes at Layer0, fails at Layer24 and Layer47, Layer24 admits only a projection-aware down path after precision refinement, and Layer47 requires F32. Finer grouping was also not monotonic at every depth. Any all-layer design that copies a sentinel policy to neighboring layers would therefore overstate the evidence.

## Decision

R2.3 begins as plan design/review only. No artifact build, conversion, runtime change, performance execution, 48-layer rollout, or M6.4 work is authorized by this contract.

The review must publish an explicit 48-layer status matrix. Layers 0, 24, and 47 retain only their already validated status; all other layers begin as `unmeasured_default_f32` and may not inherit a non-F32 policy by interpolation.

The review must define a deterministic future characterization ladder, cumulative interaction/admission procedure, full-model quality re-entry, production artifact/layout design, paired performance/resource proof, rollback/F32 fallback rules, and an uncertainty-bounded impact model. Physical-I/O claims remain prohibited unless Windows telemetry is separately revalidated.

A complete review `GO` may authorize only an R2.3 implementation-contract design for the exact reviewed plan. It does not itself authorize implementation or M6.4.
