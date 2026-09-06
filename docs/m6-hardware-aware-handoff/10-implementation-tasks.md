# Implementation Order

1. M6.0: pin `reference-f32-v1`, comparator, backend-neutral contracts, and
   bilingual deterministic fixtures.
2. M6.1: implement versioned hardware/model profiles and controlled benchmarks.
3. M6.2: implement deterministic candidate enumeration and budgeted ranking.
4. M6.3: review one backend/precision proposal, build bounded vertical slices,
   and require evidence-backed stop/go decisions before expansion.

## Current execution state

M6.3 is active and has progressed through the D5 production-like hybrid proof
and the R2.3 all-layer planning gates. `M6.3-R2.2-D5c` is closed `GO`, the
R2.3 all-layer plan review is closed `GO`, the R2.3 implementation contract is
frozen, and `M6.3-R2.3a` is closed `PASS` with a four-token canonical-F32
reference.

The exact next task is `M6.3-R2.3b`: characterize the 45 unmeasured layers
(`1-23,25-46`) over the frozen 21-candidate grid per layer. No unmeasured layer
may inherit a policy from a neighboring or sentinel layer. F32 is the default
fallback until direct evidence admits a non-F32 candidate.

The authoritative executable checklist is `docs/tasks.md`. The frozen R2.3
implementation contract is
`models/qwen3-30b-a3b/m6.3-r2-3-implementation-contract-v1.json`. M6.4 remains
blocked until R2.3 completes and a separate M6.4 entry review authorizes it.
