# ADR 0099: M6.3-R2.2-D2 grouped-int8 granularity NO-GO

- Status: Accepted
- Date: 2026-09-05
- Milestone: M6.3-R2.2-D2

## Context

ADR 0098 pre-registered scalar-only depth-sensitive characterization for
Layers 24 and 47 with candidate groups `32 -> 16 -> 8`, a fixed local routed
output max-abs limit of `0.001`, and canonical F32 fallback when no group passes
all frozen contexts.

The frozen D2 execution completed successfully in `352.44s` with exit code `0`,
12/12 evidence rows, empty stderr, exact final router guards, and finite
candidate outputs. Evidence SHA-256 is
`f15f2286fb622475a9bfcfc125beb14823e5e76f112886c65e070f1db2c551dc`.

## Result

Layer 24 worst-context max-abs is `0.0116818845` for group32,
`0.0132874548` for group16, and `0.0151521564` for group8. None passes.
Layer 47 is `0.0688347816`, `0.0515899658`, and `0.0298538208`
respectively. Finer grouping helps Layer 47 but remains far above the limit.
## Decision

Apply the pre-registered fallback: Layers 24 and 47 remain canonical F32.
The resulting sentinel policy is therefore Layer 0 group32 plus Layers 24/47
F32. This is not a new multi-layer candidate; it collapses to the already
validated R2.1 Layer-0 vertical slice for these sentinels.

Do not run a redundant held-out quality re-entry for that no-op expansion.
R2.2c/R2.2d remain blocked. No additional group size may be introduced after
these results under D2.

The next authorized work is diagnostic design only: localize sensitivity by
expert projection (`gate`, `up`, `down`) at Layers 24/47 before choosing a new
representation family or higher bit width. Any hybrid projection candidate
requires a new prospective contract and quality-before-performance review.

Result SHA-256:
`847b146f56e0c04f5836875ab178bab320a76a493df64b6a6ba5d8c517bb518c`.

## Consequences

- Group granularity alone is closed as the R2.2 recovery path.
- The AVX2 group32 kernel remains unchanged and is not implicated by D2.
- No R2.2 official performance samples are authorized.
- All-layer rollout and M6.4 remain blocked.
