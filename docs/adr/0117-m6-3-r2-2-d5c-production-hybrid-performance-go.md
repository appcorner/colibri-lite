# ADR 0117: M6.3-R2.2-D5c Production-Hybrid Performance GO

## Status

Accepted. D5c closes `GO` and authorizes `M6.3-R2.3` all-layer plan design/review only.

Frozen paired samples SHA-256:
`47ac917cb7b3886e4ff786639c832e28936f1d64fd6c4335e5819d3f924c8415`.

Frozen validator result SHA-256:
`b6387a2975074a9f2fc164c04a1741fc6324d64c54fd1947a27fd4e023a730e3`.

## Context

ADR 0112 pre-registered the D5 production-hybrid performance contract. ADR 0114 closed production-path quality `GO`. ADR 0116 recorded that the first D5c transport attempt admitted zero samples because the reused R1.3 ETW exact physical-I/O join is unavailable on the current Windows trace; the D5 contract did not require physical-I/O correlation. The accepted v2 run therefore used the frozen D5-specific READY/GO process-memory collector and started a fresh 40-process matrix.

The final evidence contains exactly 40 samples: two fixtures x two cache labels x five balanced pairs x two paths. The frozen validator produced `decision=go` with all four gates true.

## Decision

The production hybrid passes the frozen D5 performance/resource gates.

- `short_english` median wall-time speedup: `3.4380408492842314%`; median prefill throughput improvement: `3.730782247729743%`; median decode throughput improvement: `3.5994472906106285%`.
- `short_thai` median wall-time speedup: `4.029556583245903%`; median prefill throughput improvement: `4.0062478462800595%`; median decode throughput improvement: `4.421853644159534%`.
- Hybrid wall-time wins are `5/5` English first-process-touch, `5/5` English likely-warm, `4/5` Thai first-process-touch, and `5/5` Thai likely-warm. Every cell meets the frozen `>=4/5` gate.
- Worst median memory regression is `0.8145196989818504%`, below the frozen `1.0%` limit. It occurs in English likely-warm private bytes. The largest working-set regression is `0.291970802919708%`.
- Logical expert payload reduction is positive in every pair: `419,954,688` bytes per English pair and `279,969,792` bytes per Thai pair. The minimum frozen reduction is `279,969,792` bytes.

Physical SSD-read reduction is not claimed. The accepted D5c result establishes end-to-end wall/throughput benefit, bounded process-memory behavior, and reduced logical expert payload for the validated production-like three-sentinel policy only.

## Consequences

`M6.3-R2.3` may begin as a separate all-layer **plan design/review**. This ADR does not authorize R2.3 implementation, a 48-layer rollout, M6.4, native group8, new precision, cache-policy changes, or a production-wide speedup claim. Historical R2.2c/R2.2d remain blocked and are not reusable evidence for this policy.
