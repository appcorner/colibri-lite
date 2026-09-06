# ADR 0115: M6.3-R2.2-D5c Paired Performance Execution Freeze

## Status

Accepted before official D5c performance samples are observed.

## Context

ADR 0112 pre-registers a 40-process paired performance/resource matrix after production-path quality passes. ADR 0114 closes D5b `GO`. The D5c sample harness is frozen at commit `322b1fc8824cd108983aac58636907feee54a57a`.

## Decision

D5c reuses the validated R1.3 ETW collector, parser, and READY/GO handshake. The official matrix is two fixtures (`short_english`, `short_thai`) x two process/cache labels (`first_process_touch`, `likely_warm`) x five balanced pairs x two paths (`reference_f32`, `production_hybrid`) = 40 fresh process samples.

The cache labels describe software/process state only and do not claim controlled cold-device storage semantics.

The frozen wall-time gate uses the median of ten paired wall-speedup percentages per fixture. Each fixture/cache-label cell must have at least four hybrid wall-time wins out of five pairs. Working-set and private-byte gates compare the reference and hybrid median process peaks within each fixture/cache-label cell; the worst regression must be <= 1%. Logical expert-payload bytes must be reduced in every pair.

No result-dependent threshold change, pair reordering, automatic retry, native group8 implementation, cache-policy change, historical R2.2c reuse, all-layer rollout, or M6.4 implementation is authorized. Exact-prefix resume support is transport recovery only and requires an explicit review; it is not an automatic retry.

A D5c `GO` authorizes only a separate all-layer plan review.