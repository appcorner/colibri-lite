# ADR 0112: M6.3-R2.2-D5 Production Hybrid Layout and Performance Contract

## Status

Pre-registered before D5 implementation.

Contract SHA-256:
`34b5a5a032bf6251b7d724c115bad9db44fc11f08092d235155e32c304d8acd3`.

## Context

ADR 0111 closes D4D4 `GO` for the exact Layer0 group32 + Layer24 F32 gate/up with group8 down + Layer47 F32 policy. D4D4 used a quality seam that may decode a complete F32 Layer24 expert, so it cannot support production resource or performance claims.

## Decision

D5 proceeds in three ordered phases: production-like artifact/layout, production-path quality revalidation, then paired end-to-end performance/resource measurement.

Layer24 must use a hybrid artifact containing F32 gate, F32 up, and packed group8 down. The candidate may not materialize the complete F32 down projection or a whole F32 expert. Layer0 reuses the frozen group32 artifact and AVX2+FMA backend; Layer24 group8 down remains scalar packed. No native group8 implementation or new unsafe boundary is allowed.

Production-path quality must pass the unchanged D4D4 gates before any official performance sample.

If quality passes, run 40 fresh process samples: two fixtures x two cache labels x five balanced F32/hybrid pairs x two paths. Record wall time, prefill/decode throughput, working-set/private-byte peaks, logical artifact bytes, expert payload bytes, and candidate artifact bytes.

A performance GO requires at least 1% median wall-time speedup in each fixture, at least 4/5 hybrid wins in every fixture/cache-label cell, no more than 1% peak working-set/private-byte regression, and positive logical expert-byte reduction.

A D5 GO authorizes only a separate all-layer plan review. Historical R2.2c, all-layer rollout, and M6.4 remain blocked.
