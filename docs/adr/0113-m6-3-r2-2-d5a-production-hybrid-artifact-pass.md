# ADR 0113: M6.3-R2.2-D5a Production Hybrid Artifact PASS

## Status

Accepted. D5a closes `PASS`.

Result SHA-256:
`8ca431533ed94c8a07fe3e88512cf3300ef41d5a78e09027a920f84c8e269836`.

## Context

ADR 0112 requires a production-like Layer24 representation before any runtime-value claim. The validated policy is F32 gate/up plus grouped-i8 group8 down, with no complete F32 down or whole F32 expert materialization in the candidate path.

## Decision

Freeze the Layer24 hybrid artifact at 1,912,602,624 bytes, or 14,942,208 bytes per expert. Its SHA-256 is `d94d12cbea648e2f2911573c893f564254cbde526d60132c600ed24e88317ac2`.

The builder copies canonical F32 gate/up and frozen group8 down bytes without requantization. Independent verification checks all 128 experts and all 1,912,602,624 hybrid bytes byte-for-byte against the two frozen sources.

The direct-consumption Rust reader has no F32 down field or buffer. Synthetic reader tests pass 3/3 and Clippy passes with `-D warnings`.

D5a authorizes only D5b production-path quality validation. D5c performance timing remains blocked until D5b passes. Historical R2.2c/R2.2d, all-layer rollout, and M6.4 remain blocked.
