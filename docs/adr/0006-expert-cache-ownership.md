# ADR 0006: Expert Cache Ownership and Leases

- Status: Accepted
- Date: 2026-07-14
- Milestone: M2.2

## Context

On-demand experts must obey a byte budget without invalidating bytes while an
inference operation is using them. Eviction order and metrics must be
deterministic enough for tests and reproducible reports.

## Decision

`ExpertKey` is `(layer_index, ExpertId)` and identifies one complete expert
payload. `ExpertStore` maps stable keys to artifact tensor names and loads them
through `ArtifactReader` only on a cache miss.

`ExpertCache` owns payloads as `Arc<[u8]>`. `ExpertLease` clones the Arc and is
the pin: an entry is evictable only when the cache holds the sole strong
reference. Admission evicts the least-recently-used unpinned entry; equal access
times are ordered by `ExpertKey`.

The budget counts exact payload bytes. A payload larger than the whole budget is
rejected. If all eviction candidates are pinned, admission fails without
exceeding the budget.

Metrics record hits, misses, successful loads, evictions, current/peak resident
bytes, and bytes read from the artifact reader.

## Invariants

- Resident bytes never exceed the configured budget.
- Leased bytes remain alive even when other entries are admitted or evicted.
- Cache hits never call the loader.
- Failed oversized/pinned admission does not alter resident accounting.
- Eviction is deterministic for a deterministic access sequence.
- Unknown and duplicate expert registrations fail with structured errors.

## Evidence

- Deterministic hit/miss/load/eviction and MRU/LRU test passes.
- Two pinned entries prevent admission; releasing one permits its eviction.
- A live lease remains readable after other cache activity.
- Oversized payload rejection and strict resident-budget assertion pass.
- `ExpertStore` performs one artifact read followed by a cache hit and reports
  exact bytes-read metrics.
