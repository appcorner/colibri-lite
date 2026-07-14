# ADR 0008: Defer Memory Mapping After M2

- Status: Accepted
- Date: 2026-07-14
- Milestone: M2.3

## Context

M2 correctness and residency goals pass with the portable reader: validated
ranges/hashes, on-demand experts, strict byte-budgeted LRU, lease safety,
deterministic metrics, corruption handling, and resident/streaming numerical
equivalence.

The release baseline for the complete open/seek/read-exact/SHA-256 path is
129.367 MiB/s for 1 MiB payloads over 200 iterations on the documented Windows
11/i7-1165G7/SSD machine. No end-to-end decode profile identifies artifact I/O
as a bottleneck. Streaming still decodes payload bytes into temporary F32
vectors, so mapping alone would not remove the principal payload copy.

## Decision

M2 ships with the portable `ArtifactReader`. Read-only memory mapping is
deferred and is not an M2 exit requirement. No `memmap2`, other mapping
dependency, OS FFI, or unsafe mapping code is added.

## Impact

Positive:

- M2 closes with the smallest correctness-proven storage boundary.
- Windows file handles remain short-lived and deterministic.
- No unmeasured optimization, dependency, or unsafe lifetime contract enters
  the runtime.
- The portable benchmark remains a stable comparison point.

Tradeoffs:

- Each cache miss opens, seeks, reads, hashes, and closes a file.
- Expert bytes are decoded/copied into F32 vectors before computation.
- A full-model profile may later show I/O or copies to be material.

## Reconsideration criteria

Mapping may be proposed only when at least one condition is measured on a
representative model/workload:

- artifact I/O accounts for at least 20% of end-to-end decode wall time;
- a reviewed mapping prototype reduces end-to-end decode latency by at least
  10% against the same workload;
- mapping removes a measured payload copy or materially simplifies residency
  ownership without increasing peak resident bytes.

Any proposal must rerun the exact portable benchmark and all correctness,
hash/corruption, byte-budget, lease, streaming-equivalence, Windows file
replacement/deletion, and mapping-lifetime tests. The boundary must be isolated
and receive dedicated unsafe/dependency review before retention.

## Evidence

- Portable benchmark: 129.367 MiB/s for 200 verified 1 MiB reads.
- Resident and streaming stages, expert IDs, and logits match at M1 tolerances.
- Strict two-expert budget forces deterministic eviction without output change.
- Oversized, truncated, and hash-invalid payloads fail before computation.
- No profile currently satisfies a reconsideration criterion.
