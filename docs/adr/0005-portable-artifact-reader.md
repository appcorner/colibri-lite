# ADR 0005: Portable Artifact Reader Contract

- Status: Accepted
- Date: 2026-07-14
- Milestone: M2.1

## Context

Expert residency requires validated tensor ranges that can be loaded without
keeping an entire model file resident. The first reader must work on Windows
x64 with ordinary file APIs before optional memory mapping is evaluated.

Manifest serialization is not yet required by the runtime path: the M4
converter can construct the validated manifest contract. Adding JSON, hashing,
or memory-mapping dependencies now would expand supply-chain and unsafe review
before the storage semantics are proven.

## Decision

Artifact format version 1 is represented by a validated `ArtifactManifest`.
Each `TensorMetadata` records:

- unique stable name;
- dense shape and dtype;
- root-relative file path;
- byte offset and exact byte length;
- SHA-256 of the tensor payload.

Only little-endian payloads are accepted. Manifest construction rejects unsafe
paths, duplicate names, range overflow, shape-derived length mismatch, and
overlap within one file.

`ArtifactReader` canonicalizes the root and each tensor file, rejects paths
escaping the root, opens the file for one read, checks file length, seeks to the
declared range, reads exactly that range, verifies SHA-256, and closes the file
before returning owned bytes.

A dependency-free SHA-256 implementation is isolated in `clr-storage` and
tested against published empty-string and `abc` vectors. It is used for
artifact integrity, not authentication.

## Invariants

- Format version is exactly 1.
- Numeric byte order is little endian.
- Tensor names are unique and non-empty.
- Paths contain only relative normal components.
- Shape/dtype byte count equals declared payload length.
- Non-empty ranges in the same file do not overlap.
- Offset plus length cannot overflow.
- Canonical tensor paths remain below the canonical root.
- Truncated or hash-mismatched bytes are never returned.
- File handles do not outlive `read_tensor`.

## Consequences

Positive:

- Storage behavior is portable and testable before memory mapping.
- Expert files can be read independently by name and range.
- Corruption and malformed metadata fail with structured errors.
- No dependency, `unsafe`, mapping lifetime, or serialization format is added.

Tradeoffs:

- The manifest is currently a Rust contract, not an on-disk JSON parser.
- Hash verification reads the requested payload before it can enter a cache.
- Files are opened per read; M2.2 metrics will expose the resulting I/O.
- SHA-256 implementation maintenance remains local until dependency evidence
  justifies replacement.

## Evidence

- Published SHA-256 vector tests pass.
- Valid non-overlapping manifest test passes.
- Version, endianness, duplicate, traversal, length, overflow, and overlap
  rejection tests pass.
- Exact range reading, unknown tensor, truncation, and corruption tests pass.
- A Windows-specific test deletes the backing file immediately after a read,
  proving the file handle was released.
- Storage Clippy passes with warnings denied.
