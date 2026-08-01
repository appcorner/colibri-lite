# ADR 0065: M6.3-R1.1a Candidate-Correlated Telemetry

## Status

Accepted before any R1.1a candidate conversion or execution. This closes a
harness contract only; it admits no candidate and does not authorize R1.2.

## Decision

Every R1.1a candidate execution must emit one
`m6.3-r1.1a-candidate-run-telemetry-v1` record bound to exactly one
pre-registered candidate, one flat temporary run directory, the promoted
temporary artifact's exact path/size/SHA-256, and the persisted runner's child
PID. The only permitted fixture is frozen `code_newline` with token IDs
`[87, 28, 16, 198]`.

The runtime record separates the full-file SHA-256 verification scan from
inference payload reads. It records packed-expert peak bytes and requires zero
complete F32 weight/projection/expert materializations. External process
telemetry must contain at least one sample and report working-set and private-
bytes peaks. Physical I/O is admissible only when Kernel-File and Kernel-Disk
events are joined by exact PID, candidate path/FileObject, and nonzero Disk
read bytes with zero lost events. Logical reads or disk-wide counters cannot
substitute for that witness.

The candidate artifact reader validates the pre-registered group size, exact
canonical artifact length, SHA-256, 64-byte offsets, expert bounds, absence of
the forbidden `-128` value, finite nonnegative F32 scales, and checked read
accounting before direct packed consumption. It does not modify the public API
or add a dependency, unsafe code, mmap, SIMD, FFI, or a new backend.

## Consequences

An execution with missing or mismatched PID/path/hash, uncorrelated physical
I/O, missing memory samples, or any complete F32 weight materialization is
invalid and cannot contribute to ranking or admission. Candidate conversion
remains prohibited until the contract and failure-mode tests pass. R1.2 and
held-out bilingual fixtures remain blocked.
