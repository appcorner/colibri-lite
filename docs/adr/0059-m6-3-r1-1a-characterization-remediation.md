# ADR 0059: M6.3-R1.1a Characterization Remediation

## Status

Proposed before R1.1a conversion. This is append-only: ADR 0058 and the R1.1
`no_candidate_admitted` result remain historical evidence.

## Decision

Evaluate only `cpu-safe-rust-int8-group64-layer0-r1-1a` and
`cpu-safe-rust-int8-group32-layer0-r1-1a` against held-in
`tier_a_control` and `tier_b_code_newline`. `short_english`, `short_thai`,
all held-out bilingual fixtures, R1.2, and M6.4 are prohibited.

Before conversion, candidate ranking is fixed as: (1) lower maximum absolute
fixed-logit error; (2) lower maximum routed-expert/MoE checkpoint error; (3)
smaller emitted artifact bytes; (4) lexical candidate ID. The cap remains
`logit_max_abs <= 0.05`; it cannot be changed after results exist. A candidate
also needs exact routed-expert IDs, finite named checkpoints with recorded
shape/tolerance, first-divergence when a mismatch exists, direct packed
consumption with no complete F32 projection/expert materialization, and
repeated-run determinism.

No dependency, unsafe code, GPU/backend, mmap, FFI, SIMD, or external layout
is introduced. Each candidate gets a new flat temporary run directory after a
`DriveInfo.AvailableFreeSpace` preflight; output is `.incomplete`, synced,
hashed, atomically promoted inside that run, recorded, then removed only after
cleanup dry-run and apply.
