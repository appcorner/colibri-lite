# ADR 0088 - M6.3-R2.1 isolated native boundary review

Status: accepted for R2.1 Layer-0 vertical slice only.

## Context

ADR 0086 localized the current group32 slowdown to packed projection compute:
91.44% English and 92.04% Thai candidate expert time. ADR 0087 therefore
pre-registered one AVX2+FMA native gate/up/down projection hypothesis while
retaining Rust orchestration and the unchanged group32 artifact.

The workspace previously used `unsafe_code = "forbid"`. R2.1 requires one FFI
call, so the workspace lint changes to `deny`; only `r2_native.rs` carries a
module-level `allow(unsafe_code)`. No public unsafe API is introduced.

## Dependency review

`cc = 1.4.4` is admitted as a build dependency only. The resolved crate declares
MIT OR Apache-2.0, Rust 1.65 MSRV, and the rust-lang/cc-rs repository. Its
resolved transitive build dependencies are `find-msvc-tools 0.1.11` and
`shlex 2.0.1`. The project toolchain is Rust 1.96.1 with workspace MSRV 1.85.
The standard library is insufficient for this boundary because Cargo does not
compile and archive the required MSVC C translation unit itself. `cc` is used
only to invoke the platform compiler and emit one static library; it is not a
runtime dependency and does not enter inference ownership or scheduling.

## Safety boundary

Rust validates AVX2 and FMA support, exact input/value/scale/output lengths,
group size 32, finite inputs/scales, nonnegative scales, and the existing
forbidden `-128` payload rule before the FFI call. Rust owns all buffers for the
entire call and no pointer or ownership crosses or escapes the ABI boundary.

The C kernel performs no allocation, keeps only SIMD/register/local scalar
state, writes exactly one `f32` per output row, and has no static mutable state.
Therefore additional native scratch and persistent prepack are both zero bytes.
The unchanged packed i8 values and f32 scales are consumed directly; no complete
F32 weight is materialized.

The one Rust unsafe call documents these invariants immediately at the call
site. Canary tests guard output bounds; invalid-length tests prove rejection
before FFI. Unsupported hosts and group64 requests use the existing scalar path.
## Decision

Admit this dependency and unsafe boundary for R2.1 only. The default runtime
remains scalar unless the R2.1 feature is compiled and an explicit native
backend is selected. AVX-512, threading, prepacking, cache changes, and any
additional native functions remain unauthorized.

This review does not claim performance and does not authorize R2.1c, R2.2,
all-layer rollout, or M6.4. R2.1b held-out quality must close before any
official performance sample is allowed.
