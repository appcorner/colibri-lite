# ADR 0087: M6.3-R2.1 Native Packed Projection Hypothesis

## Status

Accepted as a pre-registration before any R2.1 native implementation, quality
result, or performance result.

The machine-readable contract is
`models/qwen3-30b-a3b/m6.3-r2-1-native-packed-projection-contract-v1.json`,
SHA-256
`76bfc4a850a8d89fb5901eda338568b7226b36acb7207324575568ea21f6cc2a`.

## Context

ADR 0086 closed R2.0 as `packed_projection_compute_bound`. On both held-out
fixtures the current group32 scalar compute-only path was slower than F32 in
5/5 pairs, with median slowdowns +77.35% English and +85.79% Thai. Packed gate,
up, and down projections consumed median 91.44% and 92.04% of candidate expert
time. The same candidate won every load-plus pair, so storage/load is not the
primary remaining slowdown.
The target host is Windows x64 on an Intel i7-1165G7. The existing M6.1 profile
records 4 cores / 8 logical processors; a pre-implementation runtime feature
probe reports AVX2 and FMA available. AVX-512 is also present but is explicitly
out of scope for this first proof.

## Decision

Test one hypothesis only: retain Rust orchestration and the unchanged group32
artifact, but replace the scalar `R1_1PackedProjection::apply_direct` dot loop
for gate/up/down with one AVX2+FMA native kernel consuming the original i8
values and f32 group scales directly.

The native kernel may change reduction order and use FMA. It may not change
routing, activation, expert selection, quantization, artifact layout, load
policy, cache policy, or persistent representation. It may not materialize a
complete F32 weight matrix or create a persistent prepacked artifact.

R2.1 remains Layer-0 only. AVX-512, GPU, threading, prefetch, new quantization,
and all-layer expansion require separate evidence and review.
## Native and unsafe boundary

This milestone explicitly requires one isolated native boundary because R2.0
measured material need and a safe scalar implementation already exists and
passes correctness tests, satisfying the AGENTS.md prerequisites for review.

R2.1a may change the workspace unsafe lint from `forbid` to `deny`, keeping it
enabled globally, then allow unsafe only inside the smallest FFI module. Rust
must perform no pointer arithmetic. The FFI wrapper validates CPU features,
lengths, group size, output capacity, and finite inputs/scales before the call;
ownership never crosses FFI; each unsafe block documents lifetime, bounds, and
aliasing invariants; the module receives a dedicated unsafe review.

The proposed build-only dependency is exact `cc = 1.4.4`, MIT OR Apache-2.0,
MSRV 1.65, used only to compile the static native kernel with the platform C/C++
compiler. Dependency addition itself remains subject to the repository's
license/MSRV/Windows/transitive-dependency review during R2.1a.

Unsupported CPUs must use the unchanged scalar group32 fallback. No unsafe API
may leak into `clr-core` or public runtime contracts.
## Quality gate before timing

Official R2.1 performance timing is prohibited until native quality passes on
both held-out fixtures. Preserve the R1.2 gates: exact generated sequences,
prompt top-20 IDs and argmax, exact safe-margin router guards at layers 0/24/47,
exact repeatability, F32 logit max-abs envelope 0.05 plus the frozen top-1
margin rule, and zero complete F32 weight materializations.

Because AVX/FMA changes reduction order, also require native versus current
scalar group32 max-abs error <=0.001 at Layer-0 routed output and <=0.002 at
prompt logits. Any quality failure closes this R2.1 candidate NO-GO without
official performance timing.

## Performance proof

After quality PASS, compare three paths in one frozen release binary:
`native_avx2_fma_group32`, unchanged `scalar_group32`, and `reference_f32`.
Use both R2.0 fixtures and both views. For each fixture/view run all six
permutations of the three paths exactly once: 72 fresh process samples total,
with no automatic retry. Keep 25 timed compute-only iterations and 10 timed
load-plus iterations with the same software-state reset semantics.
Compute-only must pass on both fixtures: native faster than scalar in >=5/6
triplets, median scalar/native speedup >=1.50x, native/F32 median ratio <=1.05,
and native/F32 <=1.10 in >=5/6 triplets.

Load-plus must pass on both fixtures: native faster than scalar in >=5/6,
median scalar/native speedup >=1.25x, native faster than F32 in 6/6, and
native/F32 median ratio <=0.50. Candidate payload-byte accounting must remain
identical to scalar group32; no new physical-I/O claim is authorized.

Additional native scratch is capped at 1 MiB per projection invocation. No
persistent native prepack and no VRAM use are allowed.

## Exit decision

If all quality/resource/performance gates pass, R2.1 may authorize **R2.2 design
only** for expanded impact modeling/proof. It does not authorize R2.2
implementation, all-layer rollout, or M6.4. A failed gate closes this candidate
NO-GO or requires a separately pre-registered compute hypothesis; thresholds
must not be weakened after results are seen.