# ADR 0089: M6.3-R2.1 held-out quality GO

Status: accepted.

## Context

ADR 0087 pre-registered an AVX2+FMA native group32 packed-projection vertical
slice and required held-out quality to pass before any official performance
timing. ADR 0088 admitted the isolated FFI boundary and R2.1a implementation.
The quality harness and execution identities were frozen before execution.

The frozen execution used source commit
`20861888a832279768d688dcaf4f1417a15a2f80`, release binary SHA-256
`f0cafe2b6bb8b0f4302e1b7c7e3dcfdde95b250cbc7f59e7860ae63fb9c69210`,
and the unchanged admitted group32 artifact SHA-256
`777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2`.

## Evidence

The held-out quality test passed 1/1 on `short_english` and `short_thai`.
Evidence SHA-256 is
`f0dea27297c03b2f2283c161c91b369169318c309394c1bb786db43a02ba6113`.
Result SHA-256 is
`8fe232bfc58381e4a3d602e67705e129dafcf133e785c6c15ca6770e1cf22866`.
Both fixtures preserved the exact two-token sequence, prompt argmax, prompt
top-20 IDs, Layer-0/24/47 router guards, and exact native repeatability.
Native-vs-scalar Layer-0 max-abs error was `2.3841858e-7` for both fixtures,
well below the frozen `0.001` limit. Native-vs-scalar prompt-logit max-abs was
`3.0517578e-5` English and `2.5749207e-5` Thai, below the frozen `0.002` limit.
The existing F32 quality envelope also remained satisfied.

## Decision

Close M6.3-R2.1b `GO` and authorize R2.1c official performance measurement
only. This does not claim a speedup and does not authorize R2.2 implementation,
all-layer rollout, or M6.4.

R2.1c must use one frozen three-path release binary and the 72-sample balanced
native/scalar/F32 matrix defined by the R2.1 contract. Performance thresholds,
resource limits, triplet order, and no-retry semantics remain unchanged.