# ADR 0080: M6.3-R2.0 Observer-Effect Measurement Remediation

## Status

Accepted for one measurement-only remediation attempt after the frozen v1
observer-control set failed and before any R2.0 localization sample.

## Context

Execution manifest v1 completed all 40 observer-control processes. Candidate
controls passed on both fixtures, while F32 reference controls failed the
unchanged observer-effect gate. The canonical failed-control evidence SHA-256
is `140c219c846af500cc99bd352c3d8b3215571048e3cd02271f414bf84d28d64c`.

No localization sample exists. The localization contract, candidate quality,
R1.3/R1.4 evidence, and all performance classification thresholds remain
unchanged. The append-only remediation contract is
`models/qwen3-30b-a3b/m6.3-r2-0-observer-remediation-contract-v1.json`, SHA-256
`7512369048ff4385bae38274f8e272eade02433d680ae01db0ebb0f12a46e714`.

The F32 expert loop interleaves gate dot, up dot, and activation per row. The
v1 instrumentation uses separate `Instant::now`/`elapsed` pairs for each
operation, plus separate start/elapsed pairs for every down row. The measured
observer overhead demonstrates that this clock-read pattern is too intrusive.

## Decision

Allow exactly one append-only measurement-instrumentation remediation:

- Preserve the existing arithmetic loop and operation order exactly.
- Preserve `std::time::Instant` as the timer implementation.
- For each F32 gate/up/activation row, use shared timestamp boundaries:
  `t0 -> gate -> t1 -> up -> t2 -> activation -> t3`; derive the three stage
  durations from adjacent timestamp differences.
- For the F32 down loop, take one initial timestamp and one timestamp after each
  unchanged dot operation; each row duration is the adjacent difference.
- Continue batching timing samples into the existing collector after compute.
- Candidate timing implementation remains unchanged.
- Timer-overhead subtraction remains prohibited.

This change reduces clock reads but does not reorder, duplicate, replace, or
optimize any model computation.

## Frozen invariants

The remediation may not change:

- localization contract SHA-256 `02322820758848b5ebfe3f020db026f1aa5e3dd54ce007fb71202f50e54aceca`;
- fixtures or frozen reference fixture record;
- stage taxonomy or cross-path stage families;
- two warm-ups and 25 compute-only timed iterations;
- observer pair order;
- median overhead maximum 5% and every-pair maximum 10%;
- output byte-identity gate;
- prospective R2.0 classification rules;
- quantization, cache policy, expert selection, or arithmetic semantics.
