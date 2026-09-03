# ADR 0085: M6.3-R2.0 Zero No-Op Timer Calibration Correction

## Status

Accepted after the 40-sample immutable set and Git-OID validator correction,
but before any successful localization classification.

## Context

The corrected validator next rejected `sample timer calibration`. Inspection of
the immutable sample set showed 36/40 samples with
`timer_noop_median_nanos = 0` and 4/40 with `100` ns.

The preregistered contract does not require a positive no-op median. It defines
child-stage interpretability as `child_median >= 20 * noop_median` and forbids
timer-overhead subtraction. A zero median is therefore a valid timer-resolution
observation, not missing evidence.
## Decision

Allow one validation-only correction:

- accept integer no-op median values `>= 0`;
- preserve the frozen `20 * noop_median` interpretability rule exactly;
- do not substitute a positive floor, subtract timer overhead, or modify any
  recorded stage duration;
- rerun classification only on the same immutable sample document;
- do not rerun any benchmark process.

All observer gates, timing scopes, pair order, byte accounting, classification
thresholds, fixtures, binary identity, and sample bytes remain unchanged.

Machine-readable correction contract:
`models/qwen3-30b-a3b/m6.3-r2-0-timer-calibration-correction-v1.json`, SHA-256
`54d7d397cceb3214f777b3312aa6f52d0d81874fb44971651d36dfa7478790b4`.

## Frozen evidence

- localization samples SHA-256:
  `e38688ae1309c411c11309def37b5b2ef6e84eb634dee244bb565a41b7eb7942`
- validator after ADR 0084 SHA-256:
  `b4dfef7e332bfa3197b2877696df3ffff6d00136f4a688d937f4ea2ef247f6ad`
- zero no-op samples: `36/40`
- 100 ns no-op samples: `4/40`
- classification seen before this correction: `false`
- R2.1 remains unauthorized; M6.4 remains blocked.