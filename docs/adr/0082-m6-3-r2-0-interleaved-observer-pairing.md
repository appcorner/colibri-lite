# ADR 0082: M6.3-R2.0 Interleaved Observer Pairing

## Status

Accepted after the frozen v3 low-observer controls failed and before any R2.0
localization sample.

## Context

Coarse F32 timing removed the systematic observer bias seen in v1/v2, but v3
still showed block-level scheduling noise. Canonical v3 control evidence
SHA-256 is
`fbc1c782beea42a306973826a98a37d21700d3bbe423205f4f23de0c6942b4cf`.

The v3 group medians are near zero, while individual 25-iteration blocks range
from -26.6135% to +23.0779%. All outputs are byte-identical. No localization
sample or bottleneck classification exists.

The machine-readable method contract is
`models/qwen3-30b-a3b/m6.3-r2-0-measurement-method-v3-contract.json`, SHA-256
`ee2a85a1531956ad790e27c52bfb87cc7014c93dc4140d85576d020e5a45307f`.

## Decision

Keep the existing model code and coarse timing scopes. Change only the
observer-control estimator:

- Keep 20 fresh outer-pair processes: 2 fixtures x 2 paths x 5 pairs.
- Preload the workload once per process exactly as v3.
- Keep two warm-ups per observer state.
- Split each state's 25 timed iterations into five matched mini-blocks of five
  iterations each.
- For an outer pair whose base order is `enabled -> disabled`, mini-pair orders
  are `ED, DE, ED, DE, ED`; reverse that sequence when the base order is
  `disabled -> enabled`.
- Compute each mini-pair overhead as `100 * (enabled - disabled) / disabled`.
- Define the outer-pair overhead as the median of its five mini-pair overheads.
- Preserve the five frozen outer-pair orders.
- Preserve the numerical observer gates: every outer-pair overhead <= 10% and
  the median of five outer-pair overheads <= 5% for every fixture/path group.
- Require byte-identical output for both states in every mini-pair.
- No retries, outlier deletion, timer subtraction, or post-hoc replacement.

This retains 25 timed iterations per state and makes the estimator robust to a
single scheduler interruption without relaxing the acceptance limits.

## Frozen invariants

This method revision may not change:

- localization contract or frozen fixtures;
- F32 coarse scopes introduced by ADR 0081;
- candidate timing implementation;
- routing, accumulation, load/decode accounting, arithmetic, quantization, or
  cache behavior;
- localization matrix or classification thresholds;
- R1.2 quality and R1.3/R1.4 evidence.

If this observer validation fails, R2.0c remains blocked and another explicit
measurement-method review is required.
