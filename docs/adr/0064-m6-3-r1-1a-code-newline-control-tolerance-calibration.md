# ADR 0064: R1.1a code_newline control tolerance calibration

## Status

Accepted before candidate conversion or candidate execution. This ADR corrects
only the numerical-budget linkage used when the canonical Rust F32 control is
compared with the frozen R1.1b `code_newline` oracle. ADR 0063's checkpoint
payload, names, shapes, dtypes, hashes, and exact selected-expert IDs remain
unchanged.

## Context

ADR 0063 linked its float checkpoints to existing M4.2 tolerance identifiers.
The first full-path control diagnosis established that the Layer-0
`expert_down_projection` value `2.2964e-6` came from ADR 0024's different M4.2
fixture (`[9707, 11, 1879, 0]`). ADR 0024 and the tolerance registry explicitly
classify that numerical value as fixture-specific. Applying it unchanged to
`code_newline` (`[87, 28, 16, 198]`) was therefore a contract-linkage error,
not evidence of a runtime or candidate failure.

The diagnosis occurred before either R1.1a candidate was reconverted or run.
No group-32 or group-64 checkpoint, logit, telemetry, or admission result
exists. One control diagnostic observation (`2.503395081e-6` maximum selected
expert output error) is already known. The calibration formulas below are
copied from the accepted M4.2 formulas rather than fitted to that observation.

## Decision

Run exactly two independent release-process canonical F32 controls against the
unchanged frozen R1.1b `code_newline` payload. Record the maximum finite
absolute error from either run for every float checkpoint. Exact selected IDs,
ordinary-versus-override equality, runtime-order aggregation replay, artifact
hashes, and final fixed-logit guards remain mandatory.

Freeze `code_newline` control budgets using these pre-registered formulas:

| Checkpoint | Budget formula |
| --- | --- |
| post-attention RMSNorm / expert input | `3 * max_observed_error + 5e-7` |
| router logits | `3 * calibrated_expert_input_error + 1.430511474609375e-5` |
| routing weights | `0.5 * calibrated_router_logit_error + 1e-7` |
| selected expert down output before weighting | `3 * max_observed_error + 1e-6` |
| aggregated MoE output | `3 * max_observed_error + 1e-6` |

All arithmetic is performed in binary64 for the record, then the stored bound
is rounded upward to the next representable binary32 value used by Rust. NaN
or infinity, an exact-ID mismatch, an internal exact mismatch, a source or
reference hash mismatch, or disagreement between the two runs' deterministic
checkpoint hashes is a stop condition and produces no calibrated contract.

The generated record must preserve both raw observations and derived budgets,
identify the two persisted process-evidence records and their log hashes, and
be validated independently. `layer0.block_output` remains an internal
ordinary-versus-override exact diagnostic and is not added to the frozen
oracle or admission gates.

## Consequences

- M4.2 fixture-specific numerical values are not silently promoted to another
  fixture.
- The checkpoint semantics and oracle payload frozen by ADR 0063 are retained.
- Calibration uses only the canonical F32 control and occurs before candidate
  conversion, so group-size selection cannot influence the budgets.
- The R1.1a fixed-logit cap remains `0.05`; held-out bilingual fixtures, R1.2,
  and M6.4 remain prohibited.
- Candidate reader, telemetry, conversion, and admission remain blocked until
  this control calibration and the remaining harness gates pass.
