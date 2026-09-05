# ADR 0103: M6.3-R2.2-D4 Hybrid Held-Out Quality NO-GO

## Status

Accepted. D4 closes `NO-GO`.

Result SHA-256:
`47dbccb7406ee12fcca5b0a4b48790a749c54388fc5780f5dfbb6765dd012822`.

## Context

ADR 0102 pre-registered held-out quality for the exact D3 policy: Layer0 all-group32, Layer24 F32 gate/up plus group32 down, Layer47 F32. The frozen execution completed normally with native test exit code 101; this was a quality assertion failure, not a transport failure.

The first failing assertion was `short_english` Layer24 scalar-hybrid versus same-input/router F32 local max-abs `0.0010073595` against the frozen `0.001` limit.

## Decision

Do not relax the `0.001` budget and do not retry D4. The D3 `down`-only group32 selection is not held-out-generalized because D3 characterized only `short_thai`.

D5 is not authorized. Historical R2.2c remains inapplicable. All-layer rollout and M6.4 remain blocked.

The next task is a separately pre-registered Layer24 down-projection precision characterization across both held-out fixtures. It may evaluate only already-existing group32/group16/group8 artifacts, under canonical-F32 and Layer0-group32 prefixes, without performance timing or post-result reselection.

## Consequences

- The miss is numerically small but contractually real: `7.3595e-6` absolute above the frozen limit.
- No conclusion is made about generation/top20/native repeatability because execution stopped at the first local scalar/F32 gate.
- D4 quality harness resource/I/O behavior remains non-evidence by design.
