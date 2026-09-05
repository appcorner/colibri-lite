# ADR 0107: M6.3-R2.2-D4D2 Group16 Hybrid Quality NO-GO

## Status

Accepted. D4D2 closes `NO-GO`.

Result SHA-256:
`4a88b339991f9e5591a3b8c7f2a3665e83f74f830878c440ea8deaeb1453bb57`.

## Context

ADR 0106 pre-registered bilingual held-out quality for Layer0 group32,
Layer24 F32 gate/up plus group16 down, and Layer47/all other layers F32.
The frozen durable execution passed preflight and completed normally with
native exit code `101` after `793.70s`; this is a quality assertion failure,
not a transport failure.

## Decision

Close D4D2 `NO-GO`. The first failing assertion is `short_thai` Layer24
scalar-hybrid versus same-input/router F32 routed-output max-abs
`0.0012040138` against the unchanged `0.001` limit.

Do not relax the threshold, retry D4D2, or infer that prompt-only D4D1 evidence
generalizes to the propagated autoregressive trajectory.