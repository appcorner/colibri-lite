# ADR 0055: Planner prediction-error reporting

## Status

Accepted for M6.2-06.

## Decision

Planner predictions are compared with recorded observed decode throughput using
`(predicted - observed) / observed`. The comparison record preserves all input
paths and SHA-256 values, selected plan, workload, observed measurement, and
comparability assessment. A material mismatch is reported unchanged; no rate,
FLOP term, or ranking rule may be retuned from one observation.

## Consequences

The available M5.3 run is retained only as partial, non-promotable directional
evidence because its cache and storage behavior does not match M6.1/M6.2.
The 528.997% error is visible in the tracked report. A future promotion needs
a benchmark with matching plan/profile/cache/fixture identity and repeated
cold/warm observations.
