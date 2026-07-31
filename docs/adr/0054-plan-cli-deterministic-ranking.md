# ADR 0054: Plan CLI and deterministic ranking

## Status

Accepted for M6.2-05.

## Decision

`clr-cli plan` accepts explicit hardware/model profile paths, RAM/VRAM/context
budgets, workload lengths, a caller-supplied model-derived GFLOP-per-token
term with a required source string, request/result IDs, and output path. It
hashes both input profile files and emits the versioned planner-result JSON
document with the complete compute-work input preserved in its request.

The current M6.1 profile supports CPU RAM and SSD placement only. Feasible
candidates rank by predicted decode tok/s descending, then lower quality risk,
then lower startup cost, then stable plan ID. Ranking validates all fields
before sorting; malformed data is an error, never a fallback score.

## Consequences

The command is reproducible from explicit inputs and cannot silently probe the
host or promote an unavailable GPU. `--context-tokens` must be greater than
zero. `--compute-gflop-per-token` and `--compute-work-source` are required
because M6.1's frozen model profile has no FLOP-per-token fact. Its output is
an `analytical-v1` prediction, not measured end-to-end performance. M6.2-06
must quantify prediction error with a recorded benchmark set.
