# ADR 0009: Token Sampling Contract

- Status: Accepted
- Date: 2026-07-14
- Milestone: M3

## Context

M3 requires deterministic greedy decoding before probabilistic sampling. The
initial API accepts token IDs directly and must remain reproducible without an
external RNG dependency. KV caching is a later task; the first generation path
may recompute the complete sequence to isolate token-selection correctness.

## Decision

`greedy_token` selects the maximum score from the final logits row. Equal scores
select the lower token ID.

`SeededRng` implements documented SplitMix64 with a fixed state transition and
output transform. Temperature sampling:

- requires finite temperature greater than zero;
- divides final-row logits by temperature;
- applies max-subtracted F32 softmax;
- draws from a 24-bit uniform F32 value from SplitMix64;
- traverses token IDs in ascending order.

`Qwen3MoeModel` exposes recomputing greedy and temperature generation methods
that return only newly generated token IDs.

## Public API surface

- `SeededRng::{new, next_u64}` defines the pinned sampling state and observable
  deterministic output stream.
- `greedy_token` and `sample_token` select from the final logits row.
- `Qwen3MoeModel::{generate_greedy, generate_temperature}` provide the
  correctness-first recomputing generation path.

No existing public sampling or generation API was removed or changed because
these are the first generation contracts.

## Invariants

- Empty prompts and empty/non-matrix logits fail with structured errors.
- NaN or infinite scores never produce a token.
- Greedy tie behavior is stable.
- The same seed, prompt, temperature, weights, and step count produce exactly
  the same sequence.
- Sampling does not weaken M1 numerical tolerances or change model forward math.

## Evidence

- Frozen prompt greedy token equals oracle argmax token 10.
- Greedy hand-value/tie, rank, empty, non-finite, and repeatability tests pass.
- First three SplitMix64 outputs for seed zero match pinned hexadecimal values.
- Same-seed temperature sequences compare exactly equal.
- Invalid temperature returns a structured error.
