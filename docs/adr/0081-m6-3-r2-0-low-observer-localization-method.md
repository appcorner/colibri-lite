# ADR 0081: M6.3-R2.0 Low-Observer Localization Method

## Status

Accepted after both pre-localization observer matrices failed and before any
R2.0 localization sample.

## Context

Observer controls v1 failed because fine-grained F32 row timing added roughly
9-10% median overhead. ADR 0080 reduced F32 clock reads, but controls v2 still
failed: English/F32 median 6.7789%, Thai/F32 median 5.5028%. A candidate Thai
pair also showed a 36.4% separate-process A/B outlier even though candidate
median overhead remained near zero.

No localization sample or bottleneck classification exists. Therefore a
measurement-method revision can be preregistered without adapting to any
localization result. The machine-readable method contract is
`models/qwen3-30b-a3b/m6.3-r2-0-measurement-method-v2-contract.json`, SHA-256
`b2df0dd3ae10028b669e0d31aaf77794eef8015e1411166a0c5a08d9ab263e94`.

## Decision

Use coarse F32 stage timing and within-process observer pairs.

- F32 gate/up/activation keeps the exact existing interleaved arithmetic loop,
  but records one coarse `f32_gate_up_activation_combined` scope per expert MLP.
- F32 down projection records one coarse `down_projection` scope around the
  unchanged complete down loop.
- Candidate gate/up/activation/down timing remains unchanged.
- Routing scan, weighted accumulation, load/decode accounting, expert selection,
  quantization, cache policy, and arithmetic remain unchanged.
## Observer-control method

For each fixture/path, run five fresh processes. Each process contains both
observer states in the frozen order for that pair: enabled/disabled,
disabled/enabled, enabled/disabled, disabled/enabled, enabled/disabled.

Each state receives two untimed warm-ups followed by 25 timed
`compute_only_preloaded` iterations using the same preloaded expert state.
Output hashes must match exactly. Timed payload reads remain zero.

The unchanged gates remain:

- median enabled-minus-disabled overhead <= 5%;
- every pair overhead <= 10%;
- byte-identical output in every pair.

No failed pair is retried. Failure stops R2.0 localization.

This yields 20 observer-control processes instead of 40 process-separated
subruns while preserving five independent fresh-process pairs for every
fixture/path group.

## Localization representation

Reference raw compute scopes become:

- `f32_gate_up_activation_combined`;
- `down_projection`;
- `routing_occurrence_scan`;
- `weighted_accumulation`;
- explicit exclusive residual.
Candidate raw scopes remain gate/up/activation/down packed projections plus
routing/accumulation and residual. For cross-path reporting, candidate
gate+up+activation is compared with the combined F32 scope. Down remains a
separate cross-path family.

The prospective classification thresholds do not change:

- compute-bound still requires candidate compute-only slower in >=4/5 pairs for
  both fixtures and candidate gate+up+down packed projections >=60% of
  candidate expert total;
- load/decode remains >=50% with the existing slower-pair condition;
- routing+accumulation remains >=30% and the largest positive delta family;
- otherwise classify mixed/distributed.

Localization still uses the original 40 fresh process samples, original
fixture/view/path pair order, two warm-ups, 25 compute-only iterations, ten
load-plus-compute iterations, no automatic retry, and no physical-I/O claim.

## Guardrails

The timer remains `std::time::Instant`. Timer subtraction, SIMD, FFI, native
kernels, arithmetic reordering, new quantization, cache-policy changes, and
prefetch remain prohibited. This ADR authorizes measurement implementation and
one observer-method validation only; it does not authorize optimization, R2.1,
or M6.4.
