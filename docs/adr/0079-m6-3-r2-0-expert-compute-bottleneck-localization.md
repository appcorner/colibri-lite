# ADR 0079: M6.3-R2.0 Expert Compute Bottleneck Localization

## Status

Accepted for pre-registration. R2.0 is diagnostic-only and authorizes measurement instrumentation, not an optimization or M6.4 implementation.

## Context

ADR 0078 closed R1.4 `NO-GO for promotion` for the current direct safe-Rust group-32 Layer-0 candidate. R1.2 quality remains valid, but R1.3 showed no repeatable latency, throughput, working-set, or physical-I/O win. The only 5/5 directional improvement was total logical bytes, lower by 0.8919485285%.

The next question is therefore not whether to quantize more layers. It is which part of the current Layer-0 expert path consumes the time that prevents the packed representation from becoming runtime benefit.

## Decision

The pre-registered contract is `models/qwen3-30b-a3b/m6.3-r2-0-localization-contract-v1.json`, SHA-256 `02322820758848b5ebfe3f020db026f1aa5e3dd54ce007fb71202f50e54aceca`.

R2.0 will localize the expert-path cost using the unchanged F32 reference and unchanged admitted group-32 artifact. No candidate reselection, quantization change, cache-policy change, SIMD, FFI, native kernel, memory mapping, prefetch, threading, or arithmetic reordering is allowed before R2.0 closes.

## Measurement views

Two views are mandatory:

1. `compute_only_preloaded`: all selected Layer-0 experts are loaded and decoded before the timed boundary. Timed payload reads must remain zero. This isolates expert arithmetic, routing scan, and weighted accumulation.
2. `load_plus_compute`: software expert state is reset before each timed iteration. F32 uses the existing 18,874,368-byte `ExpertStore` budget; group-32 retains no candidate cache. OS filesystem cache is observed and not flushed. This measures software load/parse/decode plus compute without claiming physical-I/O attribution.

The held-out `short_english` and `short_thai` fixtures from R1.2 are both required. A reference-only fixture-freeze step must record the exact Layer-0 expert input tensor, selected expert IDs, router weights, and their SHA-256 identities before candidate localization timing starts.

Each fixture/view uses five independent release-process pairs with frozen order: candidate/reference, reference/candidate, candidate/reference, reference/candidate, candidate/reference. Each process performs two untimed warm-up iterations and ten timed iterations; `compute_only_preloaded` may use twenty-five timed iterations because no payload loads occur.

## Timing taxonomy

Candidate load scopes: seek, packed value read, packed value decode/validate, scale read, scale decode/validate. Candidate compute scopes: gate packed projection, up packed projection, activation/product, down packed projection. The packed projection scope explicitly includes the current fused scalar `i8 -> f32`, scale multiplication, and dot accumulation; R2.0 must not report a fictitious standalone dequantization time.

F32 scopes: cache lookup/load, F32 payload decode, gate projection, up projection, activation/product, and down projection. Both paths also record routing-occurrence scan, weighted accumulation, parent expert total, exclusive residual, calls, logical bytes, expert occurrences, and unique expert loads.

## Validity gates

- Instrumented and uninstrumented execution of each path must produce byte-identical output for the same frozen input; instrumentation may observe but not alter arithmetic or allocation order inside the measured algorithm.
- Timer overhead is calibrated with no-op scopes. A child stage is independently interpretable only when its median is at least 20 times the median no-op timer cost; otherwise it is merged into its parent for classification and never corrected by subtraction.
- Instrumentation observer effect is measured by paired instrumented/uninstrumented `compute_only_preloaded` totals. Median overhead must be at most 5% and every paired overhead at most 10% for both paths and both fixtures. Failure invalidates R2.0 localization.
- `compute_only_preloaded` must show zero timed candidate payload bytes and zero timed F32 expert-load bytes.
- `load_plus_compute` must reproduce the exact expected logical expert bytes and reset software cache state every timed iteration.
- Parent timing must be greater than or equal to the sum of nested child time. Exclusive residual is reported rather than hidden.
- Same release binary, host, candidate identity, F32 artifact identity, fixture identity, and timer implementation are required for all measured pairs.

## Classification rule

Cross-path stage families are `load_decode`, `gate_projection`, `up_projection`, `activation_product`, `down_projection`, `routing_accumulation`, and `exclusive_residual`.

For each process pair, report candidate-minus-reference nanoseconds and candidate stage share. Classification is based on medians across the five pairs for both fixtures:

- `packed_projection_compute_bound` when candidate `compute_only_preloaded` is slower in at least four of five pairs for both fixtures and gate+up+down packed projections account for at least 60% of candidate expert total in both fixtures.
- `load_decode_bound` when `load_plus_compute` is slower in at least four of five pairs for both fixtures, `load_decode` is at least 50% of candidate expert total in both fixtures, and `compute_only_preloaded` does not satisfy the compute-bound rule.
- `routing_accumulation_bound` when routing scan plus weighted accumulation is at least 30% of candidate expert total in both fixtures and is the largest positive cross-path delta family.
- otherwise `mixed_or_distributed`.

Classification selects only the hypothesis to design in R2.1. It does not authorize optimization implementation, all-layer rollout, or M6.4.