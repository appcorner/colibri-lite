# ADR 0078: M6.3-R1.4 Re-entry Review

## Status

Accepted. R1.4 closes `NO-GO` for promotion of `cpu-safe-rust-int8-group32-layer0-r1-1a` beyond the measured Layer-0 slice.

## Inputs

R1.2 quality remains `GO` under ADR 0071. The candidate preserves the held-out bilingual correctness gates and its frozen artifact identity.

R1.3 closes with a valid 20-sample paired measurement set under ADR 0077. The result artifact is `models/qwen3-30b-a3b/m6.3-r1-3-paired-result-v1.json`, SHA-256 `491ea0490e400e6563807ec9000eee2cbe78f908d5c4ce709ef18b5f32f2d04b`.

The R1.4 decision record is `models/qwen3-30b-a3b/m6.3-r1-4-reentry-review-v1.json`, SHA-256 `49e470254afb64cbd40426f0c833eff57e9ce787f28f638ecd7ab2ff98ada041`.

## Review

The candidate achieves a deterministic 0.8919485285% reduction in total logical bytes in both declared cache conditions. That byte reduction is real but is not accompanied by a consistent runtime-resource benefit.

Cold median candidate deltas are `+2.8183%` timed wall, `+9.1717%` TTFT, `-8.4011%` prefill tok/s, and `-3.4114%` decode tok/s. Warm medians are `+8.7446%`, `+7.4820%`, `-6.9612%`, and `-6.6696%`, respectively.

Working-set and private-byte deltas are effectively flat to slightly worse and fail the 5/5 directional-win rule. Physical I/O is cache-sensitive and also fails the directional-win rule.
## Decision

Do not promote the current group-32 direct safe-Rust candidate into an all-layer implementation. R1.4 therefore closes the current re-entry path as evidence-only `NO-GO for promotion`.

This does not invalidate the R1.2 quality result or the packed format as a characterization artifact. It means the current direct dequantization/consumption path does not convert its byte reduction into material, repeatable end-to-end benefit under the reviewed host and protocol.

M6.4 remains blocked. No all-layer rollout, candidate reselection, or cache-policy change is authorized by this review.

A future re-entry requires a new pre-registered plan with a materially different runtime hypothesis, such as an optimized native dequant/GEMM backend, measured overlap/placement strategy, or another explicitly reviewed bottleneck intervention. It must not reuse favorable samples or weaken the established quality/resource gates.