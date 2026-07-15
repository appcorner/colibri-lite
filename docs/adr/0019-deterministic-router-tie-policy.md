# ADR 0019: Deterministic Router Tie Policy

- Status: Accepted
- Date: 2026-07-15
- Milestone: M4.2
- Task: M4.2-02
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

The pinned Transformers Qwen3-MoE router computes F32 softmax probabilities and
calls `torch.topk`. PyTorch 2.12.1 documents that indices for tied elements are
not guaranteed stable. The frozen Rust correctness path already ranks higher
scores first and lower expert IDs first when scores are exactly equal.

Changing Rust to imitate an unspecified ordering would reduce reproducibility
without improving model semantics. At the same time, BF16 reference execution
and F32 runtime execution can perturb router logits near the top-k boundary, so
expert IDs are only a sound exact oracle when that boundary is numerically
separated.

## Decision

Preserve the Rust ordering contract:

1. Higher router score ranks first.
2. Equal scores rank lower expert ID first.

This is a runtime determinism refinement, not a model-semantic change. Router
softmax, selected-probability normalization, tensor orientation, and projection
math remain unchanged.

For each token and validated layer, record selected logits, the kth selected
logit, highest unselected logit, absolute selection margin, selected expert
IDs, routing weights, and maximum Rust-versus-Transformers router-logit error.

Transformers expert IDs are exactly assertable only when:

```text
selection margin > max(documented router-logit error bound,
                       2 * measured maximum router-logit error)
```

When this condition holds, differing IDs are a real mismatch and validation
stops. When the boundary is tied or ambiguous, Transformers ordering is marked
non-assertable; router logits must still pass tolerance, and Rust IDs must match
the deterministic higher-score/lower-ID policy independently.

## Evidence requirements

- Strict-score, all-equal, selected-set tie, boundary tie, and repeat-run tests
  execute the Rust router implementation.
- Margin-policy tests reject safely separated ID disagreement and classify a
  tied/ambiguous upstream boundary as non-assertable.
- Real-token comparisons must report the complete boundary evidence before an
  exact expert-ID assertion is accepted.

## Consequences

The same internal router remains shared by resident and streaming execution.
No public API, artifact format, dependency, sorting rule, or numerical
tolerance changes. A future backend may produce a different upstream tie order,
but that is not treated as a Rust mismatch unless the boundary is safely
positive.
