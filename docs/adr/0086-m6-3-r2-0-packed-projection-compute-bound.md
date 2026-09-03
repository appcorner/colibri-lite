# ADR 0086: M6.3-R2.0 Packed Projection Compute-Bound Closure

## Status

Accepted after the immutable 40-sample R2.0 localization set validated under
the pre-registered ADR 0079 classification rules.

## Evidence

- localization samples SHA-256:
  `e38688ae1309c411c11309def37b5b2ef6e84eb634dee244bb565a41b7eb7942`
- localization result SHA-256:
  `044e6d1d6a5b59953c2b603489d24e304290d5ce0d85d82b4de6d4c8e5b362cb`
- final release binary SHA-256:
  `aced59ad9ffa9ed153f1d817f37beeda521061f0d2025935aab8c341889efb8e`
- final observer controls v5 SHA-256:
  `24d1801dc4e15948552dc29abdc700dd81a7866f497ef1bf0d14807132fe237d`
- corrected validator SHA-256:
  `62f84cdf22b831c5423e5ba182962217394466bb34f107e2cd6b5eb95c3c69fa`

ADR 0084 and ADR 0085 record two validation-only corrections. Neither changed
measurement samples, observer gates, timing scopes, classification thresholds,
or candidate/reference execution.
## Decision

Close R2.0 as `packed_projection_compute_bound`.

The pre-registered compute-bound conditions are satisfied for both held-out
fixtures:

- candidate compute-only total is slower in 5/5 English pairs and 5/5 Thai
  pairs (required: at least 4/5);
- candidate gate + up + down packed projections account for median 91.44% of
  English and 92.04% of Thai expert time (required: at least 60%).

Compute-only median slowdown versus F32 is +77.35% English and +85.79% Thai.
By contrast, load-plus-compute candidate total is faster in all five pairs for
both fixtures, with median deltas -88.42% English and -90.39% Thai. The result
therefore does not support load/decode as the primary group32 slowdown source.

R2.1 hypothesis **design** is authorized. Optimization implementation is not.
M6.4 remains blocked pending a separately pre-registered R2.1 vertical-slice
hypothesis, quality guard, paired performance evidence, and all-layer review.