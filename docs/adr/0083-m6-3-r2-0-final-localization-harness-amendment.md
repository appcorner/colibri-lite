# ADR 0083: M6.3-R2.0 Final Localization Harness Amendment

## Status

Accepted after observer controls v4 passed and before any R2.0 localization
sample.

## Context

Interleaved observer controls v4 passed on release binary SHA-256
`432ff725e6598b9dbc6739ecd36ef52a38d13e094c7275a96fa05fe8b01a4f70`.
Canonical v4 controls SHA-256 is
`a6cf0d7518e889ddc27234de7a45100a0c8363e31fa0a3eec61dc29feed3a687`.

Before the first localization sample, an execution-readiness audit found two
measurement-harness gaps:

- the frozen binary has compute-only paths but no `load_plus_compute` sample
  entrypoint required by the original localization contract;
- the localization validator still maps reference F32 gate/up/activation to
  fine-grained scopes, while ADR 0081 deliberately replaced those observations
  with `f32_gate_up_activation_combined` to reduce observer overhead.

No localization sample or bottleneck classification exists.

The machine-readable amendment contract is
`models/qwen3-30b-a3b/m6.3-r2-0-localization-harness-amendment-v1.json`, SHA-256
`c3409036b22b85ab176477e662d26c15fa18cede724e57f8ea61ecc3ddc75894`.

## Decision

Allow one measurement-only final-harness amendment before localization:

- add the missing `load_plus_compute` runner using existing F32 streaming and
  group32 direct-consumption paths;
- reset software expert state each load-plus iteration as preregistered;
- keep OS filesystem cache observed and uncontrolled;
- update validator reference timing to the coarse F32 scope without inventing
  gate/up/activation splits;
- compare gate+up+activation cross-path as one combined family;
- preserve candidate packed-projection share classification using gate + up +
  down packed projection time only, excluding activation exactly as originally
  preregistered;
- add no model-compute, quantization, cache-policy, arithmetic, or performance
  optimization.

The v4 observer PASS is retained as historical evidence but does not authorize
localization on a newly built binary. After this amendment, freeze a final
binary/execution manifest and rerun the same ADR 0082 observer controls on that
exact binary. Only a fresh PASS may authorize the 40 localization samples.

## Frozen invariants

The original 40-sample matrix, fixtures, pair order, iteration counts, 5%/10%
observer gates, byte-identity requirement, classification thresholds, no-retry
rule, and no standalone dequant timing remain unchanged.
