# ADR 0091: M6.3-R2.2 three-sentinel-layer proof

Status: accepted for implementation after contract freeze.

## Context

ADR 0090 closes R2.1 `GO`: the unchanged group32 artifact plus AVX2+FMA
projection kernel passed held-out quality and all 72 official Layer-0
performance samples. That evidence proves the Layer-0 vertical slice only.
It does not prove that the same path generalizes across depth or that an
all-layer implementation is justified.

R2.2 therefore expands proof only to three representative MoE layers: `0`,
`24`, and `47`. These are the existing early/middle/final guard layers, so they
exercise depth while reusing established semantic/router checkpoints. All other
layers remain canonical F32.

Frozen contract:
`models/qwen3-30b-a3b/m6.3-r2-2-multilayer-sentinel-proof-contract-v1.json`
SHA-256:
`3071f717b9ea12091323e596b9a3d40aa776b34f27f0cc66db63774e493af0c4`.
## Decision

Freeze the R2.2 sentinel set as Layers `0/24/47`. Reuse the exact group32
quantization grammar, exact R2.1 AVX2+FMA arithmetic, isolated FFI boundary,
and scalar fallback. No new kernel optimization is allowed inside R2.2.

R2.2 implementation may generalize the packed reader to an explicit layer-to-
artifact map and may create independently verified group32 artifacts for Layers
24 and 47. It may not build a 48-layer candidate artifact or change default
runtime behavior.

Quality runs before official performance. Both held-out fixtures must preserve
exact two-token generation, prompt top-20/argmax, Layers 0/24/47 router guards,
repeatability, and the existing F32 logit envelope. Native-vs-scalar drift must
remain <=`0.001` at each candidate-layer routed output and <=`0.002` at prompt
logits.

After quality PASS, run 108 fresh compute-only process samples: three sentinel
layers x two fixtures x six balanced native/scalar/F32 triplets x three paths.
Each layer/fixture must independently meet the frozen R2.1 compute thresholds.
No retry is allowed because a sample is unfavorable.
## Consequences

The R2.2 design review is complete and, once this contract is committed, R2.2
implementation is authorized within this exact three-layer boundary. A PASS may
authorize `R2.3` all-layer plan design only. It still cannot authorize R2.3
implementation, a 48-layer candidate rollout, M6.4, or a production throughput
claim.

The final R2.2 review must also publish an explicit impact model using measured
sentinel distributions and R2.1 load evidence. That model must state uncertainty
and may not claim a measured or guaranteed 48-layer speedup.
