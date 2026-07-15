# ADR 0021: Layer-1 Propagated Validation Budgets

- Status: Accepted provisionally for M4.2 Layer 1
- Date: 2026-07-15
- Milestone: M4.2
- Task: M4.2-02
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

Layer 1 must receive the genuine Rust Layer-0 block output. That input includes
only the 27 experts selected by the approved Layer-0 router, evaluated through
the normal `ExpertStore` and streaming numerical path. The resulting F32
Layer-1 input differs from the same-weight Transformers F32 control by at most
`1.9073486328125e-6`.

Layer-1 checkpoint guards cannot copy Layer-0 tolerances because accumulated
Layer-0 error enters every subsequent operation. They must combine measured
incoming error with a justified local-operation variance and the provisional
factor 3 established by ADR 0020.

## Decision

The selected Layer-0 expert down-projection outputs, routing-weighted combined
MoE output, and final block output retain the existing F32 correctness rule:

```text
absolute_error <= 1e-6 + 1e-5 * abs(reference)
```

This is not a new tolerance. It is the existing scalar F32 contract applied to
new expert checkpoints. All 32 selected-expert occurrences must pass before
Layer 1 executes.

Layer-1 checkpoint-specific absolute budgets are:

```text
incoming Layer-1 hidden-state error = 1.9073486328125e-6
isolated RMSNorm absolute guard     = 5.0e-7
Layer-0 local attention variance    = 1.2516975402832031e-6
Layer-0 same-input router variance  = 1.430511474609375e-5
safety factor                       = 3.0

input RMSNorm:
  3 * 1.9073486328125e-6 + 5.0e-7
  = 6.2220458984375e-6

attention output:
  3 * measured Layer-1 input-RMSNorm error + Layer-0 local attention variance
  = 3 * 2.905726432800293e-7 + 1.2516975402832031e-6
  = 2.123415470123291e-6

residual output:
  incoming Layer-1 error + attention-output budget
  = 4.030764102935791e-6

post-attention RMSNorm:
  3 * measured residual error + isolated RMSNorm guard
  = 3 * 1.9073486328125e-6 + 5.0e-7
  = 6.2220458984375e-6

router logits:
  3 * measured post-attention RMSNorm error + Layer-0 same-input router variance
  = 3 * 2.6226043701171875e-6 + 1.430511474609375e-5
  = 2.2172927856445312e-5

routing weights:
  1.0e-6 absolute
```

The routing-weight guard narrowly covers the approved Layer-0 observed maximum
of `5.960464477539063e-7` and the Layer-1 observed maximum of
`5.066394805908203e-7`. It does not change router normalization or selection.

Router IDs retain ADR 0019: F32 IDs are exact only when a positive boundary
margin exceeds twice the measured per-token F32-versus-Rust router-logit error.

## Consequences

- No Layer-1 reference input is injected into the primary Rust path.
- No Layer-1 expert executes, and Layer 2 remains unreachable.
- No global RMSNorm, router, artifact, or public runtime contract changes.
- These budgets are provisional evidence for this input and layer. Layers 24
  and 47 require new measured incoming errors and checkpoint-specific budgets.
- An isolated Layer-1 diagnostic is required only if the normal end-to-end path
  exceeds one of these budgets.

## Evidence

- `models/qwen3-30b-a3b/m4.2-02-rust-layer0-expert-evidence-v1.tsv`
- `models/qwen3-30b-a3b/m4.2-02-rust-layer1-checkpoint-evidence-v1.tsv`
- `models/qwen3-30b-a3b/m4.2-02-rust-layer1-router-evidence-v1.tsv`
