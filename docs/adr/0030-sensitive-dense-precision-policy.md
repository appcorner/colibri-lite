# ADR 0030: Keep Router and Sensitive Dense Tensors at Safe Precision

- Status: Accepted for M4.3-03 policy definition
- Date: 2026-07-17
- Milestone: M4.3
- Task: M4.3-03

## Context

M4.3-01 freezes ordered Rust F32 as the comparison baseline and M4.3-02
defines an expert-only INT8 candidate. This task measures non-expert weight
sensitivity without changing the runtime or creating a mixed artifact. The
diagnostic rounds selected canonical F32 weights to BF16 and reconstructs
selected matrices with symmetric per-output-channel INT8, while activations and
accumulations remain F32.

## Evidence

The evidence covers embedding and LM-head rows plus Q/K/V/O, normalization,
Q/K normalization, and router matrices at Layers 0, 1, 24, and 47. The
canonical artifact is BF16-derived, so BF16-rounded weights represented in F32
are bit-identical to the stored F32 weights. This proves only a storage-rounding
property; it does not validate BF16 activations or BF16 kernels.

INT8 router diagnostics changed the selected IDs at Layer 0 despite a positive
F32 boundary margin (`0.0468793`), and were therefore rejected for router use.
INT8 attention output projection local error reached `0.2976232`; other
attention projections also require propagated guard evidence. Normalization
vectors save little memory and participate in the frozen reduction-order
contract.

## Decision

The policy keeps router weights, input/post-attention RMSNorm weights, Q/K norm
weights, final RMSNorm, routing weights, residuals, activations, and all
accumulations at F32. Embedding, attention Q/K/V/O, and LM-head weights are
`candidate_bf16` only as future storage candidates with F32 compute; each needs
Tier C, Tier B, and Tier A evidence before acceptance. INT8 for these dense
groups is diagnostic-only, with INT8 router explicitly rejected.

The machine-readable policy is
`models/qwen3-30b-a3b/m4.3-03-mixed-precision-policy-v1.json`; the group
classification registry is
`models/qwen3-30b-a3b/m4.3-03-tensor-precision-registry-v1.json`.

## Consequences

- The canonical F32 artifact, ExpertStore, cache, arithmetic, router ordering,
  and M4.2 tolerance registry are unchanged.
- The largest possible BF16 storage reductions are embedding and LM-head,
  `622,329,856` bytes each, but no semantic claim is made until full-vocabulary
  and generation gates pass.
- Future mixed-precision work must report correctness, numerical, memory, I/O,
  latency, artifact, and platform deltas and must stop on the first semantic
  mismatch.

## Supporting Records

- `docs/reports/m4.3-03-precision-sensitivity.md`
- `models/qwen3-30b-a3b/m4.3-03-precision-sensitivity-evidence-v1.json`
- `models/qwen3-30b-a3b/m4.3-03-tensor-precision-registry-v1.json`
- `models/qwen3-30b-a3b/m4.3-03-mixed-precision-policy-v1.json`
- `docs/adr/0029-first-expert-int8-format.md`
