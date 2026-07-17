# ADR 0028: Frozen F32 Baseline Bundle and Fixture Hierarchy

- Status: Accepted for M4.3-01
- Date: 2026-07-16
- Milestone: M4.3
- Task: M4.3-01
- Model revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

M4.2 established that the current safe scalar Rust F32 runtime is numerically
credible for the pinned Qwen3-30B-A3B model. M4.3-01 therefore does not create
another unquantized implementation. It must freeze the proven path as the
reference for later quantized, lower-precision, alternative-kernel, cache,
I/O, and accelerator variants.

The existing six-position generation regression takes roughly 394 to 438
process seconds on the recorded host. Running it for every local change would
make focused development impractical. Conversely, a small operation test alone
cannot establish full-model semantic equivalence.

## Decision

The authoritative machine-readable baseline is
`models/qwen3-30b-a3b/m4.3-01-f32-baseline-manifest-v1.json`. It references
canonical model identity, M4.2 contracts and evidence, deterministic fixtures,
selected intermediates, final norm and LM-head evidence, generated IDs, KV
guarantees, and the unoptimized resource baseline by relative path and SHA-256.
It does not duplicate model or checkpoint payloads.

The precision hierarchy is:

1. selected F64 diagnostics;
2. Transformers F32 using pinned BF16-derived weights;
3. the ordered scalar Rust F32 runtime;
4. future lower-precision or optimized variants under separately reviewed
   comparison contracts.

F64 remains diagnostic-only. Proximity to an F64 value does not authorize an
arithmetic change.

The fixture hierarchy is:

- Tier A: the authoritative prompt `[9707, 11, 1879, 0]` generating
  `[1096, 374]`, including cached decode and all M4.2-04 contracts;
- Tier B: six complete-forward fixtures totaling 11 positions and covering a
  one-token low ID, English, Thai, code/newline, repeated text, and the
  end-of-text special token;
- Tier C: existing focused embedding, RMSNorm, attention, router, selected
  expert, final norm, LM-head, and KV-cache evidence.

Tier B stores compact evidence only: reference and Rust final-normalized-state
digests, selected values, fixed logits, top-20 values and IDs, argmax, margins,
finite counts, guard router IDs, KV allocation, bytes read, and expert-cache
counters. It does not retain full vocabulary or checkpoint tensors.

Tier B final-norm and compact-logit limits are fixture-specific. Following the
M4.2-04 safety model, each frozen budget is derived as:

```text
final norm = 3 * observed fixed-index error + 5e-7
logits     = 3 * observed checked-logit error + 2e-6
```

These values characterize only the named Tier B fixture and pinned host/tool
environment. They do not replace or widen the M4.2 tolerance registry.

The `single_low_token` fixture initially exceeded the largest prior M4.2
fixture-specific logit value. A same-input diagnostic measured only
`1.9073486328125e-5` maximum PyTorch-F32-versus-Rust LM-head difference over
all logits. The end-to-end compact maximum was `2.841949462890625e-4`, with
selected incoming-state effects up to `2.7370452880859375e-4`. The difference
is accumulated incoming drift, not a local LM-head arithmetic or layout defect.

Future variants use
`models/qwen3-30b-a3b/m4.3-01-comparison-schema-v1.json` and are classified as
exact-equivalent, numerically equivalent within contract, semantically
equivalent, quality-risk, or correctness failure. Performance alone cannot
accept a variant.

## Consequences

- The current ordered Rust F32 path is the authoritative implementation
  baseline; runtime arithmetic and artifacts do not change.
- Tier C supports fast local checks, Tier B expands representative prompt
  coverage, and Tier A remains the final end-to-end acceptance gate.
- A future variant must report correctness, numerical, memory, I/O,
  latency/throughput, artifact-size, and platform/dependency deltas.
- A changed dtype or accumulation order requires a versioned comparison
  contract; it cannot edit M4.2 or this F32 baseline in place.
- The current runtime remains correctness-valid but not performance-ready.

## Supporting Records

- `docs/reports/m4.3-01-f32-baseline.md`
- `docs/m4.3-f32-baseline-invariants.md`
- `docs/adr/0027-m4.2-correctness-contract-registry.md`
- `docs/m4.2-optimization-invariants.md`
- `models/qwen3-30b-a3b/m4.3-01-fixtures-v1.json`
- `models/qwen3-30b-a3b/m4.3-01-f64-diagnostics-v1.json`
