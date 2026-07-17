# ADR 0031: Expert INT8 Degradation Decision

- Status: Accepted as M4.3-04 quality-risk record
- Date: 2026-07-17
- Milestone: M4.3
- Task: M4.3-04

## Context

M4.3-02 selected symmetric INT8 per-output-channel expert weights with F32
scales and accumulation. M4.3-04 tested that format as a diagnostic simulation
against the frozen Rust F32 path while keeping every non-expert tensor and
operation at F32.

## Evidence

All eight Tier-C expert cases passed the representative M4.3-02 gates. Full
Tier-B propagation then showed substantial accumulated drift: the earliest
material increase was Layer 1 (`0.1724930` final-block maximum), and the largest
observed Layer-47 block difference was `152.1300659`. The Thai Tier-B fixture
changed argmax from `7360` to `16222` under a numerically ambiguous margin.

The Tier-A cached sequence still generated `[1096, 374]`, but several steps had
ambiguous vocabulary margins and several guard IDs were no longer exact-safe.
No safe-margin true mismatch was observed, and no F32 contract changed.

## Decision

Classify the candidate as `quality_risk`. It is not accepted for runtime
prototype implementation. The result is not a permanent format rejection;
future work may evaluate a lower-error group-wise candidate or a separately
reviewed quantization scheme. Router, RMSNorm, Q/K norm, routing weights,
residuals, activations, and accumulations remain F32.

The complete machine-readable evidence is
`models/qwen3-30b-a3b/m4.3-04-degradation-evidence-v1.json`, with provisional
decision gates in
`models/qwen3-30b-a3b/m4.3-04-provisional-degradation-gates-v1.json`.

## Consequences

- No quantized runtime, ExpertStore, cache, or artifact implementation begins
  from this result.
- The selected expert format remains a storage/format candidate, not an
  accepted quality-preserving model format.
- Any future prototype must rerun Tier C, all Tier B fixtures, and Tier A,
  preserve the F32 registry, and report semantic and numerical deltas.

## Supporting Records

- `docs/reports/m4.3-04-expert-int8-degradation.md`
- `docs/adr/0029-first-expert-int8-format.md`
- `models/qwen3-30b-a3b/m4.3-02-provisional-correctness-gates-v1.json`
- `models/qwen3-30b-a3b/m4.3-03-tensor-precision-registry-v1.json`
