# ADR 0003: Qwen3-MoE Block Correctness Contract

- Status: Accepted
- Date: 2026-07-14
- Milestone: M1.2

## Context

M1.2 must reproduce one frozen Transformers 5.12.1 Qwen3-MoE decoder block
stage by stage. The architecture-specific values must remain outside
`clr-core`, and the implementation must use model configuration rather than
assume constants from another Qwen revision.

During implementation, a draft test assumed `rope_theta = 1000000.0`. The
frozen fixture's `config.json` records `rope_theta = 10000.0`, and the pinned
Transformers source uses that value directly when deriving inverse rotary
frequencies. Work stopped until the conflict was reviewed.

## Decision

`Qwen3MoeConfig` layers validated architecture-specific values over the generic
`ModelConfig`. `rope_theta`, RMS epsilon, expert counts, top-k, expert hidden
width, and routing normalization are supplied through configuration.

The fixture generator emits `rust-config.rs` from the frozen Transformers
configuration. Rust oracle tests use that generated value and assert that the
current fixture maps to `rope_theta = 10000.0`. Production RoPE contains no
hard-coded theta value and reads `Qwen3MoeConfig::rope_theta()` for every
frequency calculation.

One correctness-first sparse block implements:

- input and per-head Q/K RMS normalization;
- default rotate-half RoPE;
- causal grouped-query attention;
- full-expert softmax followed by deterministic top-k selection;
- lower expert ID as the explicit equal-score tie breaker;
- optional selected-probability renormalization;
- packed gate/up projections, `SiLU`, down projection, routing weights, and
  accumulation in ascending expert-ID order;
- attention and MoE residual connections.

The M1 fixture path accepts one batch represented as `[sequence, hidden]`. It
does not implement padding, KV cache, arbitrary position offsets, sliding-window
attention, or optimized kernels.

## Oracle extension

The frozen fixture was extended compatibly with:

- generated Rust constants derived from `config.json`;
- post-RoPE query tensors;
- post-RoPE key tensors.

The fixture hash manifest and tensor inventory include these additions, and
offline byte-for-byte regeneration still passes. Existing checkpoint names and
semantics remain unchanged.

## Invariants

- M1 compute type is F32.
- Head dimension is even and is derived from hidden/query-head dimensions.
- `rope_theta` and RMS epsilon are finite and positive.
- Top-k is non-zero and does not exceed expert count.
- Weight shapes are validated before block construction.
- Attention is causal and KV heads are repeated by the configured group count.
- Equal router scores select lower expert IDs first.
- Selected experts are compared exactly; floating stages use frozen tolerances.
- No external Rust dependency, `unsafe`, SIMD, FFI, or storage reader is added.

## Evidence

- Frozen config mapping test confirms exact `rope_theta = 10000.0`.
- A two-theta test (`10000.0` and `1000000.0`) confirms RoPE output changes
  with configuration.
- Query/key RoPE tensors match the frozen oracle.
- Causal grouped-query attention output matches the frozen oracle.
- Router logits/weights match within tolerance and expert IDs match exactly.
- Equal-score tie breaking and optional top-k renormalization have explicit
  tests.
- Routed expert and full block outputs match every recorded stage.
- A diagnostic regression test reports the first mismatching stage.
- All standard verification commands pass with 42 Rust tests and zero Clippy
  warnings.
