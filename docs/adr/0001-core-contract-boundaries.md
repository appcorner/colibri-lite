# ADR 0001: Core Contract Boundaries

- Status: Accepted
- Date: 2026-07-14
- Milestone: M0.2

## Context

M1 needs a small shared vocabulary for dense tensor metadata, checked shape
arithmetic, decoder dimensions, and errors. These contracts will be consumed by
storage, Qwen3-MoE, and CLI crates, so allowing model-format or
architecture-specific fields into `clr-core` would make later boundaries hard
to reverse.

M0.2 must not implement tensor storage, arithmetic, file I/O, serialization,
quantization, or Qwen-specific routing behavior.

## Decision

`clr-core` exposes five dependency-free contract areas through crate-root
re-exports:

- `RuntimeError` provides structured, matchable error categories for invalid
  shape/configuration values, size overflow, and invalid index access.
- `DataType` describes the dense metadata types `F32`, `F16`, and `BF16`.
  Only `F32` computation is in scope through the initial M1 correctness path.
- `TensorShape` owns dimensions. Scalar `[]` contains one element; a shape with
  any zero dimension contains zero elements. Element and byte counts use
  checked arithmetic.
- `ModelConfigSpec` is explicitly unvalidated input. `ModelConfig` can only be
  created by validation and contains architecture-neutral decoder dimensions.
- `RuntimeInfo` remains behavior-compatible while moving into `runtime.rs`.

The modules remain private implementation details. Primary types are re-exported
from the crate root so internal file organization can change without forcing
downstream import changes.

## Invariants

- `ModelConfig` required dimensions are non-zero.
- `hidden_size` is divisible by `attention_head_count`.
- `attention_head_count` is divisible by `key_value_head_count`.
- Expert counts, top-k routing, rotary settings, and Qwen-specific semantics do
  not enter the generic configuration.
- Shape-derived multiplication never wraps or panics.
- `clr-core` adds no external dependency.
- No public contract requires `unsafe`.

## Consequences

Positive:

- Later crates receive small validated values with predictable failure modes.
- Qwen-specific configuration can evolve independently in `clr-qwen3-moe`.
- Checked byte counts can be reused by storage budget validation.
- Crate-root re-exports reduce coupling to module layout.

Tradeoffs:

- `ModelConfigSpec` may represent invalid input until passed to
  `ModelConfig::new`.
- `F16` and `BF16` are metadata-only until explicit compute support is planned.
- Quantized block formats require a later contract because byte width alone is
  insufficient to describe them.

## Evidence

M0.2 unit tests cover error categories/messages, every dense data-type variant,
scalar and zero-sized shapes, invalid dimension access, arithmetic overflow,
valid model configuration, each zero dimension, attention relationships, and
the reviewed architecture-neutral field set.
