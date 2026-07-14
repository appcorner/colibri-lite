# ADR 0002: Dense F32 Tensor Contract

- Status: Accepted
- Date: 2026-07-14
- Milestone: M1.1

## Context

The frozen tiny Qwen3-MoE fixture requires a small set of dense tensor
operations before model behavior can be compared with the Python oracle. This
first path must make shape failures explicit and remain readable enough to
audit numerical differences stage by stage.

Introducing strides, broadcasting, generic data types, SIMD, FFI, or fused
kernels at this point would expand the correctness surface before the scalar
reference path is proven.

## Decision

`clr-core` owns three contiguous row-major tensor contracts:

- `Tensor` owns a `TensorShape` and `Vec<f32>`;
- `TensorView` borrows checked immutable storage;
- `TensorViewMut` borrows checked mutable storage.

Construction requires shape-derived element count to equal storage length.
Multidimensional indexing checks rank and every coordinate before calculating a
row-major offset. Scalar and zero-sized behavior follows ADR 0001.

The public `ops` module contains only operations needed by the frozen fixture:

- elementwise add and multiply;
- sum and mean reductions;
- matrix-vector and matrix-matrix multiplication;
- final-dimension softmax;
- `SiLU` activation.

All implementations use straightforward scalar `f32` loops. Softmax subtracts
the row maximum before exponentiation. Softmax and `SiLU` reject NaN/infinite
inputs with an error identifying the flat element index.

## Invariants

- Storage is contiguous row-major with no implicit strides or broadcasting.
- Tensor construction cannot expose mismatched shape/storage length.
- Checked indexing returns an error rather than panicking for user coordinates.
- Binary elementwise operations require identical shapes.
- Matrix operations require explicit compatible ranks and inner dimensions.
- No kernel uses `unsafe`, SIMD, FFI, parallelism, or an external dependency.
- F16/BF16 remain metadata-only; computation is F32-only.

## Consequences

Positive:

- M1 model code has a small auditable numerical base.
- Failure categories preserve shape/rank/non-finite diagnostics.
- Views avoid copies while retaining construction checks and Rust lifetimes.
- Hand-calculated tests provide evidence independent of the Python oracle.

Tradeoffs:

- All tensors are owned or viewed as contiguous F32 data.
- No broadcasting or batched matrix multiplication exists yet.
- Scalar loops are intentionally slower than optimized kernels.
- The mean divisor is represented as F32, matching the correctness path's
  accumulation precision.

## Evidence

- Five tensor/view tests cover owned storage, immutable/mutable views,
  row-major reads/writes, scalar/empty tensors, wrong rank, and bounds errors.
- Ten operation tests cover independent expected values plus shape, rank,
  empty-reduction, and non-finite failures.
- Existing error regression tests cover every new structured diagnostic.
- Workspace Clippy passes with warnings denied.
