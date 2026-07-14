# colibri-lite-rs Implementation Plan

## Purpose

Build a Rust-first, CPU-first inference runtime for low-memory
Mixture-of-Experts models. The runtime must prioritize numerical correctness,
predictable memory use, and on-demand expert loading before performance work.

This document is the implementation plan. Executable work items and their
status are tracked in [tasks.md](tasks.md).

## Delivery principles

1. Correctness before optimization.
2. Keep the core contracts independent of model files and operating-system I/O.
3. Make memory limits explicit and testable.
4. Add model-specific behavior outside `clr-core`.
5. Require repeatable tests and benchmarks for every optimization.
6. Keep `unsafe` forbidden until a measured requirement justifies a narrowly
   scoped exception.

## Workspace boundaries

| Crate | Responsibility | Must not own |
| --- | --- | --- |
| `clr-core` | Shared data types, validation, errors, and runtime contracts | File formats, Qwen-specific logic, CLI output |
| `clr-storage` | Model files, memory mapping, expert loading, and cache policy | Tensor arithmetic, model architecture |
| `clr-qwen3-moe` | Qwen3-MoE configuration and forward-pass implementation | CLI concerns, generic file caching |
| `clr-cli` | User-facing commands and runtime diagnostics | Inference algorithms and storage policy |

Dependencies flow in one direction:

```text
clr-cli ---------> clr-core
                      ^
                      |
clr-qwen3-moe --> clr-storage
       |              |
       +--------------+
```

No lower-level crate may depend on `clr-cli` or `clr-qwen3-moe`.

## Milestones

### M0 - Bootstrap and core contracts

#### M0.1 - Workspace bootstrap

Status: complete.

Deliverables:

- Cargo workspace with four crates.
- Shared package metadata and lint policy.
- Runtime identity exposed by `clr-core`.
- CLI reports `bootstrap ready`.
- Format, build, test, and Clippy checks pass on Windows x64 MSVC.

#### M0.2 - Core contracts

Status: next.

Goal: define stable, well-tested vocabulary for later tensor, storage, and
model work without implementing inference yet.

Planned module layout:

```text
crates/clr-core/src/
|-- config.rs
|-- dtype.rs
|-- error.rs
|-- lib.rs
|-- runtime.rs
`-- shape.rs
```

Contracts:

- `TensorShape`: owns tensor dimensions and provides rank, dimension access,
  scalar detection, and checked element-count calculation.
- `DataType`: initially represents only dense compute types required by the
  correctness path (`F32`, `F16`, and `BF16`). Quantized encodings are deferred
  until their storage and block-size semantics are known.
- `ModelConfig`: contains architecture-independent dimensions needed to
  validate a decoder-only MoE model. Construction must reject zero dimensions,
  inconsistent attention head counts, and invalid expert routing counts.
- `RuntimeError`: one crate-wide error type with structured variants for shape
  overflow and invalid configuration. It must implement `Display` and
  `std::error::Error` without adding an external dependency.
- Existing `RuntimeInfo` behavior moves to `runtime.rs` and remains publicly
  available through re-exports from `lib.rs`.

API rules:

- Fields that can violate invariants remain private.
- Constructors validate input and return `Result` when validation can fail.
- Arithmetic derived from dimensions uses checked operations.
- Public types implement `Debug`, equality traits where meaningful, and concise
  rustdoc with invariant notes.
- `clr-core` remains free of filesystem, serialization, and model-specific
  dependencies.

Test strategy:

- Unit tests cover normal shapes, scalar shapes, zero-sized dimensions, and
  element-count overflow.
- Configuration tests cover a valid MoE configuration and each validation
  failure independently.
- Error tests verify useful `Display` output.
- Existing runtime identity test remains intact.
- Workspace format, build, tests, and Clippy remain clean with warnings denied.

Definition of Done:

- All five modules exist with documented public APIs.
- Invalid states cannot be created through public constructors.
- Checked shape arithmetic cannot panic or wrap.
- `clr-core` has no new third-party dependency.
- `cargo fmt --all --check` passes.
- `cargo check --workspace` passes.
- `cargo test --workspace` passes.
- `cargo clippy --workspace --all-targets -- -D warnings` passes.
- `cargo run -p clr-cli` still reports `status: bootstrap ready`.

### M1 - Tiny Qwen3-MoE correctness

Goal: run a deterministic forward pass for a tiny Qwen3-MoE fixture and match a
Python/Transformers reference.

Planned scope:

- Dense tensor storage and checked views.
- Required CPU `f32` operations only.
- Qwen3 attention, normalization, rotary embeddings, router, and expert MLP.
- Tiny deterministic model fixture and Python reference output.
- Layer-by-layer comparison before full-logit comparison.

Exit condition: Rust logits match the reference within a documented tolerance.

### M2 - Storage and expert residency

Goal: load model tensors without requiring all experts to be resident in RAM.

Planned scope:

- Model manifest and tensor metadata validation.
- Read-only memory mapping behind a narrowly reviewed safety boundary.
- On-demand expert loading.
- RAM-budgeted LRU expert cache.
- Cache and I/O metrics.

Exit condition: repeated expert access respects the configured RAM budget and
passes deterministic eviction tests.

### M3 - Autoregressive generation

Goal: produce tokens from a prompt using the correctness-proven model path.

Planned scope:

- Token sampling with deterministic seeded tests.
- KV cache with explicit size accounting.
- Prefill and decode loops.
- Minimal CLI generation command.

Exit condition: a tiny model produces reproducible multi-token output without
unbounded memory growth.

### M4 - Full Qwen3-30B-A3B path

Goal: run the target model with storage-aware expert loading under a defined
memory budget.

Planned scope:

- Target model conversion/loading contract.
- Quantized expert representation selected from measured evidence.
- Full-model compatibility and memory tests.
- Baseline JSON report for speed, peak RAM, model size, hardware, and commit.

Exit condition: the target model generates tokens and emits a reproducible
baseline report while respecting the documented RAM budget.

## Deferred until after M4

- GPU backends.
- OpenAI-compatible HTTP server.
- Web UI.
- Continuous batching.
- Speculative decoding.
- Multimodal models.
- Distributed inference or RPC.
- Agent frameworks and tool calling.
- Broad model-family support.

## Review gates

Every milestone ends with these gates:

1. Contract review: public APIs and invariants are documented.
2. Correctness review: tests fail for known invalid inputs.
3. Quality review: format, check, test, and Clippy commands pass.
4. Scope review: deferred features have not entered the implementation.
5. Evidence review: benchmark or numerical claims include reproducible inputs.
