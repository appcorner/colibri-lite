# colibri-lite-rs Implementation Plan

## Purpose

Build a Rust-first, CPU-first, storage-aware inference runtime for low-memory
Mixture-of-Experts models.

The first supported architecture is Qwen3-MoE. The first full-size target is
Qwen3-30B-A3B on Windows x64. The runtime must prioritize numerical
correctness, predictable memory use, and on-demand expert residency before
performance optimization.

This document defines milestone scope and engineering gates. Executable work
items and status are tracked in [tasks.md](tasks.md).

## Product boundary

`colibri-lite-rs` is not a Rust rewrite of `llama.cpp`, `ik_llama.cpp`, or
Colibri. It is a focused runtime for MoE models whose total weights may exceed
the configured resident-memory budget.

North-star capability:

```text
Load a Qwen3-MoE model, keep dense tensors resident, load routed experts on
demand, enforce a byte-level RAM budget, and produce numerically validated
tokens on Windows x64.
```

## Reference roles

| Project | Role in this project |
| --- | --- |
| Hugging Face Transformers | Numerical correctness oracle |
| Colibri | Expert streaming and storage-hierarchy reference |
| ik_llama.cpp | Performance, quantization, and CPU-kernel baseline |
| katgpt-rs | Post-MVP algorithm research reference |
| colibri-lite-rs | Product codebase |

Reference projects may inform design and benchmarks, but code is not copied
without an explicit license and provenance review.

## Delivery principles

1. Correctness before optimization.
2. Prove one small vertical slice before broadening abstractions.
3. Keep core contracts independent of file formats and operating-system I/O.
4. Keep model-specific behavior outside `clr-core`.
5. Make memory limits explicit, byte-based, observable, and testable.
6. Require repeatable tests or benchmarks for every optimization claim.
7. Prefer safe Rust; isolate and document unavoidable `unsafe`.
8. Optimize only after a profiler or benchmark identifies a bottleneck.
9. Add one architecture first; do not create a premature general model zoo.
10. Do not let UI, server, agent, GPU, or speculative-decoding work enter MVP.

## Workspace boundaries

| Crate | Responsibility | Must not own |
| --- | --- | --- |
| `clr-core` | Shared value types, tensor contracts, validation, errors, runtime traits | Filesystem I/O, serialization formats, Qwen-specific logic, CLI presentation |
| `clr-storage` | Artifact access, tensor metadata, expert loading, residency policy, cache metrics | Tensor arithmetic, attention, router or model architecture |
| `clr-qwen3-moe` | Qwen3-MoE configuration mapping and forward implementation | CLI concerns, generic caching policy |
| `clr-cli` | User-facing commands, diagnostics, fixture execution | Inference algorithms, tensor kernels, cache policy |

Initial dependency direction:

```text
clr-cli ----------> clr-core
   |                    ^
   |                    |
   +--> clr-qwen3-moe --> clr-storage
             |               |
             +---------------+
```

`clr-storage` depends only on `clr-core`.
`clr-qwen3-moe` depends on `clr-core` and `clr-storage`.
`clr-cli` may compose all three but must not contain their implementation.

A new crate must not be added until an existing crate has a demonstrated
boundary problem.

## Non-goals through M4

- General replacement for llama.cpp.
- Broad model-family or GGUF compatibility.
- Production HTTP service.
- GPU acceleration.
- Continuous batching or concurrent multi-user scheduling.
- Tool calling, agent framework, latent reasoning, or speculative decoding.
- Multimodal or distributed inference.
- Training, fine-tuning, or LoRA.
- Performance parity with optimized C/C++ runtimes.

## Milestone branch policy

Each milestone is developed on its own branch. Create the branch from an
up-to-date `main`, keep commits focused, and merge only after the milestone's
acceptance criteria and required verification pass.

| Milestone | Branch |
| --- | --- |
| M0 | `milestone/m0-core-contracts` |
| M1 | `milestone/m1-tiny-qwen-correctness` |
| M2 | `milestone/m2-expert-residency` |
| M3 | `milestone/m3-generation` |
| M4 | `milestone/m4-full-qwen3` |

Only one milestone branch should be active for implementation at a time. A
later milestone branch must not begin while the current milestone has failed
acceptance criteria. Focused commits remain required within a milestone branch;
the branch is not a substitute for the commit policy in `AGENTS.md`.

## Milestones

### M0 - Bootstrap, contracts, and reference harness

#### M0.1 - Workspace bootstrap

Status: complete.

Deliverables:

- Cargo workspace with four crates.
- Shared package metadata and lint policy.
- Runtime identity exposed by `clr-core`.
- CLI reports `bootstrap ready`.
- Format, build, test, Clippy, and CLI smoke checks pass on Windows x64 MSVC.

#### M0.2 - Core value contracts

Status: complete.

Goal: define the minimum stable vocabulary required by the first tiny-model
vertical slice. Do not implement inference, I/O, serialization, or
quantization.

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

- `TensorShape`: owns dimensions and provides rank, dimension access, scalar
  detection, empty-tensor detection, and checked element-count calculation.
- `DataType`: metadata for `F32`, `F16`, and `BF16`. Only `F32` computation is
  required before later milestones.
- `ModelConfig`: minimum architecture-neutral decoder dimensions shared by the
  tiny fixture. Qwen-specific fields and semantics stay in `clr-qwen3-moe`.
- `RuntimeError`: structured errors for invalid shapes, arithmetic overflow,
  and invalid configuration, with useful `Display` output.
- Existing `RuntimeInfo` moves to `runtime.rs` and remains re-exported.

Contract rules:

- Fields that can violate invariants remain private.
- Constructors validate input and return `Result`.
- Derived dimension arithmetic uses checked operations.
- Error categories are matchable without parsing messages.
- Public APIs document invariants and scalar/zero-size behavior.
- No filesystem, serialization, model-specific, or third-party dependency is
  added to `clr-core`.

Important design guard:

`ModelConfig` must not become a dump of Qwen configuration fields. If a field
is not needed by at least the generic tensor/runtime boundary, it belongs in
`clr-qwen3-moe`.

Definition of Done:

- Modules and public contracts exist with tests.
- Invalid public states cannot be constructed.
- Dimension arithmetic cannot wrap or panic.
- `clr-core` has no new external dependency.
- All standard verification commands pass.
- CLI still reports `bootstrap ready`.

#### M0.3 - Deterministic fixture and oracle contract

Status: complete.

Goal: freeze the evidence used to decide whether the Rust implementation is
correct before writing the forward pass.

Scope:

- Pin Python, PyTorch, Transformers, and model-configuration versions.
- Create a deterministic tiny Qwen3-MoE configuration.
- Fix random seeds and export input IDs, weights, selected intermediate
  tensors, and expected logits.
- Define a versioned fixture manifest including tensor name, shape, dtype,
  byte order, offset or file path, and SHA-256.
- Define absolute/relative numerical tolerances per comparison point.
- Record commands that regenerate and verify the fixture.
- Keep generated model artifacts out of Git when large; keep a tiny fixture in
  Git only if license and size permit.

Exit condition:

A clean machine can regenerate the same fixture metadata and expected outputs,
or verify a checked-in fixture, without relying on undocumented state.

### M1 - Tiny Qwen3-MoE correctness

Status: complete.

Goal: execute a deterministic tiny Qwen3-MoE model in Rust and match the frozen
oracle layer by layer.

#### M1.1 - Dense tensor and kernel correctness

Scope:

- Owned dense `f32` tensor storage.
- Checked immutable and mutable views.
- Minimal operations required by the fixture only: indexing, reshape/view,
  elementwise add/multiply, matrix-vector or matrix-matrix multiply, softmax,
  SiLU, and reductions as required.
- Shape-validation and numerical unit tests.

Exit condition:

Every primitive matches a small independently calculated test case.

#### M1.2 - Single decoder/MoE block correctness

Scope:

- RMS normalization.
- Rotary embeddings.
- Causal grouped-query attention for the frozen fixture.
- Router logits, deterministic top-k selection, and routing weights.
- Expert gated MLP and weighted expert-output combination.
- Comparison of router selections and intermediate outputs with the oracle.

Exit condition:

One decoder block matches the recorded reference within its documented
tolerance, including exact selected expert IDs.

#### M1.3 - Full tiny decoder correctness

Scope:

- Embedding lookup.
- Multiple decoder blocks.
- Final normalization and language-model head.
- Final-logit comparison.
- Diagnostic output that identifies the first mismatching stage.

Exit condition:

Final logits match the reference tolerance and all selected expert IDs match
exactly.

### M2 - Storage and expert residency

Status: complete.

Goal: run the correctness-proven path while experts are loaded on demand under
a strict byte budget.

#### M2.1 - Artifact reader

Scope:

- Versioned manifest and tensor metadata.
- Validation of paths/offsets, lengths, shapes, dtypes, endianness, and hashes.
- A portable buffered/read-at implementation first.
- Clear ownership of loaded bytes and tensor views.

Exit condition:

Malformed artifacts fail before tensor execution, with deterministic errors.

#### M2.2 - Expert store and byte-budgeted cache

Scope:

- `ExpertId` and stable cache key contract.
- On-demand expert loading.
- Byte-budgeted LRU cache.
- Pin/lease semantics preventing eviction while an expert is in use.
- Strict handling of an expert larger than the configured budget.
- Hit, miss, load, eviction, resident-byte, and bytes-read metrics.

Exit condition:

Deterministic tests prove eviction order, no budget overrun, no use-after-
eviction, and unchanged numerical output.

#### M2.3 - Optional memory mapping

Status: deferred by evidence; not an M2 exit requirement.

Scope:

- Benchmark the portable reader before considering mapping.
- Add no mapping dependency or unsafe boundary without profiling evidence.
- Record the defer decision and measurable reconsideration criteria.
- If reconsidered, keep mapping behind the artifact-reader interface, isolate
  the boundary, and test Windows file/mapping lifetimes.

Exit condition:

Portable access has a reproducible baseline and mapping is either rejected or
retained from measured evidence. The approved M2 decision is to defer mapping:
portable access is not a demonstrated decode bottleneck and mapping would not
remove the current F32 decode/copy step.

### M3 - Autoregressive generation

Status: complete.

Goal: generate deterministic token IDs using the tiny correctness-proven path.

Scope:

- Greedy decoding first.
- Seeded temperature sampling second.
- Explicit KV-cache shape and byte accounting.
- Prefill and single-token decode loops.
- Context-length checks.
- Minimal CLI command accepting token IDs directly.
- Reproducible multi-token tests and bounded-memory tests.

Tokenizer integration is not required for M3; accepting token IDs keeps this
milestone focused on runtime correctness.

Exit condition:

The tiny model produces reproducible token-ID sequences with no unbounded
resident-memory growth.

### M4 - Full Qwen3-30B-A3B path

Goal: generate tokens with Qwen3-30B-A3B while enforcing a documented
resident-memory budget.

#### M4.1 - Full-model artifact conversion

Status: complete.

Scope:

- Pin one exact upstream model revision.
- Convert only the required Qwen3-MoE tensor set from Safetensors.
- Validate tensor names, shapes, config values, tokenizer assets, and hashes.
- Produce a versioned artifact that supports independent dense and expert
  access.
- Document conversion provenance and licensing.

#### M4.2 - Full-model correctness checkpoint

Status: in progress. M4.2-01 through M4.2-04 are complete.

Scope:

- Validate selected layers/tensors against Transformers before quantization.
- Run a short deterministic prompt or token sequence.
- Compare expert selections and selected intermediate outputs.
- Record peak RAM and bytes read even if performance is poor.

Exit condition:

The unoptimized storage-aware path is numerically credible and debuggable.

#### M4.3 - Evidence-driven quantization

Scope:

- Select a first expert quantization only after measuring memory and I/O.
- Keep sensitive dense/router tensors at a higher precision when evidence
  supports it.
- Validate degradation against a defined prompt/evaluation set.
- Treat ik_llama.cpp as a performance and quantization baseline, not a code
  target.

#### M4.4 - Reproducible baseline

Required JSON fields:

```json
{
  "runtime": "colibri-lite-rs",
  "runtime_commit": "",
  "model_id": "",
  "model_revision": "",
  "artifact_version": "",
  "quantization": "",
  "hardware": {},
  "resident_budget_bytes": 0,
  "peak_resident_bytes": 0,
  "bytes_read": 0,
  "cache_hit_rate": 0.0,
  "prompt_tokens_per_second": 0.0,
  "generation_tokens_per_second": 0.0
}
```

Exit condition:

The target model generates tokens, respects the configured budget, and emits a
reproducible report with known limitations.

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
- katgpt-rs-inspired reasoning policies.
- Production-grade tokenizer/chat-template abstraction.
- Cross-platform tuning beyond Windows x64 correctness.

## Review gates

Every milestone ends with these gates:

1. Contract review: public APIs and invariants are documented.
2. Correctness review: positive and known-invalid cases are tested.
3. Quality review: format, check, test, and Clippy pass.
4. Scope review: deferred work has not entered the implementation.
5. Evidence review: numerical or performance claims are reproducible.
6. Provenance review: fixtures, weights, and borrowed ideas have documented
   source revision and license.
7. Windows review: path, file-lifetime, and resource-release behavior is tested
   on Windows x64 MSVC.

## Standard verification commands

```powershell
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p clr-cli
```
