# colibri-lite-rs Tasks

Task states:

- `[x]` complete
- `[ ]` not started

## M0.1 - Workspace bootstrap

- [x] M0.1-01 Create the Cargo workspace.
- [x] M0.1-02 Create `clr-core`, `clr-storage`, `clr-qwen3-moe`, and `clr-cli`.
- [x] M0.1-03 Configure shared package metadata and workspace lint policy.
- [x] M0.1-04 Add crate dependencies in the intended direction.
- [x] M0.1-05 Add `RuntimeInfo` and `runtime_info()` to `clr-core`.
- [x] M0.1-06 Make `clr-cli` report the runtime identity and bootstrap status.
- [x] M0.1-07 Add repository README, ignore rules, and model placeholder.
- [x] M0.1-08 Pass format, check, tests, Clippy, and CLI smoke test.
- [x] M0.1-09 Commit the bootstrap baseline.

## M0.2 - Core contracts

### Error contract

- [ ] M0.2-01 Add `crates/clr-core/src/error.rs`.
- [ ] M0.2-02 Define structured `RuntimeError` variants for invalid shapes,
  element-count overflow, and invalid model configuration.
- [ ] M0.2-03 Implement `Display` and `std::error::Error` for `RuntimeError`.
- [ ] M0.2-04 Add tests for stable, useful error messages.

Acceptance:

- Callers can match error categories without parsing strings.
- No external error-handling dependency is added.

### Data type contract

- [ ] M0.2-05 Add `crates/clr-core/src/dtype.rs`.
- [ ] M0.2-06 Define `DataType::{F32, F16, BF16}`.
- [ ] M0.2-07 Add byte-width and display-name queries.
- [ ] M0.2-08 Test every variant's width and display value.

Acceptance:

- Dense correctness-path types have unambiguous byte widths.
- Quantized types are not introduced before block semantics are designed.

### Shape contract

- [ ] M0.2-09 Add `crates/clr-core/src/shape.rs`.
- [ ] M0.2-10 Implement `TensorShape` with private owned dimensions.
- [ ] M0.2-11 Add rank, dimensions, scalar, and empty-tensor queries.
- [ ] M0.2-12 Implement checked element-count calculation.
- [ ] M0.2-13 Test scalar, vector, matrix, zero-sized, and overflowing shapes.

Acceptance:

- Dimension multiplication never wraps or panics.
- Scalar and zero-sized behavior is explicitly tested and documented.

### Model configuration contract

- [ ] M0.2-14 Add `crates/clr-core/src/config.rs`.
- [ ] M0.2-15 Define architecture-independent decoder/MoE dimensions in
  `ModelConfig`.
- [ ] M0.2-16 Provide a validating constructor or builder.
- [ ] M0.2-17 Reject zero required dimensions.
- [ ] M0.2-18 Validate attention-head and KV-head relationships.
- [ ] M0.2-19 Validate expert count and experts-per-token relationships.
- [ ] M0.2-20 Test one valid configuration and each invalid invariant.

Acceptance:

- Public construction cannot produce a configuration that violates a listed
  invariant.
- Validation errors identify the invalid field or relationship.

### Module integration

- [ ] M0.2-21 Move runtime identity code to `runtime.rs` without changing its
  public behavior.
- [ ] M0.2-22 Declare the new modules from `lib.rs`.
- [ ] M0.2-23 Re-export the main contract types from the crate root.
- [ ] M0.2-24 Add rustdoc for every public type, constructor, and invariant.
- [ ] M0.2-25 Confirm `clr-storage` and `clr-qwen3-moe` still compile against
  `clr-core` without adding implementation code.

### M0.2 verification

- [ ] M0.2-26 Run `cargo fmt --all --check`.
- [ ] M0.2-27 Run `cargo check --workspace`.
- [ ] M0.2-28 Run `cargo test --workspace`.
- [ ] M0.2-29 Run `cargo clippy --workspace --all-targets -- -D warnings`.
- [ ] M0.2-30 Run `cargo run -p clr-cli` and confirm `bootstrap ready`.
- [ ] M0.2-31 Review the diff for inference, I/O, serialization, or
  quantization work that belongs to a later milestone.
- [ ] M0.2-32 Commit M0.2 with a focused commit message.

## M1 - Tiny Qwen3-MoE correctness

- [ ] M1-01 Freeze the tiny model configuration and deterministic fixture.
- [ ] M1-02 Create the Python/Transformers reference script.
- [ ] M1-03 Define dense tensor storage and safe views.
- [ ] M1-04 Implement only the required `f32` tensor operations.
- [ ] M1-05 Implement RMS normalization and rotary embeddings.
- [ ] M1-06 Implement causal attention and grouped-query attention behavior.
- [ ] M1-07 Implement router scoring and top-k expert selection.
- [ ] M1-08 Implement the expert MLP and routed output combination.
- [ ] M1-09 Compare intermediate layer outputs with the Python reference.
- [ ] M1-10 Compare final logits within the documented tolerance.
- [ ] M1-11 Record fixture provenance, tolerance, and reproduction commands.

## M2 - Storage and expert residency

- [ ] M2-01 Define the model manifest and tensor metadata contract.
- [ ] M2-02 Validate offsets, lengths, shapes, and data types before reads.
- [ ] M2-03 Add read-only memory mapping behind a reviewed safety boundary.
- [ ] M2-04 Implement on-demand expert loading.
- [ ] M2-05 Implement a byte-budgeted LRU expert cache.
- [ ] M2-06 Add cache hit, miss, load, eviction, and resident-byte metrics.
- [ ] M2-07 Test deterministic eviction and strict RAM-budget behavior.

## M3 - Autoregressive generation

- [ ] M3-01 Define token sampling inputs and seeded RNG behavior.
- [ ] M3-02 Implement greedy and temperature sampling.
- [ ] M3-03 Implement KV cache allocation and byte accounting.
- [ ] M3-04 Implement prefill and single-token decode paths.
- [ ] M3-05 Add a minimal generation command to `clr-cli`.
- [ ] M3-06 Test reproducible multi-token generation and bounded memory use.

## M4 - Full Qwen3-30B-A3B path

- [ ] M4-01 Define the supported model artifact and conversion procedure.
- [ ] M4-02 Validate full-model tensor names, shapes, and configuration.
- [ ] M4-03 Select quantization from measured correctness and memory evidence.
- [ ] M4-04 Run the full model with on-demand expert loading.
- [ ] M4-05 Emit the common baseline JSON report.
- [ ] M4-06 Record tokens/second, peak RAM, model size, hardware, and commit.
- [ ] M4-07 Document the supported configuration and known limitations.

## Standard verification commands

Run these commands before closing any milestone:

```powershell
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p clr-cli
```
