# colibri-lite-rs Tasks

Task states:

- `[x]` complete
- `[ ]` not started
- `[~]` in progress
- `[!]` blocked

Rules:

- Close tasks in order unless a task explicitly says it may run in parallel.
- Every completed milestone must pass the standard verification commands.
- New ideas that do not unblock the current milestone go to `docs/backlog.md`.
- Do not start optimization work before a correctness test exposes the same
  execution path.
- Work on the branch assigned to the current milestone before changing code.
- Record each meaningful work session in `docs/work-log.md`.

Milestone branches:

| Milestone | Branch |
| --- | --- |
| M0 | `milestone/m0-core-contracts` |
| M1 | `milestone/m1-tiny-qwen-correctness` |
| M2 | `milestone/m2-expert-residency` |
| M3 | `milestone/m3-generation` |
| M4 | `milestone/m4-full-qwen3` |

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

## M0.2 - Core value contracts

### Error contract

- [x] M0.2-01 Add `crates/clr-core/src/error.rs`.
- [x] M0.2-02 Define structured `RuntimeError` variants for invalid shapes,
  checked-arithmetic overflow, invalid configuration, and out-of-range access.
- [x] M0.2-03 Implement `Display` and `std::error::Error` without an external
  error crate.
- [x] M0.2-04 Test error categories and useful stable message fragments.

Acceptance:

- Callers match categories without parsing messages.
- No external dependency is added to `clr-core`.

### Data type contract

- [x] M0.2-05 Add `crates/clr-core/src/dtype.rs`.
- [x] M0.2-06 Define metadata variants `DataType::{F32, F16, BF16}`.
- [x] M0.2-07 Add byte-width, display-name, and floating-point queries.
- [x] M0.2-08 Document that M0/M1 computation supports `F32` only.
- [x] M0.2-09 Test every variant.

Acceptance:

- Metadata widths are unambiguous.
- Quantized block formats are not introduced.

### Shape contract

- [x] M0.2-10 Add `crates/clr-core/src/shape.rs`.
- [x] M0.2-11 Implement `TensorShape` with private owned dimensions.
- [x] M0.2-12 Add rank, dimensions, dimension access, scalar, and empty queries.
- [x] M0.2-13 Implement checked element-count and checked byte-count helpers.
- [x] M0.2-14 Freeze and document scalar `[]` and zero-sized `[2, 0, 3]`
  semantics.
- [x] M0.2-15 Test scalar, vector, matrix, zero-sized, invalid access, and
  overflow cases.

Acceptance:

- Multiplication never wraps or panics.
- Scalar and zero-sized behavior is explicit.

### Model configuration contract

- [x] M0.2-16 Add `crates/clr-core/src/config.rs`.
- [x] M0.2-17 List the truly architecture-neutral fields required by M1.
- [x] M0.2-18 Define a minimal validated `ModelConfig`; keep Qwen-only fields
  out of `clr-core`.
- [x] M0.2-19 Reject zero required dimensions.
- [x] M0.2-20 Validate generic hidden/head/KV-head relationships only when they
  are genuinely architecture-neutral.
- [x] M0.2-21 Test one valid configuration and every invariant independently.
- [x] M0.2-22 Add a review test/checklist ensuring no Qwen field has leaked into
  the generic config.

Acceptance:

- Public construction cannot create a listed invalid state.
- Validation identifies the invalid field or relationship.
- `ModelConfig` does not become a mirror of Hugging Face Qwen config.

### Module integration

- [x] M0.2-23 Move runtime identity code to `runtime.rs` without changing its
  public behavior.
- [x] M0.2-24 Declare modules from `lib.rs`.
- [x] M0.2-25 Re-export primary contract types from the crate root.
- [x] M0.2-26 Add rustdoc for every public type, constructor, and invariant.
- [x] M0.2-27 Confirm dependent crates compile without adding inference,
  serialization, storage, or model implementation.

### M0.2 verification

- [x] M0.2-28 Run the standard verification commands.
- [x] M0.2-29 Review the diff for out-of-scope I/O, serialization,
  quantization, tensor math, and Qwen-specific behavior.
- [x] M0.2-30 Commit with `feat(core): add validated runtime value contracts`.

## M0.3 - Deterministic fixture and oracle contract

### Environment and provenance

- [x] M0.3-01 Add `python/reference/requirements.lock` or an equivalent pinned
  environment file.
- [x] M0.3-02 Record Python, PyTorch, Transformers, and Safetensors versions.
- [x] M0.3-03 Pin the exact Qwen3-MoE architecture/config reference revision.
- [x] M0.3-04 Add fixture license and provenance notes.

### Tiny model definition

- [x] M0.3-05 Define a tiny Qwen3-MoE config with small vocabulary, hidden size,
  layer count, expert count, and top-k.
- [x] M0.3-06 Fix all random seeds and deterministic settings.
- [x] M0.3-07 Freeze a short token-ID input sequence.
- [x] M0.3-08 Export model configuration and deterministic weights.

### Oracle outputs

- [x] M0.3-09 Record router logits and selected expert IDs.
- [x] M0.3-10 Record outputs after normalization, attention, MoE, one full
  decoder block, and final logits.
- [x] M0.3-11 Define per-stage absolute and relative tolerances.
- [x] M0.3-12 Add SHA-256 values for all fixture files.
- [x] M0.3-13 Add commands to regenerate and verify the fixture.
- [x] M0.3-14 Verify regeneration or verification on a clean environment.

Acceptance:

- Expert IDs are deterministic and exact.
- Numerical checkpoints are versioned and reproducible.
- Rust implementation work does not begin until this contract is frozen.

## M1.1 - Dense tensor and kernel correctness

### Tensor ownership and views

- [x] M1.1-01 Define owned dense `f32` tensor storage.
- [x] M1.1-02 Define checked immutable and mutable views.
- [x] M1.1-03 Enforce shape/length equality at construction.
- [x] M1.1-04 Add checked indexing and contiguous-layout documentation.

### Minimal operations

- [x] M1.1-05 Implement only operations required by the fixture.
- [x] M1.1-06 Implement elementwise add/multiply and required reductions.
- [x] M1.1-07 Implement matrix-vector or matrix-matrix multiplication.
- [x] M1.1-08 Implement softmax and SiLU.
- [x] M1.1-09 Add independent hand-calculated unit tests for every primitive.
- [x] M1.1-10 Add shape-error and non-finite-input diagnostic tests where
  applicable.

Acceptance:

- No operation exists only because it may be useful later.
- All operations pass independent small-value tests.

## M1.2 - Single Qwen3-MoE block correctness

- [x] M1.2-01 Define Qwen3-specific config mapping in `clr-qwen3-moe`.
- [x] M1.2-02 Implement RMS normalization.
- [x] M1.2-03 Implement rotary embeddings.
- [x] M1.2-04 Implement causal grouped-query attention for the fixture.
- [x] M1.2-05 Implement router logits and deterministic top-k selection.
- [x] M1.2-06 Define tie-breaking behavior and test it explicitly.
- [x] M1.2-07 Implement routing-weight normalization.
- [x] M1.2-08 Implement gated expert MLP.
- [x] M1.2-09 Implement weighted expert-output combination.
- [x] M1.2-10 Compare expert IDs exactly with the oracle.
- [x] M1.2-11 Compare every recorded intermediate output within tolerance.
- [x] M1.2-12 Add diagnostics naming the first mismatching stage.

Acceptance:

- One decoder/MoE block matches the frozen oracle.
- Router tie behavior is deterministic.

## M1.3 - Full tiny decoder correctness

- [x] M1.3-01 Implement embedding lookup.
- [x] M1.3-02 Compose multiple decoder blocks.
- [x] M1.3-03 Implement final normalization and LM head.
- [x] M1.3-04 Compare final logits with the oracle.
- [x] M1.3-05 Test repeated runs for identical output.
- [x] M1.3-06 Record reproduction commands and first correctness report.

Acceptance:

- Final logits satisfy documented tolerance.
- All expert selections match exactly.
- Standard verification commands pass.

## M2.1 - Artifact reader

- [x] M2.1-01 Define a versioned artifact manifest.
- [x] M2.1-02 Define tensor metadata: name, shape, dtype, byte order, location,
  length, and hash.
- [x] M2.1-03 Validate duplicate names, paths/offsets, lengths, shape-derived
  byte counts, and hashes.
- [x] M2.1-04 Implement portable read/read-at access before memory mapping.
- [x] M2.1-05 Reject malformed artifacts before tensor execution.
- [x] M2.1-06 Add corruption, truncation, and wrong-endianness tests.

## M2.2 - Expert store and cache

- [x] M2.2-01 Define `ExpertId` and a stable cache key.
- [x] M2.2-02 Implement on-demand expert loading through the artifact reader.
- [x] M2.2-03 Implement a byte-budgeted LRU cache.
- [x] M2.2-04 Define lease/pin behavior while an expert is in use.
- [x] M2.2-05 Define behavior when one expert exceeds the entire budget.
- [x] M2.2-06 Add hit, miss, load, eviction, resident-byte, peak-byte, and
  bytes-read metrics.
- [x] M2.2-07 Test deterministic eviction order.
- [x] M2.2-08 Test strict budget enforcement and no use-after-eviction.
- [x] M2.2-09 Run the tiny model through resident and on-demand paths and prove
  identical output.

## M2.3 - Optional memory mapping

- [x] M2.3-01 Benchmark portable access before adding mapping.
- [x] M2.3-02 Review mapping evidence and approve the portable backend as the
  M2 production path.
- [x] M2.3-03 Confirm no mapping dependency or `unsafe` boundary was added.
- [x] M2.3-04 Record the Windows portable baseline and current copy behavior.
- [x] M2.3-05 Document measurable criteria required to reconsider mapping.
- [x] M2.3-06 Add deferred mapping work to `docs/backlog.md`.

## M3 - Autoregressive generation

- [x] M3-01 Implement greedy token-ID decoding.
- [x] M3-02 Define seeded RNG behavior.
- [x] M3-03 Implement temperature sampling after greedy decoding passes.
- [x] M3-04 Define KV-cache layout, context limit, and byte accounting.
- [x] M3-05 Implement prefill.
- [x] M3-06 Implement single-token decode.
- [x] M3-07 Add a CLI command accepting token IDs directly.
- [x] M3-08 Test reproducible token sequences.
- [x] M3-09 Test bounded memory over repeated decode steps.
- [x] M3-10 Record a tiny-generation correctness report.

## M4.1 - Full-model artifact conversion

- [x] M4.1-01 Pin exact Qwen3-30B-A3B model ID and revision.
- [x] M4.1-02 Document upstream license and artifact provenance.
- [x] M4.1-03 Map required Hugging Face configuration fields.
- [x] M4.1-04 Map and validate required tensor names and shapes.
- [x] M4.1-05 Convert dense tensors for resident access.
- [x] M4.1-06 Convert experts for independent on-demand access.
- [x] M4.1-07 Include tokenizer assets required for the first full-model test.
- [x] M4.1-08 Generate hashes and a reproducible conversion manifest.

## M4.2 - Full-model correctness checkpoint

- [ ] M4.2-01 Validate selected tensor values against Safetensors.
- [ ] M4.2-02 Validate selected layer router IDs against Transformers.
- [ ] M4.2-03 Validate selected intermediate outputs.
- [ ] M4.2-04 Run a short deterministic token sequence.
- [ ] M4.2-05 Record peak resident bytes, bytes read, and cache metrics.
- [ ] M4.2-06 Document failures or tolerance differences before optimization.

## M4.3 - Evidence-driven quantization

- [ ] M4.3-01 Establish an unquantized or higher-precision correctness baseline.
- [ ] M4.3-02 Define the first candidate expert quantization format.
- [ ] M4.3-03 Keep router and sensitive dense tensors at measured safe precision.
- [ ] M4.3-04 Compare output degradation against the baseline.
- [ ] M4.3-05 Compare memory/I/O and speed against ik_llama.cpp where formats
  and hardware permit.
- [ ] M4.3-06 Select or reject the candidate based on recorded evidence.

## M4.4 - Reproducible full-model baseline

- [ ] M4.4-01 Emit versioned baseline JSON.
- [ ] M4.4-02 Record runtime/model commits and artifact version.
- [ ] M4.4-03 Record hardware and Windows version.
- [ ] M4.4-04 Record resident budget, peak resident bytes, total bytes read, and
  cache hit rate.
- [ ] M4.4-05 Record prompt and generation throughput.
- [ ] M4.4-06 Document supported configuration and known limitations.
- [ ] M4.4-07 Repeat the run and verify the report is reproducible.

## Standard verification commands

Run before closing every milestone:

```powershell
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p clr-cli
```
