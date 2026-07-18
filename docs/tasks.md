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

- [x] M4.2-01 Validate selected tensor values against Safetensors.
- [x] M4.2-02 Validate selected layer router IDs against Transformers.
- [x] M4.2-03 Validate selected intermediate outputs.
- [x] M4.2-04 Run a short deterministic token sequence.
- [x] M4.2-05 Record peak resident bytes, bytes read, and cache metrics.
- [x] M4.2-06 Document failures or tolerance differences before optimization.

## M4.3 - Evidence-driven quantization

- [x] M4.3-01 Establish an unquantized or higher-precision correctness baseline.
- [x] M4.3-02 Define the first candidate expert quantization format.
- [x] M4.3-03 Keep router and sensitive dense tensors at measured safe precision.
- [x] M4.3-04 Compare output degradation against the baseline.
- [x] M4.3-05 Compare memory/I/O and speed against ik_llama.cpp where formats
  and hardware permit.
- [x] M4.3-06 Select or reject the candidate based on recorded evidence.

M4.3 is closed with the F32 baseline accepted, full-model expert INT8 rejected
for production, and the first optimization pivot recorded in
`docs/m4.3-next-phase-memory-hierarchy-roadmap.md`.

## M4.4 - Reproducible full-model baseline

- [x] M4.4-01 Emit versioned baseline JSON.
- [x] M4.4-02 Record runtime/model commits and artifact version.
- [x] M4.4-03 Record hardware and Windows version.
- [x] M4.4-04 Record resident budget, peak resident bytes, total bytes read, and
  cache hit rate.
- [x] M4.4-05 Record prompt and generation throughput.
- [x] M4.4-06 Document supported configuration and known limitations.
- [x] M4.4-07 Repeat the run and verify the report is reproducible.

M4 is complete. The release provenance and closure record are in
`models/qwen3-30b-a3b/m4-release-provenance-v1.json` and
`docs/reports/m4-release-closure.md`. No M5 implementation had started at
the release boundary.

## M5 - Memory hierarchy and performance recovery

- [x] M5.1-00 Capture authoritative ordered expert trace.
- [x] M5.1-01 Trace-driven memory hierarchy simulation.
- [x] M5.1-02 Implement the reviewed configurable expert-cache prototype.
- [x] M5.1-03 Validate the configurable expert cache on the canonical full model.
- [x] M5.2-01 Capture broader representative expert traces.
- [x] M5.2-02 Simulate cache policies and RAM budgets across the representative trace corpus.
- [x] M5.2-03 Validate 8 GiB versus 16 GiB global LRU across representative full-model workloads.
- [x] M5.3-01 Study mmap and coalesced expert access.
- [x] M5.3-02 Implement reusable aligned read-buffer prototype.
- [x] M5.3-03 Compute profiling.
- [x] M5.3-04 Isolated read-only mmap expert-access prototype (complete for review; rejected for production adoption).

M5.1-00 is complete as a deterministic measurement supplement. The ordered
trace and validator are recorded in
`models/qwen3-30b-a3b/m5.1-00-ordered-expert-trace-v1.json` and
`scripts/validate_m5_1_00_trace.py`. No cache simulation or runtime prototype
has started.

M5.1-01 is complete as a deterministic, simulation-only study. Results are
recorded in `models/qwen3-30b-a3b/m5.1-01-memory-hierarchy-results-v1.json`
and the first prototype decision is recorded in ADR 0035. No Rust runtime
behavior, cache capacity, artifact, or numerical execution changed.

M5.1-02 is accepted with limitations. The configurable payload-byte LRU cache
and expanded accounting are implemented in `clr-storage`; ordered trace replay
matches the M5.1-01 counters at the reviewed operating points. ADR 0036 records
the decision.

M5.1-03 validated the same primitive against the canonical artifact at the
baseline, exact 8 GiB, and exact 16 GiB payload budgets. Correctness invariants,
generated IDs, bounded residency, and exact-budget trace counters passed.
Results are recorded in
`models/qwen3-30b-a3b/m5.1-03-full-model-cache-results-v1.json` and ADR 0037.
The classification remains `accepted_with_limitations` because the fixture is
short, filesystem cache state was uncontrolled, process working-set sampling
and full-vocabulary logits were unavailable, and timing uses one sample per
mode. No resident-dense or other optimization prototype has started.

M5.2-01 is complete for review as an evidence-only corpus capture. The
representative corpus contains eight deterministic workload traces, including
the frozen Tier-A control, English and Thai prompts, source code, repeated
text, formatting-heavy input, longer context, and longer decode. Results are
recorded in
`models/qwen3-30b-a3b/m5.2-01-trace-corpus-manifest-v1.json`, the individual
traces, and
`docs/reports/m5.2-01-representative-expert-traces.md`. The descriptive result
classifies the existing 8 GiB recommendation as `inconclusive`; no cache
simulation or cache-policy change was performed. ADR 0038 records the v2
trace schema and measurement contract.

M5.2-02 is complete as a deterministic, simulation-only replay over all eight
accepted corpus traces. The input manifest validates the canonical artifact,
M4 baseline/provenance, corpus aggregate, trace hashes, fixture boundaries,
ordinals, ranges, payload sizes, and M5.1 record adapter before replay. The
results cover per-session cold caches, manifest/reverse persistent orders,
binary 1/2/4/6/8/12/16/24/32/48 GiB payload budgets, strict global LRU,
architecture-only and calibrated layer LRU diagnostics, observed LFU,
segmented LRU, and offline Belady. The descriptive decision is to classify
8 GiB as `useful_for_selected_workloads`, retain strict global LRU for the next
runtime experiment, and validate 8 versus 16 GiB without executing that matrix
in this task. Results are recorded in
`models/qwen3-30b-a3b/m5.2-02-cache-simulation-results-v1.json`, the input
manifest, and
`docs/reports/m5.2-02-corpus-cache-simulation.md`; ADR 0039 records the
simulation policy contract and decision. No Rust runtime, ExpertCache,
artifact, numerical path, or dense-residency implementation changed.

M5.2-03 is complete for review as a representative full-runtime validation of
the selected strict global-LRU policy at exact 8 GiB and 16 GiB payload
budgets. Six workload classes were executed, with Tier-A, long-context, and
long-decode repeated twice per budget. All 18 runs matched the exact M5.2-02
simulation counters, retained deterministic generated IDs and request traces,
preserved KV and bounded-memory invariants, and reported zero oversized-entry
or blocked-eviction events. The cache remains
`accepted_with_workload_limitations`: 8 GiB is useful for selected workloads,
16 GiB is useful for cacheable workloads, and neither is a universal preset.
Results are recorded in
`models/qwen3-30b-a3b/m5.2-03-runtime-cache-results-v1.json`, validated traces
and metrics, and `docs/reports/m5.2-03-representative-runtime-cache-validation.md`;
ADR 0040 records the decision. No cache policy, runtime semantics, artifact,
or numerical path changed.

Exact next task after review was the M5.3-01 mmap/coalesced expert access
study; that task is now recorded below as complete for review.

M5.3-01 is complete for review as a storage-path measurement and prototype
selection study. The current reader was instrumented behind the
`m5-3-instrumentation` feature, the canonical artifact layout was validated,
authoritative miss ranges were replayed, and hash-checked layer-47 storage
microbenchmarks were recorded. Exact-adjacent grouping gave only a small
operation reduction; broad layer batching caused severe over-read. Persistent
handles and mmap were not selected. The next selected prototype is reusable
aligned read buffers, and it has not started. Evidence is recorded in
`docs/reports/m5.3-01-expert-access-study.md`,
`models/qwen3-30b-a3b/m5.3-01-expert-access-results-v1.json`, and ADR 0041.

M5.3-02 is complete for review as a feature-gated reusable aligned staging
buffer prototype. The implementation preserves the reference reader, cache
policy, artifact layout, leases, request order, and numerical execution. It
passes byte-equivalence/lifecycle tests and a 24-run full-model matrix across
Tier-A control, Thai, special-token, code, long-context, and long-decode
fixtures at exact 8 and 16 GiB budgets. All runtime counters and traces match
the M5.2-02 simulation and all correctness/budget invariants pass. The
isolated microbenchmark eliminates per-miss allocations, but the full-model
matrix shows no generalizable end-to-end benefit and slower timing in 9 of 12
matched comparisons. The prototype is classified
`microbenchmark_only_value`; the reference reader remains the default. Results
are recorded in
`models/qwen3-30b-a3b/m5.3-02-reusable-buffer-results-v1.json`, the 72-file
runtime evidence directory, the storage benchmark, and
`docs/reports/m5.3-02-reusable-read-buffer.md`; ADR 0042 records the decision.

Exact next task after review: `M5.3-03 Compute profiling`. Do not start it in
this task.

M5.3-03 is complete for review as a feature-gated hierarchical compute
profiling study. The profiler preserved the reference reader, strict global
LRU, numerical execution, request order, and bounded residency across Tier-A,
code, long-context, and long-decode full-model workloads at exact 8 and 16
GiB budgets. All eight detailed rows and three profiling-mode comparison rows
passed deterministic non-timing validation and exact M5.2 simulation-counter
comparison. The measured runtime is storage-bound: the cache lookup/expert
load path is 71.6--76.4% of profile time, while expert MLP is 4.1--5.5% and
LM head is 2.8--3.8%. No kernel, reader default, cache policy, artifact, or
numerical path changed. The historical M4 guard test was corrected to use a
historical task snapshot for repeated-build validation while continuing to
reject current M5 progress. Results are recorded in
`models/qwen3-30b-a3b/m5.3-03-compute-profile-results-v1.json`, the aggregate
JSON, `docs/reports/m5.3-03-compute-profile.md`, and ADR 0043.

The selected next prototype is an isolated read-only mmap expert-access study.
It is not implemented here and must remain feature-gated and outside the
default runtime path.

M5.3-04 is complete for review as an isolated read-only mmap expert-access
prototype. The `clr-mmap` boundary maps complete expert shards lazily and
copies validated ranges into owned expert storage; the reference reader remains
default. Byte-equivalence, lifecycle, cleanup, deterministic trace, cache, KV,
and bounded-memory gates passed, and all 16 reference/mmap full-runtime runs
matched M5.2 simulation exactly. Mmap regressed all eight paired timing
comparisons with a median `+5.92%` change and raised measured working set to
29.46--39.00 GiB while mapping 108 GiB of virtual shard space. The prototype
is classified `insufficient_runtime_value`; no mmap promotion or mapping-cache
follow-up is selected. Evidence is recorded in
`models/qwen3-30b-a3b/m5.3-04-mmap-results-v1.json`,
`models/qwen3-30b-a3b/m5.3-04-mmap-benchmark-v1.json`,
`docs/reports/m5.3-04-mmap-expert-access.md`, and ADR 0044.

Exact next task after review: stop the current storage-access optimization path
due insufficient runtime value. M5.3-04 is complete for review; mmap is
rejected for production adoption. Do not start another storage-access
optimization without a new reviewed, measurement-first proposal.

## Standard verification commands

Run before closing every milestone:

```powershell
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p clr-cli
```
