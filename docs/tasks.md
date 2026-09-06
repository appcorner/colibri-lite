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
- [x] M5.4-01 Resident-dense plus strict global-LRU simulation (complete for review).
- [x] M5.4-02 Measurement-only resident-dense runtime prototype (complete for review; 24-row paired runtime matrix recorded; measurement-only with no production/default adoption authorization).

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

M5.4-01 is complete as a deterministic, simulation-only resident-dense
candidate study. It validates the complete eight-fixture M5.2 corpus and uses
the six fixtures with recorded full-runtime dense-read evidence for the
candidate matrix. Under total-RAM accounting and strict global LRU, resident
dense models 40.43% total logical-read reduction at 8 GiB and 56.90% at
16 GiB, compared with 16.31% and 21.36% for streamed dense in the same
six-fixture subset. At 8 GiB resident dense leaves only 1.981 GiB for experts
and records no simulated expert hits; at 16 GiB it retains 27.64% expert-byte
hits. These are modeled logical-read results, not latency or throughput claims.
Evidence is recorded in
`models/qwen3-30b-a3b/m5.4-01-resident-dense-simulation-v1.json`,
`docs/reports/m5.4-01-resident-dense-simulation.md`, and
`scripts/simulate_m5_4_resident_dense.py`. No Rust runtime, artifact, cache
policy, numerical path, or production default changed.

The simulation selects resident dense plus strict global LRU for a separate
measurement-only runtime prototype review. M5.4-02 is not authorized by this
simulation result alone; it must preserve the frozen F32 invariants, explicit
total-RAM accounting, and the reference reader unless a separate review
approves a runtime change. The two corpus fixtures without M5.2 full-runtime
dense-read evidence remain outside the candidate matrix.

## M6 - Hardware-aware performance runtime

M6 is the approved pivot: maximize measured tokens/s within explicit RAM,
VRAM, I/O, context, quality, and correctness constraints. `reference-f32-v1`
remains the authoritative oracle. Do not start M6.1 before M6.0 is accepted.

### M6.0 - Freeze reference runtime and backend-neutral contracts

- [x] M6.0-01 Create a `reference-f32-v1` manifest pinning source, artifacts,
  fixture hashes, tolerances, router selections, and baseline data.
- [x] M6.0-02 Define backend-neutral tensor, operation, and execution
  contracts without leaking Qwen or storage policy into `clr-core`.
- [x] M6.0-03 Define a differential report that identifies the first divergent
  stage and records comparison tolerance.
- [x] M6.0-04 Freeze deterministic English and Thai quality fixtures and
  expected reference outputs.
- [x] M6.0-05 Add contract, failure-mode, and repeatability tests; run standard
  verification and record the baseline.

M6.0-01 is complete. `models/qwen3-30b-a3b/reference-f32-v1-manifest.json`
pins the M4 release provenance, canonical artifact identity, F32 baseline,
comparison schema, tolerance registry, deterministic Tier-A generation,
intermediate evidence, and Tier-B English/Thai evidence. Its 12 integrity
records passed byte-size and SHA-256 validation in
`python.reference.test_reference_f32_manifest`; the existing frozen F32 bundle
and release-provenance tests also passed. No payload was copied, regenerated,
or downloaded. The exact next task is `M6.0-02`.

M6.0-02 is complete. `clr-core` now exposes metadata-only tensor, operation,
budget, request, metrics, and result contracts. The contracts validate generic
operation signatures, exact output metadata, and admitted host/device byte
limits through `RuntimeError::BackendContractViolation`; they do not include
model fields, payload ownership, files, cache policy, device APIs, threading,
or quantization layouts. ADR 0047 records the boundary. `cargo fmt --all
--check`, `cargo check --workspace`, `cargo test --workspace` (131 tests),
`cargo clippy --workspace --all-targets -- -D warnings`, and `cargo run -p
clr-cli` passed. The exact next task is `M6.0-03`.

M6.0-03 is complete. `clr-core::DifferentialReport` compares ordered stages,
checks tensor metadata and value lengths before values, stops at the first
divergence, and records its stage ID/index, category, element/value diagnostics
where applicable, and the exact absolute/relative tolerance plus source used.
ADR 0048 records why router/semantic-margin and quality gates remain separate.
`cargo fmt --all --check`, `cargo check --workspace`, `cargo test --workspace`
(135 tests), `cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo run -p clr-cli` passed. The exact next task is `M6.0-04`.

M6.0-04 is complete. `m6.0-04-quality-fixtures-v1.json` freezes the existing
Tier-B `short_english` and `short_thai` reference inputs and outputs: token
IDs, expected argmax, compact logits, top-20 IDs, finite counts, final-norm
digest, safe-margin router IDs, and integrity records. It does not rerun the
model or widen a numerical/quality gate. Eleven fixture, F32-manifest, and
baseline-bundle integrity tests passed. The exact next task is `M6.0-05`.

M6.0-05 is complete. The M6 reference-contract validator covers deterministic
repeatability plus wrong-reference-ID, corrupted-SHA-256, and changed-expected-
output failures. The verification baseline records `reference-f32-v1`, the
bilingual fixtures, and the historical M4 performance baseline without claiming
a new measurement. Python contract/oracle tests passed 20 tests. Standard
verification passed: `cargo fmt --all --check`, `cargo check --workspace`,
`cargo test --workspace` (135 tests), `cargo clippy --workspace --all-targets
-- -D warnings`, and `cargo run -p clr-cli`. M6.0 is complete. The exact next
task is `M6.1-01`.

M6.0 review is accepted after remediation. The comparator now rejects a
non-finite calculated allowed error, and the cross-contract validator compares
all frozen finite counts to Tier-B source evidence. Reverification on commit
`84f0cff09ed089f4bae84652f9dd800b05ac8990` passed 21 Python tests, 136 Rust
tests, and all standard commands. See `docs/reports/m6.0-review-remediation.md`.

### M6.1 - Hardware and model profiler

- [x] M6.1-01 Define versioned hardware-profile and model-profile schemas.
- [x] M6.1-02 Measure CPU backend kernel throughput and RAM bandwidth.
- [x] M6.1-03 Measure SSD sequential and expert-sized random-read latency and
  throughput with controlled cache-state semantics.
- [x] M6.1-04 Detect usable RAM, available GPU backends, usable VRAM, and
  measured host/device transfer without assuming a GPU is present.
- [x] M6.1-05 Implement `doctor` and `profile-model` with machine-readable,
  reproducible output and explicit confidence/limitations.

- [x] M6.1-06 Validate profile schema, invalid inputs, repeatability bounds,
  and Windows resource-release behavior.

M6.1-01 is complete. `hardware-profile-v1` defines versioned host, CPU, RAM,
storage, backend, benchmark-distribution, and safe-budget records.
`model-profile-v1` defines versioned pinned model/artifact, routed-expert,
KV-cache, precision-inventory, and `reference-f32-v1` quality-reference
records. `measured`, `unavailable`, and `not_run` are distinct states, so
unmeasured inputs cannot be interpreted as performance data. Two schema
contract tests passed; no hardware or model measurement ran. The exact next
task is `M6.1-02`.

M6.1-02 is complete. The tracked release-build evidence records a safe `f32`
1024×4096 matrix-vector median of 1.861 GFLOP/s and a 128 MiB streaming-copy
RAM proxy median of 65.581 GiB/s on the Windows x64 host, including all nine
sorted samples, warm-ups, repetitions, clock semantics, checksums, and known
limits. `python/reference/validate_m6_1_02_cpu_ram_benchmark.py` validated the
machine-readable record, and its two failure-mode tests passed. This is not an
end-to-end inference claim and does not select a backend. See
`docs/reports/m6.1-02-cpu-ram-benchmark.md`. The exact next task is
`M6.1-03`.

M6.1-03 is complete. The release-build storage evidence uses an opaque,
1,056,964,608-byte temporary payload comprising 56 exact F32 expert-sized
records (18,874,368 bytes each). It records a first-touch sequential result of
2.026 GiB/s and a likely-warm sequential median of 2.985 GiB/s. Expert-sized
random reads recorded first-touch medians of 4.521 ms and 3,984.1 MiB/s, plus
likely-warm medians of 4.812 ms and 3,758.1 MiB/s. Windows physical cache
eviction was not requested, so neither first-touch result claims cold-device
latency. Preflight and cleanup accounting passed; the unique run directory was
removed. The JSON validator and its two failure-mode tests passed. See
docs/reports/m6.1-03-storage-benchmark.md. The exact next task is M6.1-04.

M6.1-04 is complete. A Windows snapshot recorded 47.73 GiB installed RAM,
23.71 GiB available RAM, a 7.16 GiB reserve, and a 16.55 GiB advisory usable
RAM budget. NVIDIA T500 (4 GiB reported by nvidia-smi) and Intel Iris Xe were
detected; CUDA/Vulkan tooling and the DirectML system library are present.
They are inventory only: no colibri GPU backend has passed the M6.3 review, so
all backend availability records are unavailable, usable VRAM and the VRAM
recommendation are zero, and both transfer directions are explicit not_run.
The machine-readable validator and its two failure-mode tests passed. See
docs/reports/m6.1-04-memory-gpu-profile.md. The exact next task is M6.1-05.

M6.1-05 is complete. clr-cli doctor composes explicit CPU/RAM, storage, and
memory/GPU evidence paths into the versioned hardware-profile-v1 schema, while
profile-model composes the pinned reference, quality, and release evidence into
model-profile-v1. Both require explicit timestamp, runtime commit, and output
path; no model payload is copied or downloaded. The recorded hardware profile
has partial confidence and keeps all unavailable GPU/transfer budgets at zero.
Both emitted profiles passed Draft 2020-12 schema validation. See
docs/reports/m6.1-05-doctor-and-profile-model.md. The exact next task is
M6.1-06.

M6.1-06 is complete. The tracked doctor and model profiles passed their Draft
2020-12 schemas. Missing evidence and missing-required-option cases both
returned CLI exit code 2. With fixed input paths, timestamp, and runtime commit,
two doctor runs and two profile-model runs were byte-identical. The Windows
handle-release test successfully renamed a JSON input after reading and a JSON
output after writing, then cleaned its unique temporary directory. See
docs/reports/m6.1-06-profile-validation.md. M6.1 is complete; the exact next
task is M6.2-01.

M6.1 review is accepted after remediation. Profile-model now rejects the
doctor-only --rust-version option, and the complete M6.1 verification passed
again. See docs/reports/m6.1-review-remediation.md.

### M6.2 - First placement planner

- [x] M6.2-01 Define planner input, candidate-plan, estimate, and rejection
  contracts.
- [x] M6.2-02 Implement analytical cost-model calculations from profile data;
  no hard-coded machine performance values.
- [x] M6.2-03 Enumerate supported RAM, VRAM, and SSD placement candidates.
- [x] M6.2-04 Enforce RAM/VRAM/context constraints and explain rejections.
- [x] M6.2-05 Implement `plan` and test deterministic ranking and boundary
  budgets.
- [x] M6.2-06 Compare selected estimates against a recorded benchmark set and
  report error rather than silently retuning the model.

M6.2 review remediation is complete. Planner results now preserve required
compute-work provenance, reject zero context at the CLI boundary, and pass the
complete M6.2 verification again. See `docs/reports/m6.2-review-remediation.md`.

### M6.3 - Native quantized vertical slice

- [x] M6.3-01 Propose one precision/backend candidate with dependency,
  licensing, unsafe-boundary, and provenance review.
- [x] M6.3-02 Implement direct quantized expert consumption for one layer;
  prohibit whole-expert expansion to F32 in the candidate path.
- [x] M6.3-03 Preserve F32 router, norms, sensitive operations, and the
  executable reference comparison path.
- [x] M6.3-04 Compare router IDs, checkpoints, logits, Thai/English fixtures,
  and quality metrics against `reference-f32-v1`.
- [x] M6.3-05 Benchmark cold/warm throughput, TTFT, RAM, VRAM, physical reads,
  cache hit rate, and bytes/token with repeated runs.
- [x] M6.3-06 Hold a stop/go review before all-layer implementation.

M6.3-04 is complete. The temporary direct-consumption Layer-0 group-128
artifact was generated from the canonical F32 Layer-0 experts under the
temporary-artifact policy, then compared end-to-end against the frozen English
and Thai fixtures. The final router IDs, argmax IDs, and frozen top-20 IDs all
matched; Layer-0 checkpoint and logit drift are recorded without applying the
F32 tolerance registry to the quantized candidate. See
`docs/reports/m6.3-04-reference-f32-comparison.md`. The exact next task is
M6.3-05.

M6.3-05 is complete. Three independent cold/warm pairs measured the candidate
with conversion excluded from timing. Warm candidate payload reads fell to zero
for the frozen two-token fixture, but median decode remained about 0.021 tok/s
because Layers 1--47 remain scalar F32. Logical I/O, explicit payload RAM
accounting, cache metrics, zero admitted VRAM, and the unavailable
per-process physical-read metric are recorded without claiming cold-device I/O.
See `docs/reports/m6.3-05-cold-warm-benchmark.md`. The exact next task is
M6.3-06.

M6.3-06 is complete with a NO-GO decision. The slice must not extend to 48
layers: English logit drift lacks a justified candidate admission rule, warm
throughput has no material end-to-end benefit, and physical per-process I/O and
working-set peak evidence are missing. M6.4 is blocked and must not start. See
`docs/reports/m6.3-06-stop-go-review.md` and ADR 0057.

### M6.3-R1 - Measurable re-entry and candidate admission

Status: approved re-entry plan; no task below is complete until its evidence
passes. The stopped M6.3 group-128 candidate remains stopped and cannot be
reused as the default candidate.

- [x] M6.3-R1.1a Characterization remediation (append-only): run only the
  `code_newline` held-in candidate study defined by ADR 0059 and resolved by
  ADR 0060; `tier_a_control` requires a separately frozen-reference task. This
  does not revise the completed R1.1 outcome and does not authorize R1.2. ADR
  0064 corrects the control-only tolerance linkage before any candidate run:
  M4.2 fixture-specific values may not be reused numerically for
  `code_newline`; two persisted canonical F32 controls must freeze the new
  fixture-scoped budgets from pre-registered M4.2 formulas first.
  The control calibration, direct candidate reader, candidate-correlated
  telemetry contract, failure-mode tests, and post-reader canonical control
  now pass; the harness is `harness_ready_for_candidate_execution`. This task
  remains open until the two pre-registered candidates are characterized,
  ranked, and admitted or rejected without using held-out fixtures.
  The first bounded execution attempt stopped before candidate launch because
  Kernel ETW requires an elevated Administrator token; its temporary group-64
  artifact was removed by reviewed dry-run/apply cleanup. See
  `docs/reports/m6.3-r1-1a-execution-blocker.md`.
  A subsequent elevated group-64 run reached the full-model workload but
  failed before checkpoint comparison because the ETW child working directory
  was the repository root. ADR 0066 now requires and pre-validates the crate
  working directory; the failed run was fully cleaned and is not admission
  evidence. See `docs/reports/m6.3-r1-1a-group64-run1-working-directory-failure.md`.
  The next group-64 run reached checkpoint/logit comparison and passed the
  fixed-logit cap, but its 805 logical candidate reads produced no correlated
  Kernel-Disk bytes, so it is invalid admission evidence rather than a zero-I/O
  pass. ADR 0067 now pre-registers a reboot-bound, metadata-only arm and
  one-shot ETW authorization without weakening the physical-I/O or numerical
  gates. The invalid run was removed by reviewed cleanup. The exact next step
  is a fresh group-64 conversion followed by cold-cache Phase A. The protocol
  validator, one-shot ETW binding, telemetry failure tests, and required
  workspace verification pass; group-32 and R1.2 remain blocked.
  Group-64 cold-cache run 1 of 3 subsequently passed every run gate, including
  the ADR 0065 physical-I/O gate with 917,504 correlated candidate disk-read
  bytes and zero lost events. Its fixed-logit maximum absolute error was
  `2.375e-2`, below the unchanged `0.05` cap; the first divergence was
  `layer0.selected_expert_output`. This single run is neither admission nor
  ranking, so the pre-registered characterization/admission and telemetry
  contracts remain `not_run` and `pre_registered_not_executed`. Run 2 requires
  a fresh conversion, fresh flat run directory, new Phase A, and reboot.
  Group-32 and R1.2 remain blocked. See
  `docs/reports/m6.3-r1-1a-group64-coldcache-run1.md`.
  Group-64 cold-cache run 2 of 3 then passed every run gate, including the
  ADR 0065 physical-I/O gate with 1,048,576 correlated candidate disk-read
  bytes and zero lost events. Its fixed-logit maximum absolute error was
  `2.375e-2`, below the unchanged `0.05` cap, and the first divergence was
  again `layer0.selected_expert_output`. Every numerical checkpoint error and
  both the Layer-0 checkpoint and final-logits hashes are bit-identical to
  run 1, so this is repeated-run determinism evidence and still not admission
  or ranking evidence. Run 3 requires a fresh conversion, fresh flat run
  directory, new Phase A, and reboot. Group-32 and R1.2 remain blocked. See
  `docs/reports/m6.3-r1-1a-group64-coldcache-run2.md`.
  A third group-64 attempt was then invalid at the ADR 0065 physical-I/O gate:
  805 logical candidate File Read events produced zero correlated Kernel-Disk
  events, so the physical-I/O status was `not_measured` with a null candidate
  read-byte count. Its process, numerical, and direct-consumption gates
  otherwise passed, and every numerical checkpoint error and both the Layer-0
  checkpoint and final-logits hashes were bit-identical to run 1 and run 2, but
  a numerical pass cannot compensate for an unmeasured physical read. The
  invalid attempt does not count as run 3 of 3, so the valid group-64 set still
  contains only run 1 and run 2 and the three-run set is not closed. A fresh
  retry requires a fresh conversion, fresh flat run directory, new Phase A, and
  one reboot with a prompt arm; no part of the invalid attempt may be reused.
  Group-32 and R1.2 remain blocked. See
  `docs/reports/m6.3-r1-1a-group64-coldcache-run3-invalid-physical-io.md`.
  A fresh group-64 run 3 then passed every run gate, including the ADR 0065
  physical-I/O gate with 1,048,576 correlated candidate disk-read bytes and zero
  lost events. Its fixed-logit maximum absolute error was `2.375e-2`, below the
  unchanged `0.05` cap, and the first divergence was again
  `layer0.selected_expert_output`. Every numerical checkpoint error and both the
  Layer-0 checkpoint and final-logits hashes are bit-identical across all three
  valid runs, and exact router IDs were preserved in each. The group-64
  three-valid-run characterization set is therefore closed; the earlier invalid
  attempt is not counted. Passing the physical-I/O gate does not establish that
  the complete 641,728,512-byte artifact was read from the device, since the
  correlated total is a small fraction of the file and most logical reads may
  still have been cache-served; that limitation carries into the final R1.1a
  review. This is still not admission or ranking, and the
  characterization/admission and telemetry contracts remain `not_run` and
  `pre_registered_not_executed`. Group-32 characterization is the next work and
  R1.2 remains blocked. See
  `docs/reports/m6.3-r1-1a-group64-coldcache-run3.md`.

  Group-32 cold-cache run 1 of 3 passed every run gate, including the ADR 0065
  physical-I/O gate with 1,048,576 correlated candidate disk-read bytes and zero
  lost events. Its fixed-logit maximum absolute error was `2.367e-2`, below the
  unchanged `0.05` cap, and the first divergence was again
  `layer0.selected_expert_output`. Exact Layer-0 router IDs were preserved. Every
  numerical checkpoint error and both the Layer-0 checkpoint and final-logits
  hashes were recorded; bit-identical repetition across three runs will be
  assessed when the set closes. Direct packed-consumption verification reads
  679,477,248 bytes with peak packed expert 5,308,416 and zero complete F32
  materializations. This single run is neither admission nor ranking, so the
  pre-registered characterization/admission and telemetry contracts remain
  `not_run` and `pre_registered_not_executed`. Group-64 is closed; group-32
  run 2 requires a fresh conversion, fresh flat run directory, new Phase A, and
  reboot. R1.2 remains blocked. See
  `docs/reports/m6.3-r1-1a-group32-coldcache-run1.md`.

  Group-32 cold-cache run 2 of 3 then passed every run gate, including the ADR
  0065 physical-I/O gate with 1,048,576 correlated candidate disk-read bytes,
  zero events lost, and zero buffers lost. Its fixed-logit maximum absolute
  error was `2.36730575561523438e-2`, below the unchanged `0.05` cap, and the
  first divergence was again `layer0.selected_expert_output`. Exact Layer-0
  router IDs were preserved. Every numerical checkpoint error and both the
  Layer-0 checkpoint hash `20a3a2db…c141cbb` and final-logits hash
  `5a7e254c…90986525` are bit-identical to group-32 run 1, and the
  direct-consumption metrics are identical as well: 679,477,248 verification
  bytes, 169,869,312 payload bytes, peak packed expert 5,308,416, and zero
  complete F32 weight materializations. This is therefore repeated-run
  determinism evidence for group-32 and still not admission or ranking
  evidence. Run 3 requires a fresh conversion, fresh flat run directory, new
  Phase A, and reboot. The pre-registered characterization/admission and
  telemetry contracts remain `not_run` and `pre_registered_not_executed`, and
  R1.2 remains blocked. See
  `docs/reports/m6.3-r1-1a-group32-coldcache-run2.md`.

  Group-32 cold-cache run 3 of 3 then passed every run gate, including the ADR
  0065 physical-I/O gate with 1,048,576 correlated candidate disk-read bytes,
  zero events lost, and zero buffers lost. Its fixed-logit maximum absolute
  error was `2.36730575561523438e-2`, below the unchanged `0.05` cap, and the
  first divergence was again `layer0.selected_expert_output`. Exact Layer-0
  router IDs were preserved across all three runs. Every numerical checkpoint
  error, both the Layer-0 checkpoint hash `20a3a2db…c141cbb` and final-logits
  hash `5a7e254c…90986525`, and every direct-consumption metric (679,477,248
  verification bytes, 169,869,312 payload bytes, peak packed expert 5,308,416,
  zero complete F32 materializations) are bit-identical across all three runs.
  The group-32 three-valid-run characterization set is therefore closed.
  Group-64 and group-32 characterization are both complete. Passing the
  physical-I/O gate does not establish that the complete 679,477,248-byte
  artifact was read from the device; the correlated total is a small fraction
  of the file and that limitation carries into the final R1.1a review. The
  The final R1.1a candidate review is now complete. ADR 0068 applies the
  pre-registered ADR 0059 ranking and records group-32 as the characterization
  winner because its fixed-logit maximum absolute error is lower than group-64.
  The append-only remediation does not revise the historical R1.1
  `no_candidate_admitted` outcome, so R1.2 remains blocked. The pre-registration
  contracts remain byte-identical; closure is recorded separately in
  `models/qwen3-30b-a3b/m6.3-r1-1a-final-review-v1.json`. See
  `docs/reports/m6.3-r1-1a-group32-coldcache-run3.md` and
  `docs/reports/m6.3-r1-1a-final-candidate-review.md`.

- [x] M6.3-R1.1c Verified M4 Pinned-Source Recovery: the verified 17-file
  Safetensors subset was atomically promoted as the read-only canonical minimal
  oracle source, fully rehashed, registry-bound, ACL-checked, and proven by an
  offline selective loader. ADR 0062 supersedes only ADR 0061's acquisition
  method; this is not a 26-file Transformers snapshot and does not start R1.1b,
  R1.2, or M6.4.

- [x] M6.3-R1.1b Freeze code_newline Layer-0 Checkpoint References: frozen
  offline Transformers-F32 Layer-0 router/expert/MoE checkpoint payload,
  existing router guard, tolerance linkage, and byte-identical two-process
  hashes are recorded in ADR 0063 and the versioned reference record.

- [x] M6.3-R1.0 Implement and validate a release-process telemetry harness for
  working set/private bytes, logical reads, process-correlated physical I/O,
  cache-state labels, and collector failure states using the unchanged
  `reference-f32-v1` path.
- [x] M6.3-R1.1 Run a deterministic candidate-admission study; select exactly
  one Qwen3 expert layout or record `no_candidate_admitted`, with pre-registered
  numerical gates, provenance, hashes, and direct-consumption proof. Completed
  with the valid historical outcome `no_candidate_admitted`: group-64 and
  group-32 were converted and reconverted byte-identically from the read-only
  canonical Layer-0 F32 shard, but neither supplied the required
  characterization-stage routed-expert/checkpoint/logit-envelope evidence. See
  ADR 0058 and `docs/reports/m6.3-r1-1-candidate-admission.md`.
- [x] M6.3-R1.1d Group-32 Admission Amendment Review: preserve the historical
  R1.1 `no_candidate_admitted` record, consume the closed R1.1a characterization
  evidence, and prospectively admit exactly one candidate for R1.2 only. ADR
  0069 admits `cpu-safe-rust-int8-group32-layer0-r1-1a`; the unchanged `0.05`
  pre-registered cap remains its held-in logit envelope. R1.3 and M6.4 remain
  blocked. Evidence is recorded in
  `models/qwen3-30b-a3b/m6.3-r1-1d-admission-amendment-v1.json` and
  `docs/reports/m6.3-r1-1d-group32-admission-amendment.md`.
- [x] M6.3-R1.2 Held-out Quality Validation: compare the admitted group-32
  candidate with `reference-f32-v1` using the pre-registered ADR 0070 gates.
  The F32-only path first froze two-token greedy references before candidate
  execution: `short_english=[0,358]` and `short_thai=[7360,91]`. The fresh
  group-32 candidate then passed exact prompt top-20, greedy, Layer-0/24/47
  safe-margin router IDs, exact two-token generation, finite Layer-0 MoE error,
  the unchanged effective `0.05` prompt-logit envelope, and two-execution
  repeatability on both held-out fixtures. ADR 0071 records `R1.2 = GO` and
  authorizes R1.3 only; M6.4 remains blocked. Evidence is in
  `models/qwen3-30b-a3b/m6.3-r1-2-quality-result-v1.json` and
  `docs/reports/m6.3-r1-2-held-out-quality-validation.md`.
- [x] M6.3-R1.3 Paired Performance and I/O Measurement: 20/20 valid
  release-process samples completed under contract v5 across five cold and five
  warm F32/candidate pairs. ADR 0077 records a valid measurement set. The
  candidate has a 5/5 logical-byte reduction of 0.8919485285%, but no
  directional TTFT, prefill, decode, working-set, private-byte, or physical-I/O
  win. Evidence is in
  `models/qwen3-30b-a3b/m6.3-r1-3-paired-samples-v1.json`,
  `models/qwen3-30b-a3b/m6.3-r1-3-paired-result-v1.json`, and
  `docs/reports/m6.3-r1-3-paired-performance-io-result.md`.
- [x] M6.3-R1.4 Re-entry Review: ADR 0078 closes `NO-GO for promotion` of
  `cpu-safe-rust-int8-group32-layer0-r1-1a`. R1.2 quality remains valid, but
  R1.3 does not show material repeatable runtime value for the current direct
  safe-Rust path. M6.4 remains blocked; any future re-entry requires a new
  pre-registered runtime hypothesis. Evidence is in
  `models/qwen3-30b-a3b/m6.3-r1-4-reentry-review-v1.json` and
  `docs/reports/m6.3-r1-4-reentry-review.md`.

### M6.3-R2 - Expert compute bottleneck localization

Status: diagnostic re-entry only. R2 may measure the unchanged Layer-0 F32 and
admitted group-32 paths, but it may not optimize either path or authorize M6.4.

- [x] M6.3-R2.0-PR Pre-register Expert Compute Bottleneck Localization: ADR 0079
  freezes two held-out fixtures, `compute_only_preloaded` and
  `load_plus_compute` views, five paired processes per fixture/view, timer and
  observer-effect gates, stage taxonomy, and prospective classification rules.
  Contract: `models/qwen3-30b-a3b/m6.3-r2-0-localization-contract-v1.json`,
  SHA-256 `02322820758848b5ebfe3f020db026f1aa5e3dd54ce007fb71202f50e54aceca`.
- [x] M6.3-R2.0a Implement measurement-only timing/byte instrumentation and a
  validator without changing arithmetic, cache policy, quantization, or expert
  selection. The test-only `m6-3-r2-localization` feature now observes the
  existing F32 and group-32 load/compute/routing paths, calibrates no-op timer
  cost, supports same-binary observer enabled/disabled controls, and preserves
  exact tiny-path outputs. The validator locks the 40-sample matrix, 20
  observer-control pairs, byte/timer/retry gates, and ADR 0079 classification.
  Focused verification passed 5 Rust and 13 Python tests. See
  `docs/reports/m6.3-r2-0a-instrumentation-validator.md`.
- [x] M6.3-R2.0b Freeze reference-only Layer-0 expert-input/router fixtures for
  `short_english` and `short_thai`, then freeze execution identities. The
  reference record SHA-256 is
  `c0490d2a40a214579e7633fcd4706e64186d183a82af6d176eaa1918d7f056e2`;
  the final measurement binary SHA-256 is
  `aced59ad9ffa9ed153f1d817f37beeda521061f0d2025935aab8c341889efb8e`.
  Observer controls v5 passed 20/20 pair-processes on that exact binary.
- [x] M6.3-R2.0c Run 40 valid release-process localization samples: two fixtures
  x two views x five F32/candidate pairs x two paths, with no automatic retry.
  Immutable sample SHA-256 is
  `e38688ae1309c411c11309def37b5b2ef6e84eb634dee244bb565a41b7eb7942`.
- [x] M6.3-R2.0d Apply ADR 0079 classification exactly. ADR 0086 closes R2.0
  `packed_projection_compute_bound`: candidate compute-only is slower 5/5 for
  both fixtures and packed gate+up+down consumes median 91.44% English / 92.04%
  Thai expert time. Result SHA-256 is
  `044e6d1d6a5b59953c2b603489d24e304290d5ce0d85d82b4de6d4c8e5b362cb`.
  R2.1 hypothesis design is authorized; optimization implementation and M6.4
  remain blocked.
- [x] M6.3-R2.1-PR Pre-register the AVX2+FMA native packed-projection vertical
  slice in ADR 0087. The unchanged group32 artifact remains authoritative;
  native code may replace only gate/up/down packed projection arithmetic behind
  Rust orchestration. The contract freezes quality-before-performance gates,
  one isolated reviewed FFI/unsafe boundary, resource limits, all six balanced
  native/scalar/F32 triplet orders, and 72 official performance samples.
  Contract SHA-256:
  `76bfc4a850a8d89fb5901eda338568b7226b36acb7207324575568ea21f6cc2a`.
- [x] M6.3-R2.1a Implement the Layer-0 AVX2+FMA kernel and smallest FFI wrapper.
  ADR 0088 records the dependency/unsafe review. Runtime AVX2+FMA detection,
  scalar fallback, canary/bounds tests, zero native scratch, zero persistent
  prepack, and zero complete F32 weight materializations are verified. Default
  runtime remains scalar; no official performance timing was run. See
  `docs/reports/m6.3-r2-1a-native-implementation-review.md`.
- [x] M6.3-R2.1b Run held-out quality/correctness gates before performance.
  ADR 0089 closes quality `GO`: both held-out fixtures preserve exact generated
  sequences, prompt top-20/argmax, Layer-0/24/47 router IDs, and repeatability.
  Native-vs-scalar Layer-0 max-abs is `2.3841858e-7` for both fixtures and
  prompt-logit max-abs is `3.0517578e-5` English / `2.5749207e-5` Thai, below
  frozen `0.001` / `0.002` limits. Evidence SHA-256 is
  `f0dea27297c03b2f2283c161c91b369169318c309394c1bb786db43a02ba6113`;
  result SHA-256 is
  `8fe232bfc58381e4a3d602e67705e129dafcf133e785c6c15ca6770e1cf22866`.
- [x] M6.3-R2.1c Freeze one three-path release binary and run all 72 official
  native/scalar/F32 process samples using all six triplet orders. The matrix
  completed 72/72 with no automatic retry; samples SHA-256 is
  `5f35dd5219079fffad361d0b74344636491359abda69f32668af7138613337ce`.
- [x] M6.3-R2.1d Apply the frozen performance/resource gates. ADR 0090 closes
  R2.1 `GO`: all four fixture/view summaries pass, native beats scalar and F32
  in every triplet, and all frozen resource guards remain satisfied. Result
  SHA-256 is `c1df174cb7fa0ab649797255d0beac2087fe728b78bfa9f621a3c87def663e7e`.
  R2.2 design only is authorized; R2.2 implementation, all-layer rollout, and
  M6.4 remain blocked.
- [x] M6.3-R2.2-PR Design and pre-register a three-sentinel-layer proof in ADR
  0091. Layers 0/24/47 are frozen; all other layers remain canonical F32. The
  exact R2.1 group32 grammar/kernel is reused unchanged. Contract SHA-256 is
  `3071f717b9ea12091323e596b9a3d40aa776b34f27f0cc66db63774e493af0c4`.
- [x] M6.3-R2.2a Generalize the packed artifact reader to explicit layer IDs and
  create/freeze independent group32 artifacts for Layers 24 and 47. Layer-0
  reproduction matched the admitted SHA exactly; Layer-24 SHA is
  `890936e78829cf013a141b225c7819cb2c10134fbd2c787e4131f7f41844d9b0`
  and Layer-47 SHA is
  `a2a45a1f407d5cd2303286843001684fc7800953cd2b67995a2c5e62a9e66a33`.
  Frozen Rust reader verification passed all three artifacts; evidence SHA-256
  is `96c424b2007ef844e622b9976be1b7de144304ea1a392ceedbab10c86385fb4c`.
  No new unsafe boundary, kernel optimization, or default-runtime change occurred.
- [x] M6.3-R2.2b Run the frozen three-sentinel held-out quality gate before any
  official performance sample. ADR 0095 closes this gate `NO-GO`: the valid
  ordinal-4 execution fails `short_thai` exact prompt top-20 ordering on the
  scalar group32 run. Result SHA-256 is
  `2b8c841daf9318fcd2b9d08ffdb03dbeef541195ae9f76090ea72e2542bf96be`.
- [x] M6.3-R2.2-D1-PR Pre-register scalar-only `short_thai` quality-drift
  localization across the fixed eight sentinel states. Contract SHA-256 is
  `6f08cdaf5c0d297f887d00c2bd8cccfb6096885772bc8e464aaba9e0eeebdb9b`.
- [x] M6.3-R2.2-D1 Implement/freeze the prompt-only diagnostic harness, then run
  the eight states exactly once in the frozen order. ADR 0097 closes D1:
  Layer 47 alone is sufficient for the lower rank swap; Layers 0+24 form a
  separate pair interaction. Evidence SHA-256 is
  `d6d05c0f5201fde047237082bb01af684215d3c17c5dd242c9357fc23c6bf473`.
- [x] M6.3-R2.2-D2-PR Pre-register depth-sensitive scalar precision
  characterization for Layers 24/47. Candidate groups are fixed at 32/16/8,
  with deterministic coarsest-passing selection and F32 fallback. Contract
  SHA-256 is `ea779da7bb4466c59a478c095ee0342c6932c482426e3331bf472c8296c31d30`.
- [x] M6.3-R2.2-D2a Add characterization-only group16/group8 scalar layout
  support, freeze a generic converter, then freeze all four new artifact
  identities before D2 measurement. Artifact manifest SHA-256 is
  `bd54c87ae20826086a9f8e6b1e06011bb4f1c65fa973494804c3fcb6b79e0a46`.
- [x] M6.3-R2.2-D2b Implement/freeze the local precision characterization
  harness and run the frozen Layer24/47 context matrix exactly once. ADR 0099
  closes D2 `NO-GO` for grouped-int8 granularity: no 32/16/8 candidate meets
  the `0.001` local budget in all contexts. Result SHA-256 is
  `847b146f56e0c04f5836875ab178bab320a76a493df64b6a6ba5d8c517bb518c`.
- [x] M6.3-R2.2-D3-PR Pre-register projection-sensitivity localization for
  gate/up/down at Layers 24/47 before implementing any hybrid representation.
  Contract SHA-256 is
  `0faa6aa9200346aad94b0c69d6c6ed353682bf53410b258c38bfb61224f63ceb`.
- [x] M6.3-R2.2-D3a Implement/freeze the test-only projection-hybrid diagnostic
  seam and execute the frozen 4-context x 7-variant matrix exactly once. ADR
  0101 selects Layer24 `down`-only group32 and Layer47 canonical F32 fallback.
  Result SHA-256 is
  `b7ff3aafae497c5ecc5e482662f8fc87ed79beaf513dc8bf41658f633ecf18a5`.
- [x] M6.3-R2.2-D4-PR Pre-register held-out quality re-entry for the exact hybrid
  policy: Layer0 all-group32, Layer24 down-group32 with F32 gate/up, Layer47 F32.
  Contract SHA-256 is
  `5f2b27b623a0456ad15e14b542816e4d5ad17bf32dacd91099184adfc86fb4fe`.
- [x] M6.3-R2.2-D4a Implement/freeze the hybrid quality harness and execute the
  frozen two-fixture quality re-entry once with one scalar and two native runs.
  ADR 0103 closes D4 `NO-GO`: `short_english` Layer24 scalar-hybrid/F32 local
  max-abs is `0.0010073595` against the frozen `0.001` limit. Result SHA-256 is
  `47dbccb7406ee12fcca5b0a4b48790a749c54388fc5780f5dfbb6765dd012822`.
- [x] M6.3-R2.2-D4D1-PR Pre-register Layer24 down-projection precision
  characterization across both held-out fixtures using existing group32/16/8
  artifacts and canonical/L0-group32 prefixes only. Contract SHA-256 is
  `16c6015c68834c639a57e4a381217e4cac6d3977c1de89316df073266998fb2a`.
- [x] M6.3-R2.2-D4D1a Implement/freeze the cross-fixture Layer24 down-only
  characterization harness and execute the frozen 2-fixture x 2-prefix x
  3-group matrix once. ADR 0105 selects group16 as the coarsest passing group.
  Result SHA-256 is
  `486679e583c1881ef97d7d35ff771cab53f0ba66dd2eac84a5da88929c22bd43`.
- [x] M6.3-R2.2-D4D2-PR Pre-register held-out quality re-entry for Layer0
  all-group32 + Layer24 F32 gate/up and group16 down + Layer47 F32. Contract
  SHA-256 is `d9f35f16387aced95ead63003c6ba56133a72dd9cc09ee0e72840a4ea4b3cfe1`.
- [x] M6.3-R2.2-D4D2a Implement/freeze the group16 hybrid quality harness and
  execute the frozen bilingual quality re-entry once. ADR 0107 closes D4D2
  `NO-GO`: `short_thai` Layer24 scalar-hybrid/F32 local max-abs is
  `0.0012040138` against the unchanged `0.001` limit. Result SHA-256 is
  `4a88b339991f9e5591a3b8c7f2a3665e83f74f830878c440ea8deaeb1453bb57`.
- [x] M6.3-R2.2-D4D3-PR Pre-register sequence-aware Layer24 down-projection
  precision characterization over the full two-token trajectory using only
  existing group32/group16/group8 artifacts. Contract SHA-256 is
  `dddc8b4ec320835686e5c5286b420e5b39bb4f3bbb3ef95e0f6b82bab473470a`.
- [x] M6.3-R2.2-D4D3a Implement/freeze the sequence-aware characterization
  harness and execute the frozen 2-fixture x 3-group trajectory matrix once.
  ADR 0109 selects Layer24 `down` group8 as the first precision that passes all
  measured positions in both held-out fixtures. Result SHA-256 is
  `6bda021e44212caf40117c6a7a80a8dbe451bfd6812968aaf315cae422d07a43`.
- [x] M6.3-R2.2-D4D4-PR Pre-register held-out quality re-entry for Layer0
  all-group32 + Layer24 F32 gate/up and group8 down + Layer47 F32. Contract
  SHA-256 is `3f8cb0c72180b46b8e6fe29d1b27ea50251119cc9f5ce9d3356d0e0eb5e1d9ed`.
- [x] M6.3-R2.2-D4D4a Implement/freeze the group8 hybrid quality harness and
  execute the frozen bilingual quality re-entry once. ADR 0111 closes D4D4
  `GO`; both held-out fixtures pass all frozen quality/local/repeatability gates.
  Result SHA-256 is
  `ab77024116847b59fe53ce8571794b6584139d9ff173a338f485e44d567dc4cc`.
- [x] M6.3-R2.2-D5-PR Pre-register the production-like hybrid artifact/layout,
  backend boundary, and paired performance/resource contract for the validated
  Layer0 group32 + Layer24 group8-down + Layer47 F32 policy. Contract SHA-256 is
  `34b5a5a032bf6251b7d724c115bad9db44fc11f08092d235155e32c304d8acd3`.
- [ ] M6.3-R2.2-D5a Implement/freeze the production-like Layer24 hybrid artifact
  and direct-consumption reader without whole-expert or F32-down materialization.
- [ ] M6.3-R2.2c HISTORICAL/BLOCKED. Its 108-sample contract belongs to the
  failed all-three-group32 candidate and must not be reused for the hybrid.
- [ ] M6.3-R2.2d BLOCKED with R2.2c. No bounded impact/promotion decision may be
  produced from the failed three-sentinel quality candidate.

The R2.0 protocol is
`docs/reports/m6.3-r2-0-expert-compute-bottleneck-localization-protocol.md`.

The complete re-entry protocol is
`docs/reports/m6.3-r1-reentry-proposal.md`. ADR 0069 prospectively admitted the
R1.1a group-32 characterization winner without rewriting the historical R1.1
`no_candidate_admitted` record; ADR 0071 closed R1.2 quality `GO`; ADR 0077
closed R1.3 measurement; and ADR 0078 closes the current re-entry path
`NO-GO for promotion`. ADR 0079 opened diagnostic R2.0; ADRs 0080-0085 record
measurement-method and validation corrections without changing the immutable
localization samples; ADR 0086 closes R2.0 `packed_projection_compute_bound`.
ADR 0087 pre-registers one AVX2+FMA native packed-projection vertical slice;
ADR 0088 accepts its isolated dependency/unsafe boundary and R2.1a is complete.
ADR 0089 closes R2.1b held-out quality `GO` without making a performance claim.
ADR 0090 closes R2.1c/R2.1d `GO` after all 72 official samples pass the frozen
performance and resource gates. ADR 0091 freezes the Layers 0/24/47 sentinel
proof; ADRs 0092-0094 record quality-execution transport recovery without
changing the frozen candidate; and ADR 0095 closes R2.2b `NO-GO` on the valid ordinal-4 `short_thai`
scalar-group32 top-20 ordering failure. ADRs 0096-0097 pre-register and close
D1, which localizes the original drift; ADR 0099 closes D2 grouped-granularity
characterization; ADR 0101 closes D3 with Layer24 `down`-only group32 and
Layer47 F32 fallback. ADR 0102 pre-registers D4 held-out quality re-entry for
that hybrid policy. The exact next task is `M6.3-R2.2-D4a` implementation and
execution. Historical R2.2c/R2.2d, all-layer rollout, and M6.4 remain blocked.

## Standard verification commands

Run before closing every milestone:

```powershell
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p clr-cli
```
