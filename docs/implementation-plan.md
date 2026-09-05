# colibri-lite-rs Implementation Plan

## Purpose

Build a Rust-first, hardware-aware inference runtime for Mixture-of-Experts
models that maximizes measured tokens per second within explicit RAM, VRAM,
storage-I/O, context, quality, and correctness budgets.

The first supported architecture is Qwen3-MoE. The first full-size target is
Qwen3-30B-A3B on Windows x64. The runtime must prioritize numerical
correctness and reproducible evidence. It may use RAM, VRAM, SSD, and a
selected compute backend when measurement shows a performance benefit; minimum
RAM is no longer the primary objective.

This document defines milestone scope and engineering gates. Executable work
items and status are tracked in [tasks.md](tasks.md).

## Product boundary

`colibri-lite-rs` is not a Rust rewrite of `llama.cpp`, `ik_llama.cpp`, or
Colibri. It is a focused runtime for MoE models whose total weights may exceed
the configured resident-memory budget.

North-star capability:

```text
Given a Qwen3-MoE model and hardware budgets, measure the machine, choose a
reproducible placement/execution plan, enforce RAM and VRAM limits, and
produce numerically validated tokens at the highest supported tokens/s on
Windows x64.
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
10. Do not promote a backend, placement, or precision policy without
    repeatable end-to-end evidence and differential correctness gates.

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

Status: complete and review-accepted after remediation recorded in
`docs/reports/m6.0-review-remediation.md`.

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

Status: complete. M4.2 is completed with documented numerical variance.

Scope:

- Validate selected layers/tensors against Transformers before quantization.
- Run a short deterministic prompt or token sequence.
- Compare expert selections and selected intermediate outputs.
- Record peak RAM and bytes read even if performance is poor.

Exit condition:

The unoptimized storage-aware path is numerically credible and debuggable.

#### M4.3 - Evidence-driven quantization

Status: complete with documented quantization rejection. The ordered Rust F32
path remains authoritative; expert INT8 per-output-channel is retained for
diagnostics but rejected as a full-model production candidate. No quantization
or optimization runtime code has begun. The next phase pivots to a simulation-
first F32 memory-hierarchy study.

Scope:

- Select a first expert quantization only after measuring memory and I/O.
- Keep sensitive dense/router tensors at a higher precision when evidence
  supports it.
- Validate degradation against a defined prompt/evaluation set.
- Treat ik_llama.cpp as a performance and quantization baseline, not a code
  target.
- Preserve the F32 correctness invariants while simulating resident dense
  weights and trace-driven expert-cache sizing at reviewed RAM budgets.

#### M4.4 - Reproducible baseline

Status: complete. M4 is closed with the release-provenance record and the
approved `m4-full-qwen3-baseline-v1` tag. No optimization or M5 implementation
has started.

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

### M5 - Memory hierarchy and performance recovery

Status: review closure after M5.4-02. M5 evidence has confirmed the
correctness-valid F32 path and the low-memory expert-streaming design, but has
not established production-ready performance. This phase must not be described
as a production-performance milestone.

Purpose:

- Measure memory hierarchy and expert-cache behavior against the frozen M4 F32
  correctness evidence.
- Preserve the reference reader, strict global LRU policy, numerical contracts,
  artifact format, and configured payload-byte budget unless a separately
  reviewed experiment supplies contrary evidence.
- Stop a storage-access candidate when its end-to-end evidence is insufficient;
  microbenchmark improvements alone are not acceptance evidence.

Current closure findings:

- M5.1 and M5.2 established deterministic expert traces, cache-policy
  simulation, and representative 8/16 GiB strict-global-LRU runtime evidence.
- M5.3-03 attributes approximately 71.6--76.4% of profiled time to cache
  lookup/expert loading; this identifies a measured area, not an automatic
  approval for another storage optimization.
- The reusable-buffer prototype has diagnostic/microbenchmark value only and
  is not the default reader.
- The isolated mmap prototype is rejected for runtime adoption: it regressed
  all paired full-runtime comparisons (median +5.92%) and increased measured
  peak working set to 29.46--39.00 GiB.
- M5.4-02 completed a 24-row paired resident-dense/streamed runtime matrix at
  8 and 16 GiB. Correctness, strict global-LRU accounting, and total-RAM
  enforcement passed, but timing is directional and physical I/O/page-cache
  behavior were not measured; the candidate did not establish runtime value.
- Strict global LRU remains the selected cache policy and the reference reader
  remains the default production path.

The current storage-access optimization path is stopped after decision review.
Resident dense weights plus strict global LRU remain a measurement-only,
feature-gated prototype and are not a production/default candidate. The
project remains a research runtime; reopening this path requires a separately
reviewed experiment with controlled, repeatable end-to-end performance
evidence and explicit physical-I/O/working-set measurement semantics.

### M5.4 - Resident-dense candidate study

Status: M5.4-01 simulation and M5.4-02 measurement-only runtime prototype are
complete for review. Production/default adoption is not authorized.

M5.4-01 is a simulation-only study over the validated M5.2 corpus and the six
fixtures with recorded full-runtime dense-read evidence. It models resident
dense weights plus strict global LRU under total-RAM budgets of 8, 12, 16, 24,
32, and 48 GiB. The simulation reserves dense/runtime components before
assigning the remainder to expert payload cache capacity. It does not claim
latency, throughput, physical I/O, allocator behavior, or concurrent safety.

Recorded results show modeled total logical-read reduction of 40.43% at 8 GiB
and 56.90% at 16 GiB for resident dense, compared with 16.31% and 21.36% for
streamed dense under the same total-RAM accounting. The 8 GiB resident-dense
case has no simulated expert hits because only 1.981 GiB remains for experts;
the 16 GiB case retains 27.64% expert-byte hits. These results selected a
measurement-only runtime prototype for review. The follow-up M5.4-02 matrix
completed all 24 eligible paired rows at 8 and 16 GiB, preserving the reference
reader, frozen F32 checks, strict global LRU, and explicit total-RAM accounting.
The prototype passed correctness and budget gates, but single-run timing
remained directional, physical I/O/page-cache behavior was not measured, and
the runtime evidence did not establish end-to-end value. Its classification is
`prototype_insufficient_runtime_value`; the project remains frozen as a
research runtime.

No further resident-dense implementation is authorized by M5.4. Reopening the
candidate requires a new reviewed measurement protocol with repeatable
performance evidence and explicit working-set and I/O semantics.

### M6 - Hardware-aware performance runtime

Status: planned. M5 is closed as research evidence. M6 formally changes the
optimization objective from minimizing RAM to maximizing measured tokens/s
within user-supplied resource and quality constraints. The F32 runtime is
retained as `reference-f32-v1`; it is a correctness oracle, not the production
performance backend.

#### M6.0 - Freeze reference runtime

Status: complete.

Freeze the current validated artifacts, fixture hashes, router selections,
intermediate checkpoints, tolerances, and baseline performance report under a
single `reference-f32-v1` identity. Define backend-neutral execution and
comparison contracts without changing the default execution path.

Exit condition: a candidate backend can be compared layer-by-layer and its
first numerical divergence can be identified deterministically.

#### M6.1 - Hardware and model profiling

Add `colibri-lite doctor` and `colibri-lite profile-model`. The profiler must
measure rather than infer CPU kernel throughput, RAM bandwidth, SSD sequential
and expert-sized random reads, usable RAM/VRAM, GPU/backend availability, and
host/device transfer where available. Model profiling must record dense and
expert footprint, routing parameters, KV-cache requirements, and supported
precision candidates.

Exit condition: versioned, reproducible profile documents have bounded
measurement semantics and are sufficient inputs to a first planner.

#### M6.2 - First placement planner

Add `colibri-lite plan` accepting RAM budget, VRAM budget, context, and target
workload. It must enumerate and rank supported candidates with estimated
tokens/s, resource use, disk bytes/token, startup cost, and quality risk.
Initial cost models may be analytical but may not hard-code hardware results.

Exit condition: every recommendation is reproducible from profile inputs,
within explicit budgets, and states its confidence and unsupported assumptions.

#### M6.3 - Native quantized vertical slice

Implement and validate one Qwen3-MoE layer only. It may use an optimized CPU
backend for directly consumed quantized expert weights while retaining F32
router, norms, sensitive operations, and an independently executable F32
reference path. Quantization and backend selection require an ADR and must
validate English and Thai fixtures, exact safe-margin router selections, layer
checkpoints, logit drift, cold/warm throughput, RAM, and physical I/O.

Stop condition: do not extend to 48 layers unless the slice directly consumes
quantized weights, has repeatable material end-to-end benefit, and passes the
defined correctness/quality gates without hidden full-expert F32 expansion.

Status: the original group-128 INT8 candidate stopped at the M6.3-06 NO-GO
review. Its result remains authoritative; it must not be reopened or promoted.
The reviewed re-entry work is defined separately in M6.3-R1 below.

#### M6.3-R1 - Measurable re-entry and candidate admission

Before creating another quantized candidate, validate a release-process
measurement harness on `reference-f32-v1`. It must record process working-set
and private-byte peaks, logical artifact reads, and process-correlated physical
storage evidence with explicit filesystem-cache semantics. Unavailable physical
I/O remains `not_measured`, never zero.

On Windows, a fresh conversion may warm the filesystem cache before the
candidate process starts. Candidate characterization therefore uses the
reviewed ADR 0067 two-phase protocol: hash and freeze the fresh artifact,
cross a recorded reboot boundary, perform metadata-only identity validation,
and consume a one-shot launch authorization before ETW. The candidate's full
verification read must occur inside the correlated process. Logical reads,
post-result cache eviction, unbuffered access, or authorization reuse cannot
substitute for the physical-I/O gate.

Only after that telemetry gate passes may a deterministic study select exactly
one candidate layout. The candidate must directly consume packed values and
scales without whole-expert F32 expansion, preserve F32 router/norm/routing
weight/residual/activation/accumulation paths, and use a candidate-specific
numerical admission envelope proposed before final fixture execution. It must
pass exact safe-margin router checks, bilingual deterministic fixtures,
stage-level first-divergence comparison, fixed top-20/greedy/multi-token
agreement, and repeated execution.

The selected candidate must then run in paired F32/candidate release processes
with working-set, private-byte, logical-I/O, physical-I/O, cache, TTFT, and
throughput evidence. A missing telemetry collector invalidates the condition.
This re-entry may establish only a Layer-0 slice result; it must not extrapolate
payload reduction or local speed into a full-model benefit.

Exit condition: a separately reviewed record either rejects all layouts or
admits one measured Layer-0 candidate. The existing M6.3 full-runtime-benefit
condition remains in force. Therefore an admitted slice alone does not open
M6.4; doing so requires an explicit all-layer plan review. The detailed
protocol is `docs/reports/m6.3-r1-reentry-proposal.md`.

R1.1d prospectively amends the admission boundary without rewriting the
historical R1.1 `no_candidate_admitted` record. ADR 0069 admits the R1.1a
characterization winner `cpu-safe-rust-int8-group32-layer0-r1-1a` for R1.2
only, using the unchanged pre-registered `0.05` held-in fixed-logit envelope.
R1.2 may validate that single candidate on the held-out bilingual fixtures but
may not reselect a layout. ADR 0071 closes that held-out gate with `R1.2 = GO`:
both `short_english` and `short_thai` preserve exact prompt top-20, greedy,
Layer-0/24/47 safe-margin router IDs, frozen two-token greedy sequences, and
repeatability while staying below the locked effective prompt-logit envelope.
ADR 0077 closes R1.3 with 20/20 valid paired release-process measurements.
The candidate achieves a 5/5 0.8919485285% logical-byte reduction, but no
repeatable TTFT, prefill, decode, working-set, private-byte, or physical-I/O
win; median timed wall is worse in both cold and warm conditions. ADR 0078
therefore closes R1.4 `NO-GO for promotion` of the current direct safe-Rust
group-32 Layer-0 candidate. R1.2 quality remains valid, but M6.4 stays blocked
and no all-layer rollout is authorized. Any future re-entry requires a new
pre-registered runtime hypothesis and separate review.

#### M6.3-R2.0 - Expert compute bottleneck localization

ADR 0079 opens a diagnostic-only re-entry to explain the R1.3 slowdown before
any new optimization is selected. The unchanged F32 and admitted group-32
Layer-0 paths are measured on both R1.2 held-out fixtures in two views:
`compute_only_preloaded` isolates arithmetic/orchestration, while
`load_plus_compute` adds software load/parse/decode with the existing F32 cache
budget and no candidate cache. OS filesystem cache is observed, not flushed,
and R1.3 remains authoritative for physical-I/O claims.

R2.0 first freezes reference-only expert inputs/router selections, then freezes
instrumented source/binary/timer identities before candidate timing. Five fresh
paired processes per fixture/view produce 40 samples total. Measurement-only
instrumentation must prove byte-identical outputs, bounded observer effect, and
timer interpretability. Classification is prospectively limited to
`packed_projection_compute_bound`, `load_decode_bound`,
`routing_accumulation_bound`, or `mixed_or_distributed`.

R2.0 may authorize only an R2.1 hypothesis design. It does not authorize SIMD,
FFI/native kernels, cache-policy changes, new quantization, all-layer rollout,
or M6.4 implementation.

R2.0 is now closed. Reference-only Layer-0 fixtures and execution identities
were frozen before candidate timing, final observer controls passed on the exact
measurement binary, and 40/40 localization samples completed without automatic
retry. ADRs 0080-0085 record measurement-method and validation-only corrections;
none changed the immutable sample set or ADR 0079 classification thresholds.

ADR 0086 classifies the result `packed_projection_compute_bound`. Candidate
compute-only expert time is slower than F32 in 5/5 English and 5/5 Thai pairs,
with median slowdowns +77.35% and +85.79%; packed gate+up+down projections
account for median 91.44% and 92.04% of candidate expert time. Conversely,
load-plus-compute candidate totals are faster in all pairs, showing that the
primary remaining group32 slowdown is packed projection compute rather than
load/decode. R2.1 hypothesis design is authorized, but optimization
implementation, all-layer rollout, and M6.4 remain blocked.

ADR 0087 completes `M6.3-R2.1-PR` by pre-registering one Layer-0 AVX2+FMA
native packed-projection hypothesis. Rust retains routing, loading, activation,
accumulation, backend selection, and scalar fallback; native code may replace
only gate/up/down group32 projection arithmetic on the unchanged artifact. The
milestone explicitly allows one isolated reviewed FFI boundary while keeping
the workspace unsafe lint enabled globally, and freezes a <=1 MiB scratch cap,
no persistent prepack, and no complete F32 weight materialization.

R2.1b quality must pass before official performance timing: preserve the R1.2
held-out sequence/router/logit/repeatability gates and additionally bound
native-vs-scalar drift at <=0.001 Layer-0 routed output and <=0.002 prompt
logits. Only then may R2.1c run 72 fresh process samples: both fixtures, both
R2.0 views, and all six balanced permutations of native/scalar/F32. The frozen
performance gates require material scalar speedup plus near-F32 compute parity
while retaining the group32 load advantage.

R2.1a is complete. The feature-gated AVX2+FMA C kernel consumes the unchanged
group32 values/scales directly, while Rust keeps explicit backend selection and
scalar fallback. ADR 0088 accepts the single isolated FFI boundary and build-only
`cc` dependency. Native scratch and persistent prepack are both zero bytes, and
complete F32 weight materializations remain zero. Default runtime behavior is
unchanged and no official performance timing has been run.

R2.1b is now closed `GO`. ADR 0089 records that both held-out fixtures preserve
exact generated sequences, prompt top-20/argmax, Layer-0/24/47 router IDs, and
repeatability. Native-vs-scalar Layer-0 max-abs is `2.3841858e-7` for both
fixtures; prompt-logit max-abs is `3.0517578e-5` English and `2.5749207e-5`
Thai, within the frozen `0.001` and `0.002` limits. This is quality evidence,
not a performance claim.

R2.1c/R2.1d are now closed `GO`. ADR 0090 records 72/72 fresh process samples
with no automatic retry. Native beats scalar and F32 in every triplet; median
scalar/native speedup is `3.6810x` English / `3.3597x` Thai compute-only and
`2.0800x` / `2.0631x` load+compute. Median native/F32 ratios are `0.5024` /
`0.4497` compute-only and `0.0633` / `0.0605` load+compute. All resource guards
remain satisfied.

ADR 0091 now completes `M6.3-R2.2-PR`. R2.2 is frozen as a three-sentinel
proof on Layers 0/24/47 with all other layers canonical F32. The exact R2.1
group32 quantization grammar, AVX2+FMA kernel, FFI boundary, scalar fallback,
and resource limits are reused unchanged. Quality must pass before a 108-sample
compute-only native/scalar/F32 matrix is allowed.

R2.2a is now closed `GO`. The frozen converter reproduced the admitted Layer-0
artifact byte-for-byte, created independently identified Layer-24/47 group32
artifacts, and the Rust layer-bound reader verified all three complete hashes,
boundary expert reads, and packed working-set accounting. Cross-layer artifact
use is explicitly rejected; the kernel, unsafe boundary, and default runtime
remain unchanged.

R2.2b is now closed `NO-GO` by ADR 0095. The valid corrected durable execution
fails `short_thai` exact prompt top-20 ordering on the scalar group32 run after
exact generated IDs already matched. The top-20 token set is unchanged, but two
adjacent rank pairs swap. Because scalar fails before native validation, the
current localization target is the frozen group32 representation/placement
across Layers 0/24/47 rather than the AVX2 kernel.

ADRs 0096-0097 now pre-register and close `M6.3-R2.2-D1`. The frozen eight-state
scalar-only `short_thai` matrix shows Layer 47 alone is sufficient for the
rank-18/19 swap, while Layers 0+24 interact to cross both observed ranking
boundaries despite each singleton preserving exact top-20 order. Local
scalar/F32 routed max-abs is `8.7054e-4` at Layer 0, `1.1682e-2` at Layer 24,
and `6.8835e-2` at Layer 47. Router guards and argmax remain exact throughout.

ADR 0098 now pre-registers `M6.3-R2.2-D2` with contract SHA-256
`ea779da7bb4466c59a478c095ee0342c6932c482426e3331bf472c8296c31d30`.
The scalar characterization candidates are frozen at group32/group16/group8
with a deterministic coarsest-passing rule and F32 fallback. Layer 24 is tested
under canonical and Layer0-group32 prefixes; Layer 47 under canonical and the
original Layer0+24-group32 failing prefix. No timing or quality-pass claim is
in scope.

`M6.3-R2.2-D2a` is now closed. Scalar characterization-only group16/group8
support is validated, the generic converter is frozen, and all four new
Layer24/47 artifact identities are frozen in manifest SHA-256
`bd54c87ae20826086a9f8e6b1e06011bb4f1c65fa973494804c3fcb6b79e0a46`.

`M6.3-R2.2-D2b` is now closed by ADR 0099. No grouped-int8 candidate among
32/16/8 meets the frozen `0.001` local budget in all contexts. The deterministic
fallback keeps Layers 24 and 47 canonical F32, so the resulting sentinel policy
collapses to the already validated Layer-0 group32 slice rather than a new
multi-layer candidate.

ADR 0100 now pre-registers `M6.3-R2.2-D3` with contract SHA-256
`0faa6aa9200346aad94b0c69d6c6ed353682bf53410b258c38bfb61224f63ceb`.
It evaluates all seven non-empty group32 projection subsets of `gate/up/down`
at Layers 24/47 under the unchanged D2 contexts. Omitted projections are F32,
target variants are not propagated, and the fixed selection rule chooses the
first all-context passing variant in packed-count-first order.

`M6.3-R2.2-D3a` is now closed by ADR 0101. Layer 24 selects only `down`
as group32 while `gate/up` remain F32; Layer 47 selects canonical F32 because
no packed subset meets the frozen local budget. The proposed three-sentinel
representation models `31.944444%` fewer bytes than all-F32, but this is not a
runtime or performance claim.

ADR 0102 now pre-registers `M6.3-R2.2-D4` held-out quality re-entry with
contract SHA-256
`5f2b27b623a0456ad15e14b542816e4d5ad17bf32dacd91099184adfc86fb4fe`.
The exact policy is Layer0 all-group32 + Layer24 down-group32/F32 gate-up +
Layer47 F32. This quality seam may decode a complete F32 expert and therefore
cannot support resource/I/O/performance claims.

The exact next task is `M6.3-R2.2-D4a`: implement/freeze the hybrid quality
harness and execute it once. The historical R2.2c 108-sample contract remains
blocked/inapplicable to this new candidate; a D4 PASS can authorize only D5
hybrid artifact/layout and performance-contract design.

#### M6.4 - Full hardware-aware runtime

Only after M6.3-R1 admits a candidate *and* a dedicated all-layer plan review
authorizes implementation, extend the validated plan to all 48 layers with
tiered placement, reserved per-layer capacity, a global hot-expert pool,
bounded positional reads and hit-first overlap where measured, runtime
telemetry, and plan re-evaluation. The all-layer runtime must independently
prove material end-to-end benefit within its quality, working-set, and physical
I/O budgets; no slice result is a substitute.

#### M6.5 - Product surface

Only after the full runtime meets the approved interactive threshold, add
tokenizer/chat integration and a minimal product surface. Serving is not
authorized until a separate post-M6 scope review.

## Deferred pending an explicit M6 decision

- GPU backends other than the single backend justified by M6 evidence.
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
