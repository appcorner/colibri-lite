# colibri-lite-rs Work Log

Append one entry after each meaningful work session. Record commands and test
results as evidence; do not rewrite earlier entries when later work changes the
project.

## Entry format

```text
Date:
Starting task:
Completed tasks:
Commands executed:
Tests:
Known issues:
Next task:
Commit:
```

## 2026-07-14 - Project control documents

Date: 2026-07-14

Starting task: Add backlog, work-log, and milestone branch conventions after
reviewing `AGENTS.md`.

Completed tasks: Created `docs/backlog.md` and `docs/work-log.md`; documented
the five milestone branches in the implementation plan and task tracker; added
the new document links to the README.

Commands executed: Read `AGENTS.md`, the implementation plan, task tracker,
README, and Git status; ran `git diff --check` and inspected the final status.

Tests: Documentation-only change. `git diff --check` passed; Cargo tests were
not required for this session.

Known issues: None in this documentation change.

Next task: M0.2-01 - add `crates/clr-core/src/error.rs` on
`milestone/m0-core-contracts`.

Commit: Pending.

## 2026-07-14 - M0.2 core value contracts

Date: 2026-07-14

Starting task: M0.2-01 - add the dependency-free runtime error contract on
`milestone/m0-core-contracts`.

Completed tasks: M0.2-01 through M0.2-30. Added structured runtime errors,
dense data-type metadata, checked tensor shapes, validated generic model
configuration, runtime module integration, crate-root re-exports, and ADR 0001.

Commands executed: `cargo fmt --all --check`, targeted `clr-core` error, dtype,
shape, and config tests, `cargo check --workspace`, `cargo test --workspace`,
`cargo clippy --workspace --all-targets -- -D warnings`,
`cargo run -p clr-cli`, source scope search, Git status/diff review, and the
focused Git commit.

Tests: All 15 `clr-core` unit tests passed; all workspace and doc-test targets
passed with zero failures. Clippy passed with warnings denied. CLI output ended
with `status: bootstrap ready`.

Known issues: None. `F16` and `BF16` remain metadata-only by design; M1 compute
support is limited to `F32`.

Next task: M0.3-01 - add a pinned Python reference environment file.

Commit: `bb7e6f2` (`feat(core): add validated runtime value contracts`).

## 2026-07-14 - M0.3 deterministic oracle fixture

Date: 2026-07-14

Starting task: M0.3-01 - pin the Python reference environment.

Completed tasks: M0.3-01 through M0.3-14. Pinned the reference environment and
Qwen3-MoE revision, created deterministic synthetic weights and a two-layer
fixture, recorded 23 oracle checkpoints and exact selected experts, defined
tolerances, generated SHA-256 evidence, and verified byte-for-byte regeneration
in an isolated venv.

Commands executed: Python/package version inspection; read-only Hugging Face
metadata lookup; fixture `generate` and `verify`; Safetensors inventory and
expert-ID inspection; isolated venv verification; all standard Cargo
verification commands; Git diff/status review; and the focused Git commit.

Tests: Fixture regeneration matched every committed SHA-256 digest. The
isolated venv verification passed. All 15 Rust unit tests and all workspace/doc
test targets passed with zero failures. Clippy passed with warnings denied. CLI
output ended with `status: bootstrap ready`.

Known issues: Python and Git HTTPS could not validate the local certificate
chain for the Hugging Face API; Windows PowerShell HTTPS returned and verified
the exact revision/license metadata. Fixture generation and verification are
offline. The requirements lock pins versions but does not yet pin wheel hashes.

Next task: Create the M0 milestone report, then begin M1.1-01 on
`milestone/m1-tiny-qwen-correctness`.

Commit: `18d905c` (`test(oracle): freeze tiny qwen3-moe reference fixture`).

## 2026-07-14 - M1.1 dense F32 tensor correctness

Date: 2026-07-14

Starting task: M1.1-01 - define owned dense F32 tensor storage.

Completed tasks: M1.1-01 through M1.1-10. Added checked owned/borrowed tensor
contracts, contiguous row-major indexing, elementwise operations, reductions,
matrix-vector/matrix-matrix multiplication, stable softmax, `SiLU`, structured
shape/rank/non-finite errors, and ADR 0002.

Commands executed: Targeted tensor/error/operation tests; all standard Cargo
verification commands; source scope search for SIMD, FFI, unsafe, broadcasting,
strides, and parallelism; Git diff/status review; and the focused Git commit.

Tests: Five tensor/view tests, ten operation tests, and two error regression
tests passed. All 30 `clr-core` unit tests and all workspace/doc-test targets
passed with zero failures. Clippy passed with warnings denied. CLI output ended
with `status: bootstrap ready`.

Known issues: Kernels are intentionally scalar, contiguous, and F32-only. There
is no broadcasting, strided layout, batched matrix multiplication, or optimized
backend.

Next task: M1.2-01 - define Qwen3-specific configuration mapping in
`clr-qwen3-moe`.

Commit: `1ed5fb5` (`feat(core): add dense f32 tensor correctness path`).

## 2026-07-14 - M1.2 stopped on RoPE configuration conflict

Date: 2026-07-14

Starting task: M1.2-01 - define Qwen3-specific configuration mapping.

Completed tasks: None. A draft Qwen-specific config and block implementation
was started but is not accepted or committed.

Commands executed: Inspected the pinned Transformers 5.12.1 Qwen3-MoE source,
ran three draft config tests, inspected frozen fixture tensor names/shapes, and
checked the frozen `config.json` RoPE values.

Tests: Three draft config validation tests passed, but their fixture mapping is
not valid evidence because the draft used the wrong `rope_theta` value.

Known issues: Stop condition 3 in `AGENTS.md` applies. The frozen fixture uses
`rope_theta = 10000.0`, while the draft test assumed `1000000.0`. Transformers
uses `config.rope_parameters["rope_theta"]` directly for inverse frequencies,
so continuing would cause deterministic RoPE and attention divergence.

Next task: Review and approve correcting the M1.2 config mapping to the frozen
fixture value `10000.0`, then rerun M1.2-01 evidence before continuing.

Commit: None; unverified M1.2 work remains uncommitted.

## 2026-07-14 - M1.2 stopped on Rust toolchain corruption

Date: 2026-07-14

Starting task: Resume M1.2 after approval to map `rope_theta` from the frozen
fixture and add RoPE/attention oracle tests.

Completed tasks: Extended the offline fixture generator to pin
`rope_theta = 10000.0`, emit a generated Rust config constant, capture post-RoPE
query/key tensors, and verify the extended fixture byte-for-byte. These changes
remain uncommitted because Rust verification is unavailable.

Commands executed: Fixture generation/verification, frozen config and tensor
offset inspection, `cargo check -p clr-qwen3-moe`, `rustc --version`,
`rustup show`, `rustup which rustc`, installed-component inspection, and
`rustup component add rustc --toolchain stable-x86_64-pc-windows-msvc`.

Tests: Python fixture generation and byte-for-byte regeneration passed. Rust
tests could not run because the active toolchain cannot execute `rustc.exe`.

Known issues: Rustup lists the `rustc` component as installed and up to date,
but `rustup which rustc` reports it missing and `rustc --version` reports that
the binary is not applicable to the active stable MSVC toolchain. Repair now
requires a force reinstall or uninstall/install of the external toolchain,
which triggers Stop Condition 12.

Next task: Obtain approval for a stable MSVC Rust toolchain reinstall, verify
`rustc --version`, then continue the requested M1.2 tests before changing the
blocked task status.

Commit: None; M1.2 remains blocked and uncommitted.

## 2026-07-14 - M1.2 sparse decoder block resolved

Date: 2026-07-14

Starting task: Resume M1.2 after the reviewed RoPE configuration conflict and
Rust toolchain repair.

Completed tasks: M1.2-01 through M1.2-12. Added config-driven Qwen3-MoE
mapping, RMSNorm, default RoPE, causal grouped-query attention, deterministic
top-k routing, optional selected-weight normalization, gated expert MLP,
weighted expert combination, full sparse block checkpoints, first-stage
diagnostics, and ADR 0003.

Commands executed: Repaired-toolchain checks; Python fixture generation and
verification; targeted config, RoPE, attention, router, expert, and block tests;
all standard Cargo verification commands; Git whitespace/status review; and the
focused Git commit.

Tests: The mapping test reads generated constants derived from frozen
`config.json` and confirms `rope_theta = 10000.0`. A two-theta regression test
confirms RoPE uses config values. Query/key RoPE, attention, router, experts,
and every block checkpoint match the frozen oracle. Expert IDs match exactly.
All 42 Rust tests passed with zero failures; Clippy passed with warnings denied;
fixture byte-for-byte regeneration and the CLI smoke test passed.

Known issues: The fixture path is batch-one, F32-only, causal, and starts at
position zero. Padding, KV cache, arbitrary position offsets, sliding-window
attention, and optimized kernels remain outside M1.2.

Next task: M1.3-01 - implement embedding lookup for the full tiny decoder.

Commit: `e951ab1` (`feat(qwen3): match frozen sparse decoder block`).

## 2026-07-14 - M1.3 full tiny decoder correctness

Date: 2026-07-14

Starting task: M1.3-01 - implement embedding lookup.

Completed tasks: M1.3-01 through M1.3-05. Added checked embedding lookup,
two-layer decoder composition, final RMSNorm, LM head, full hidden-stage and
logit oracle comparisons, deterministic repeated runs, and ADR 0004.

Commands executed: Targeted embedding/model tests; layer-stage diagnostic tests;
all standard Cargo verification commands; Git status review; and the focused
Git commit.

Tests: Five M1.3 model tests passed. Both layers match every frozen stage and
expert ID; final norm and `[4, 64]` logits match tolerance; repeated complete
forwards are exactly equal. All 47 Rust tests passed with zero failures. Clippy
passed with warnings denied and CLI output ended with `status: bootstrap ready`.

Known issues: Rust model output retains raw block hidden states, while
Transformers' final `hidden_states` entry is post-final-normalization. Tests map
these semantic checkpoints explicitly. Tokenizer, generation, KV cache,
artifact loading, and optimized kernels remain outside M1.

Next task: M1.3-06 - record the first correctness/milestone report, then begin
M2.1-01 on `milestone/m2-expert-residency`.

Commit: `57add35` (`feat(qwen3): match frozen full tiny decoder`).

## 2026-07-14 - M2.1 portable artifact reader

Date: 2026-07-14

Starting task: M2.1-01 - define a versioned artifact manifest.

Completed tasks: M2.1-01 through M2.1-06. Added versioned tensor metadata,
little-endian/path/range/hash validation, dependency-free SHA-256, portable
seek/read-exact access, structured storage errors, Windows handle-lifetime
coverage, and ADR 0005.

Commands executed: Targeted storage tests and Clippy; all standard Cargo
verification commands; Git status review; and the focused Git commit.

Tests: Six storage tests passed, including published SHA-256 vectors,
version/endian/duplicate/path/length/overflow/overlap rejection, exact range
reads, unknown tensors, truncation, corruption, and immediate Windows file
deletion after read. All 53 workspace tests passed with zero failures; Clippy
passed with warnings denied and CLI output ended with `status: bootstrap ready`.

Known issues: Manifest serialization is deferred; the M4 converter must
construct this validated Rust contract. Files are opened per tensor read and
payload hashes are verified before cache admission. Memory mapping remains
deferred to M2.3.

Next task: M2.2-01 - define `ExpertId` and a stable cache key.

Commit: `63ee620` (`feat(storage): add validated portable artifact reader`).

## 2026-07-14 - M2.2 expert cache through integration stop

Date: 2026-07-14

Starting task: M2.2-01 - define `ExpertId` and a stable cache key.

Completed tasks: M2.2-01 through M2.2-08. Added on-demand `ExpertStore`, stable
layer/expert keys, byte-budgeted deterministic LRU, Arc lease/pin semantics,
oversize/pinned admission errors, cache/I/O metrics, and ADR 0006.

Commands executed: Targeted expert-cache/store tests; all standard Cargo
verification commands; Git status review; and the focused Git commit.

Tests: Three expert tests pass for deterministic hit/miss/eviction order,
strict budget, live lease safety, oversized payloads, artifact-backed loading,
cache hits, unknown keys, and exact metrics. All 56 workspace tests passed with
zero failures; Clippy passed with warnings denied and CLI output ended with
`status: bootstrap ready`.

Known issues: M2.2-09 requires the public Qwen block/model weight ownership to
support expert payloads supplied by `ExpertStore` leases rather than only owned
`Tensor` fields. This is a public API redesign affecting `clr-qwen3-moe`,
`clr-storage`, fixture constructors, and future CLI composition, so Stop
Condition 2 requires review before implementation.

Next task: Review the expert-provider boundary for M2.2-09, then prove resident
and on-demand tiny-model outputs are identical.

Commit: `3a3a437` (`feat(storage): add byte-budgeted expert cache`).

## 2026-07-14 - M2.2 streaming equivalence

Date: 2026-07-14

Starting task: Resume M2.2-09 with the approved `StreamingQwen3MoeModel` and
packed per-expert F32 payload design.

Completed tasks: M2.2-09. Added a separate streaming model, config-derived
gate/up/down payload layout, shard-independent expert ranges, lease-scoped
decode/computation, shared resident/streaming expert MLP and routing combination,
full resident equivalence tests, and ADR 0007.

Commands executed: Targeted packed/streaming failure and equivalence tests; all
standard Cargo verification commands; Git diff/status review; and the focused
Git commit.

Tests: Exact packed F32 round-trip passed for all eight fixture experts.
Resident and streaming paths match every block stage, exact expert IDs, expert
outputs, block outputs, and final logits at existing M1 tolerances. A strict
two-expert budget produced 8 misses, 8 loads, 6 evictions, 0 hits, and exact
resident/peak/bytes-read metrics. Oversize, hash-invalid, and truncated payloads
failed before computation. All 59 workspace tests passed; Clippy passed with
warnings denied and CLI smoke output remained correct.

Known issues: Streaming currently decodes a leased expert payload into temporary
F32 vectors for computation. This preserves cache residency and correctness but
is not optimized. Memory mapping remains unevaluated until M2.3 evidence.

Next task: M2.3-01 - benchmark portable artifact access before considering
memory mapping.

Commit: `441b491` (`feat(qwen3): add storage-aware streaming model`).

## 2026-07-14 - M2.3 portable baseline and mapping stop

Date: 2026-07-14

Starting task: M2.3-01 - benchmark portable access before adding mapping.

Completed tasks: M2.3-01. Added a dependency-free release benchmark for the
complete portable open/seek/read-exact/SHA-256 path and recorded hardware,
software, workload, and result evidence.

Commands executed: Release portable-reader benchmark; Windows CPU/RAM/OS/disk
inventory; Rust version and Git commit inspection; standard Cargo verification
for the benchmark target; and Git diff/status review.

Tests: The release benchmark read and verified 200 MiB total at 129.367 MiB/s
for 1 MiB payloads. Existing portable reader, cache, streaming equivalence, and
workspace correctness tests remain passing.

Known issues: M2.3-02 requires a read-only memory-mapping implementation and
dedicated Windows mapping/file lifetime invariants. This introduces a new
unsafe or externally-audited mapping boundary, triggering Stop Condition 5
before implementation.

Next task: Review whether benchmark evidence justifies implementing a read-only
mapping backend; if approved, define the smallest isolated boundary and compare
it against this exact workload.

Commit: `5ffd40c` (`bench(storage): record portable reader baseline`).

## 2026-07-14 - M2 closure with portable backend

Date: 2026-07-14

Starting task: Resolve the reviewed M2.3 mapping decision and close M2.

Completed tasks: M2.3-02 through M2.3-06. Approved portable access as the M2
production path, confirmed no mapping dependency/unsafe code, documented the
baseline and copy behavior, added measurable reconsideration criteria, moved
mapping to the backlog, created ADR 0008, and produced the M2 milestone report.

Commands executed: Documentation scope review, all standard Cargo verification
commands, Git diff/status review, and the focused M2 closure commit.

Tests: All 59 workspace tests passed with zero failures. Clippy passed with
warnings denied, CLI smoke output remained correct, and the recorded portable
release benchmark remains 129.367 MiB/s for the verified 200 MiB workload.

Known issues: Streaming expert payloads are decoded/copied into temporary F32
vectors. Mapping is intentionally deferred until profiling proves artifact I/O
or copy cost is material under a representative full-model decode workload.

Next task: Merge the reported M2 branch, create
`milestone/m3-generation`, and begin M3-01 greedy token-ID decoding.

Commit: `fdcfc13` (`docs: close M2 with portable backend evidence`).

## 2026-07-14 - M3 sampling through KV-cache stop

Date: 2026-07-14

Starting task: M3-01 - implement greedy token-ID decoding.

Completed tasks: M3-01 through M3-03. Added deterministic greedy argmax with
lower-ID ties, frozen-oracle first-token coverage, documented SplitMix64,
temperature sampling, recomputing generation methods, validation tests, and ADR
0009.

Commands executed: Frozen oracle argmax inspection; targeted generation tests;
all standard Cargo verification commands; Git diff/status review; and the
focused Git commit.

Tests: Six generation tests pass for oracle token 10, tie/rank/empty/non-finite
behavior, repeated greedy output, pinned SplitMix64 outputs, same-seed sampling,
and invalid temperature. All 65 workspace tests passed with zero failures;
Clippy passed with warnings denied and CLI smoke output remained correct.

Known issues: Generation currently recomputes the complete sequence. M3-04 and
M3-05 require persistent per-layer KV state, position-aware single-token
attention, explicit byte accounting, context limits, and shared resident/
streaming session ownership. This is a public API/attention-state redesign
under Stop Condition 2.

Next task: Review a separate generation-session/KV-cache boundary that preserves
the resident and streaming full-forward oracle APIs.

Commit: `a141551` (`feat(qwen3): add deterministic token sampling`).

## 2026-07-14 - M3 KV-cache contract review

Date: 2026-07-14

Starting task: M3-04 - define KV-cache layout, context limit, and byte
accounting.

Completed task: M3-04. Review accepted the generation-session boundary in ADR
0010. Added a fixed-capacity, per-layer contiguous F32 key/value cache with
checked byte accounting, transactional append validation, an explicit context
limit error, and fixed-allocation/repeated-release tests.

Commands executed: repository recovery inspection, `cargo fmt --all`,
`cargo test -p clr-core error::tests`, and
`cargo test -p clr-qwen3-moe cache::tests`.

Tests: Two core error tests and four KV-cache tests passed with zero failures.
The recovered partial workspace also passed all 69 tests before this task was
closed; full milestone verification remains deferred until M3-10.

Known issues: The cache contract is not yet connected to model execution, so
the private append/view paths produce dead-code warnings in a normal check.
M3-05 will connect them through prefill.

Next task: M3-05 - implement prefill.

## 2026-07-14 - M3 cached prefill and decode

Date: 2026-07-14

Starting task: M3-05 - implement prefill.

Completed tasks: M3-05 and M3-06. Added a shared position-aware cached
attention/block path and `GenerationSession` backends for resident and
on-demand expert models. Prefill validates capacity and token IDs before
mutation. Greedy and temperature decode use cached logits; sampling RNG state
is committed only after successful execution.

Commands executed: `cargo fmt --all`, focused prefill tests, focused decode
tests, `cargo test -p clr-qwen3-moe`, and
`cargo clippy -p clr-qwen3-moe --all-targets -- -D warnings`.

Tests: Three prefill tests passed for resident-oracle logits, exact expert IDs,
streaming equivalence, and pre-mutation validation. Five decode-related tests
passed for recomputing equivalence, seeded sampling, resident/streaming
equivalence, context failure, and prefill requirements.
All 38 `clr-qwen3-moe` tests and doc tests passed; crate Clippy passed with
warnings denied.

Known issues: The CLI does not yet expose token-ID generation.

Next task: M3-07 - add a CLI command accepting token IDs directly.

## 2026-07-14 - M3 token-ID CLI

Date: 2026-07-14

Starting task: M3-07 - add a CLI command accepting token IDs directly.

Completed task: M3-07. Exposed the versioned frozen tiny fixture through a
narrow constructor and added a dependency-free `generate` command with direct
comma-separated token IDs, a required new-token count, optional temperature,
and an optional seed. The no-argument bootstrap smoke output remains unchanged.

Commands executed: `cargo fmt --all --check`, `cargo test -p clr-cli`,
`cargo clippy -p clr-cli --all-targets -- -D warnings`, and
`cargo run -p clr-cli -- generate --tokens 1,7,3,12 --max-new-tokens 4`.

Tests: All three CLI tests passed. The command generated `10,11,54,11`, emitted
the complete sequence, and reported a 1,024-byte cache at 8/8 initialized
positions. CLI Clippy passed with warnings denied.

Known issues: Milestone-wide repeatability and fixed-allocation stress evidence
remain to be recorded.

Next task: M3-08 - test reproducible token sequences.

## 2026-07-14 - M3 reproducibility and bounded memory

Date: 2026-07-14

Starting task: M3-08 - test reproducible token sequences.

Completed tasks: M3-08 and M3-09. Froze the oracle-prompt greedy and seeded
temperature sequences, repeated sampled generation from independent sessions,
filled the complete 32-token context, and checked KV allocation capacities at
every decode step. Repeated streaming decode also checked the expert budget.

Commands executed: the temperature CLI run,
`cargo test -p clr-qwen3-moe reproducible`, and
`cargo test -p clr-qwen3-moe repeated`.

Tests: The oracle prompt `1,5,7,2` reproducibly generates greedy
`10,11,10,11` and temperature-0.8/seed-42 `47,10,18,22`. Four repeated-path
tests passed. The full 32-position cache stayed exactly 4,096 bytes with fixed
allocation capacities; the streaming expert cache stayed within its 9,216-byte
budget.

Known issues: Full standard verification and the M3 correctness report remain.

Next task: M3-10 - record a tiny-generation correctness report.

## 2026-07-14 - M3 closure

Date: 2026-07-14

Starting task: M3-10 - record a tiny-generation correctness report.

Completed task: M3-10 and milestone M3. Added the generation correctness,
memory, CLI, scope, and limitation evidence in `docs/reports/m3.md`.

Commands executed: `cargo fmt --all --check`, `cargo check --workspace`,
`cargo test --workspace`,
`cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo run -p clr-cli`.

Tests: All 83 workspace tests passed with zero failures; doc tests passed.
Workspace Clippy passed with zero warnings. The required CLI smoke reported
`x86_64-windows` and `status: bootstrap ready`.

Known issues: Full-size Qwen3-30B-A3B artifacts, provenance, tensor mapping,
and correctness are not implemented and remain M4 work.

Next task: Create `milestone/m4-full-qwen3` from the reviewed M3 closure and
begin M4.1-01 by pinning the exact full-model ID and revision.

## 2026-07-14 - M4.1-01 full-model source contract

Date: 2026-07-14

Starting task: M4.1-01 - pin the exact Qwen3-30B-A3B model ID and revision.

Completed task: M4.1-01. Merged the approved M3 branch into `main` with merge
commit `dfa800b`, tagged it `m3-generation`, and created
`milestone/m4-full-qwen3`. Pinned `Qwen/Qwen3-30B-A3B` at immutable revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. Added source manifest v1 and ADR
0011 with Apache-2.0 license, architecture/config, tokenizer, 16-shard
inventory, source hashes, and disk requirements.

Commands executed: all five standard M3 verification commands; `main...M3`
scope, test, dependency, unsafe, deferred-feature, and public-API diff review;
official Hugging Face immutable-revision API queries; in-memory SHA-256 hashing
of non-weight source files; and Safetensors index inspection.

Evidence: All 83 workspace tests passed; Clippy passed with warnings denied;
the CLI smoke passed. The source tree contains 26 files totaling
61,084,187,391 bytes (56.889 GiB), including 16 weight shards totaling
61,066,575,648 bytes. The index maps 18,867 tensors. No weight shard was
downloaded.

Known issue and stop condition: upstream `head_dim = 128`, while the current
public runtime derives `2048 / 32 = 64`. M4 configuration mapping would be
incorrect without a reviewed public contract change. BF16 source storage and
the 40,960 model versus 131,072 tokenizer context limits also require explicit
mapping decisions.

Next task: M4.1-02 - document upstream license and artifact provenance, only
after review of ADR 0011 and the `head_dim` contract conflict.

## 2026-07-14 - M4.1-02 provenance approval

Date: 2026-07-14

Starting task: M4.1-02 - document upstream license and artifact provenance.

Completed task: M4.1-02. ADR 0011 and source manifest v1 were approved. The
immutable model revision, Apache-2.0 license, complete source inventory, file
hashes, and download-space requirements are now the accepted provenance
contract.

Known issue: Configuration mapping remains gated on the separately reviewed
explicit `head_dim` contract in ADR 0012.

Next: Implement and verify ADR 0012 before beginning M4.1-03.

## 2026-07-14 - Explicit attention head dimension contract

Date: 2026-07-14

Starting work: Implement the approved public-contract prerequisite for
M4.1-03.

Completed: Added explicit generic head dimension and checked query/KV
projection widths, removed the hidden/query-width equality assumption, updated
Qwen attention/output projection and KV-cache construction, and recorded ADR
0012. The tiny fixture now states `head_dim = 4` explicitly without changing
weights or checkpoints.

Commands executed: focused core/Qwen config tests; exact frozen M1 logits and
M3 greedy/seeded regression tests; `python python\reference\generate_fixture.py
verify`; and all five standard Cargo verification commands.

Evidence: All 87 workspace tests passed; Clippy passed with warnings denied;
CLI smoke passed. Fixture regeneration matched every committed artifact hash.
Pinned dimensions `2048/32/4/128` are accepted with query width 4,096 and KV
width 512. Zero head dimension and query/KV overflow cases return structured
errors. Frozen numerical outputs and generation sequences are unchanged.

Known issues: BF16 remains source metadata only. Model maximum positions,
tokenizer limit, and session capacity remain intentionally separate.

Next task: M4.1-03 - map required Hugging Face configuration fields.

## 2026-07-14 - M4.1-03 pinned configuration mapping

Date: 2026-07-14

Starting task: M4.1-03 - map required Hugging Face configuration fields.

Completed task: M4.1-03. Added a typed Qwen3-MoE source configuration and
validated source-to-F32-runtime mapping for the immutable Qwen3-30B-A3B
revision. `head_dim=128` maps directly; query/KV widths are 4,096/512. BF16 is
retained separately as source storage metadata. ADR 0013 records field
mappings, rejected unsupported features, and deliberately separate context
limits.

Commands executed: focused source-mapping tests and all five standard Cargo
verification commands.

Evidence: All 90 workspace tests passed; doc tests passed; Clippy passed with
warnings denied; CLI smoke passed. Unsupported architecture, dtype, activation,
attention bias/dropout, RoPE scaling, dense-only layers, sliding window, and
tied embeddings return structured errors. Frozen M1/M3 regressions remain
unchanged through the full workspace suite.

Known issues: Source JSON parsing is intentionally deferred. Tensor names and
shapes have not been mapped or validated, and no weight shard is downloaded.

Next task: M4.1-04 - map and validate required tensor names and shapes.

## 2026-07-14 - M4.1-04 pinned tensor inventory

Date: 2026-07-14

Starting task: M4.1-04 - map and validate required tensor names and shapes.

Completed task: M4.1-04. Added metadata-only Qwen3-MoE tensor-role mapping and
validation for the complete pinned Safetensors inventory. The validator uses
checked config-derived shapes, explicit query/KV widths, complete layer and
expert coverage, separate untied embedding/LM-head requirements, and
structured errors for all required invalid inventory categories. ADR 0014 and
tensor inventory summary v1 record the frozen grammar and source evidence.

Commands executed: bounded upstream Safetensors index/header inspection with
no tensor payload reads; inventory evidence reconciliation; focused inventory
tests; `cargo fmt --all --check`, `cargo check --workspace`,
`cargo test --workspace`,
`cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo run -p clr-cli`.

Evidence: All 18,867 pinned index tensors are classified and shape-validated:
3 top-level tensors, 48 copies of each of 9 per-layer roles, and 6,144 copies
of each expert gate/up/down role. Unknown and unsupported counts are zero. The
16 shard headers contain the same 18,867 tensors in 2,330,272 inspected bytes;
no tensor payload byte was downloaded. All 102 workspace tests passed with
zero failures or ignored tests; 12 tensor-inventory tests cover the full valid
inventory and required failure modes. Clippy and CLI smoke passed.

Known issues: Source BF16 remains storage metadata only. No weights have been
downloaded or converted, and no tokenizer parsing or optimized execution path
was introduced.

Next task: M4.1-05 - convert dense tensors for resident access.

## 2026-07-14 - M4.1-05 dense resident artifact conversion

Date: 2026-07-14

Starting task: M4.1-05 - convert dense tensors for resident access.

Completed task: M4.1-05. Added safe, chunked BF16-to-F32 conversion with full
source-shard hash verification before decode, Qwen role/shape validation,
preflight disk accounting, deterministic multi-range artifact manifests,
temporary output plus manifest-last atomic commit, and streaming exact F32
round-trip verification. ADR 0015, the 435-tensor source plan, report, and
compact evidence record the accepted design and real conversion results.

Real evidence: The initial three-norm slice verified all 1,087,928,584 bytes of
shard 16 and produced a 24,576-byte artifact. The complete conversion verified
61,066,575,648 source-shard bytes before decoding and converted all 435 dense
tensors into one 6,164,373,504-byte physical payload with independent logical
ranges. Maximum explicit working buffers were 327,680 bytes. Both slice and
complete conversions were independently repeated with identical payload and
manifest hashes; every F32 element matched direct BF16 decoding byte-for-byte.

Commands executed: immutable pinned index/shard downloads into `D:\tmp`;
source length/hash/header/index reconciliation; focused storage, Qwen, and plan
parser tests; two real slice runs; two complete release-profile conversion
runs; external artifact hash checks; and all five standard verification
commands.

Tests: All 114 workspace tests passed with zero failures or ignored tests, plus
the focused conversion-plan parser test. Failure coverage includes wrong hash,
truncation, wrong dtype, wrong pinned shape, invalid range, insufficient disk,
expert selection, incomplete full inventory, and incomplete-output cleanup.
Clippy passed with warnings denied and the CLI smoke passed.

Known issues: Source and converted multi-gigabyte files remain local and are
not committed. BF16 is source metadata/decoding only. Expert conversion,
tokenizer parsing, quantization, mmap, SIMD, FFI, and optimized kernels remain
unimplemented.

Next task: M4.1-06 - convert experts for independent on-demand access.

## 2026-07-15 - M4.1-06 independently accessible expert conversion

Date: 2026-07-15

Starting task: M4.1-06 - convert experts for independent on-demand access.

Completed task: M4.1-06. Added pinned expert source-plan parsing and validation,
safe chunked BF16-to-F32 conversion, one deterministic container per selected
layer, versioned arbitrary-shard expert mappings, exact preflight accounting,
complete source hash verification, manifest-last transaction cleanup, and
streaming exact round-trip verification. The existing `ArtifactReader`,
`ExpertStore`, packed gate/up/down layout, leases, views, and shared numerical
path remain unchanged. ADR 0016, the 18,432-tensor source plan, report, and
compact evidence record the design and real conversion results.

Real evidence: The vertical slice converted experts `0:0`, `0:127`, and
`47:127` across two layer containers. The complete conversion accounted for all
6,144 experts and 18,432 expert tensors, bringing cumulative pinned inventory
coverage to all 18,867 tensors. It wrote 115,964,116,992 F32 bytes across 48
2,415,919,104-byte layer containers plus a 3,130,926-byte manifest. Maximum
explicit buffers were 327,680 bytes. Loading expert `23:64` through the existing
store read exactly its 18,874,368-byte logical payload.

Determinism: Two independent complete runs produced byte-identical manifests
with SHA-256
`9c581c6c46ecf830e7d0dd0e380d26b17803784f009b37ef2657ae34d06b2939`
and identical ordered shard-set SHA-256
`b90d537f5c0c202b2bf5db0e74b8bf8b9ba9ea5378d788733d2bf9e11d36bf91`.
All 48 shard records and 6,144 expert records matched. External hashes of first,
middle, and last physical layer files matched both manifests.

Commands executed: source-plan count/order/range/hash audits; two real
vertical-slice runs; two complete release-profile conversion runs; independent
manifest, record, file-length, and representative physical hash comparisons;
focused expert conversion and plan parser tests; `cargo fmt --all --check`,
`cargo check --workspace`, `cargo test --workspace`,
`cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo run -p clr-cli`.

Tests: All 118 workspace tests passed with zero failures or ignored tests, plus
the focused conversion-plan parser test. Failure coverage includes corruption,
truncation, invalid offsets, wrong shard, wrong layer/expert identity, wrong
shape, incomplete inventory, duplicate experts, invalid projection order,
invalid chunk size, insufficient disk space, one-layer slices, and incomplete
output cleanup. Clippy passed with warnings denied and the CLI smoke passed.

Known issues: The 16 source shards and 232 GB of independently repeated expert
artifacts remain local evidence and are not committed. BF16 is decoded to F32;
quantization, BF16 kernels, mmap, async prefetch, SIMD, FFI, tokenizer parsing,
optimized kernels, and full-model generation remain unimplemented.

Next task: M4.1-07 - include tokenizer assets required for the first full-model
test.

## 2026-07-15 - M4.1-07 pinned offline tokenizer assets

Date: 2026-07-15

Starting task: M4.1-07 - include tokenizer assets required for the first
full-model test.

Completed task: M4.1-07. Added all four canonical Qwen3-30B-A3B tokenizer
assets from the approved immutable revision, a versioned artifact manifest, a
frozen seven-case reference fixture, and an offline-only verification adapter.
ADR 0017, the task report, and compact evidence record the byte-level BPE
contract, vocabulary and added-token metadata, special IDs, chat-template hash,
separate length concepts, provenance, and dependency decision.

Artifact evidence: `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and
`merges.txt` total 15,881,072 bytes. All sizes and SHA-256 hashes match ADR 0011
at revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. The tokenizer has
151,643 base entries, 151,387 merges, 26 added tokens, and 151,669 total entries
against the model's separate 151,936 vocabulary size. The 4,168-byte preserved
chat template hashes to
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.

Reference evidence: The existing Transformers 5.12.1/tokenizers 0.22.2 oracle
loaded only committed local files with Hugging Face offline mode forced. Exact
token IDs, decoded text, and round trips matched 7/7 cases covering English,
Thai, source code/indentation, whitespace/newlines, Unicode/emoji, special
tokens, and empty input.

Dependency decision: No Cargo or Python dependency, native library, unsafe
boundary, or public API was added. Rust inference remains token-ID based. A
production Rust Unicode-regex/BPE tokenizer and chat-template renderer remain
deferred for separate review; no chat framework, tool calling, or API was
implemented.

Commands executed: Pinned HTTPS downloads for four tokenizer files only;
source-contract size/hash reconciliation; tokenizer JSON/vocabulary/merge,
added-token, special-ID, limit, and chat-template inspection;
`python python\reference\verify_tokenizer.py`; `cargo fmt --all --check`,
`cargo check --workspace`, `cargo test --workspace`,
`cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo run -p clr-cli`.

Tests: Offline tokenizer verification passed all seven exact comparison cases.
All 118 workspace tests passed with zero failures or ignored tests. Clippy
passed with warnings denied and the CLI smoke reported `bootstrap ready`.

Known issues: Rust text tokenization and chat-template rendering are not
implemented. Model maximum positions (40,960), tokenizer-declared length
(131,072), and caller-configured runtime session capacity remain separate.

Next task: M4.1-08 - generate hashes and a reproducible conversion manifest.

## 2026-07-15 - M4.1-08 unified model manifest and M4.1 closure

Date: 2026-07-15

Starting task: M4.1-08 - generate hashes and a reproducible conversion
manifest.

Completed task: M4.1-08 and milestone M4.1. Added deterministic root artifact
format `colibri-lite-model` version 1, a standard-library generator, strict
metadata/full validators, synthetic failure tests, ADR 0018, task/closure
reports, and compact non-canonical build evidence. Existing dense, expert,
tokenizer, `ArtifactManifest`, and `ExpertStore` contracts remain unchanged.

Canonical evidence: Two independent generations produced the same 11,395 bytes
and root SHA-256
`f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`.
The root contains 57 unique canonical relative file records and no timestamp,
absolute/local path, temporary directory, user/host name, or build identity.
Regeneration after moving the complete directory produced the same bytes/hash.

Closure footprint: 122,147,678,312 logical bytes across 58 files including the
root manifest. Components are 12,786 source-provenance bytes; 6,164,520,354
dense bytes across a manifest and 435-tensor payload; 115,967,247,918 expert
bytes across a manifest and 48 shards for 6,144 experts; and 15,885,859
tokenizer bytes across its manifest and four assets.

Validation evidence: Metadata mode passed after hashing 19,187,816 bytes across
9 files and checking sizes/cross-references for every required file. Full mode
passed after hashing 122,147,678,312 bytes across all 58 files. Metadata
validation passed after moving the real complete artifact, and synthetic full
validation passed after relocation.

Tests: Four grouped synthetic tests cover deterministic generation, metadata
and full validation, relocation, missing/renamed/truncated/corrupted files,
hash/size mismatch, missing/duplicate expert shards, unsupported versions,
incompatible architecture/model type, unknown critical root fields, and
incomplete temporary output. All tests passed.

Commands executed: Two canonical generations; canonical-environment audit;
real metadata/full validation; real relocation and post-move regeneration;
`python -m unittest python\reference\test_model_artifact_manifest.py -v`;
`cargo fmt --all --check`, `cargo check --workspace`,
`cargo test --workspace`,
`cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo run -p clr-cli`.

Standard verification: All 118 workspace tests passed with zero failures or
ignored tests. Clippy passed with warnings denied and the CLI smoke reported
`bootstrap ready`.

Known issues: The F32 artifact requires approximately 113.76 GiB. Rust text
tokenization, chat rendering, quantization, mmap, SIMD, GPU work, performance
optimization, and full-model numerical inference remain unimplemented.

Next task: M4.2-01 - validate selected tensor values against Safetensors.

## 2026-07-15 - pre-M4.2 temporary storage audit

Date: 2026-07-15

M4.2-01 remains paused and unstarted. Audited all project-owned
`D:\tmp\colibri-m4.1-*` directories without deleting or creating model
payloads. Recorded full path, recursive bytes/file count, creation/modification
times, duplicate relationships, hard-link identities, classifications, and
preflight/post-dry-run disk accounting in the storage audit report.

Canonical artifact root:
`D:\tmp\colibri-m4.1-08-moved\relocated-model-artifact-v1`. The 17 pinned
source files under `D:\tmp\colibri-m4.1-05` remain separately protected and
required for M4.2-01.

Evidence: 244,392,722,124 logical bytes are proposed cleanup entries. Exact
last-link reclaimable file bytes are 122,264,231,628; 122,128,490,496 bytes are
shared dense/expert hard-link names that remain through canonical paths. Dense,
expert, vertical-slice, tokenizer, and source classifications were established
from hashes, component manifests, file IDs, and hard-link lists. No incomplete
or orphaned payload was found.

Added `docs/temp-artifact-policy.md`, a reviewed machine cleanup plan,
dry-run-first `scripts/cleanup_temp_artifacts.py`, and five safety tests. The
real dry run passed with 11 candidates, 117 files, no deletion, and zero
free-space delta. Cleanup apply mode was not invoked.

Standard verification passed: formatting, workspace check, all 118 Rust tests,
warning-free clippy, CLI bootstrap, all 5 cleanup safety tests, and
`git diff --check`.

Next: review the classification and cleanup decision. Do not resume M4.2-01
until cleanup approval is recorded.

## 2026-07-15 - stable artifact promotion and approved cleanup

Date: 2026-07-15

The conditional cleanup review was approved. Promoted the canonical artifact
by same-volume directory rename to
`D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1`; no 122 GB payload copy was
created. Added a tracked canonical-root registry and made cleanup require an
exact registry match outside the temporary namespace.

Pre-cleanup metadata and full-integrity validation passed at the stable root.
The second dry-run matched exactly 11 approved paths, 117 files,
244,392,722,124 logical bytes, 122,128,490,496 shared hard-link bytes, and
122,264,231,628 expected last-link bytes. Applied only that reviewed set.

Disk free bytes increased from 202,957,639,680 to 325,221,920,768. Logical
bytes removed were 244,392,722,124; actual physical bytes reclaimed were
122,264,281,088. All 11 candidates are absent. The stable 58-file artifact and
17 pinned source files remain present.

Post-cleanup metadata/full validation passed across all 122,147,678,312
canonical bytes with root hash
`f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`.
Pinned-source validation passed across 17 files and 61,068,275,406 bytes. All 6
cleanup tests, 3 source-validator tests, 118 Rust tests, formatting, workspace
check, warning-free clippy, and CLI bootstrap passed.

Next task: M4.2-01 - validate selected tensor values against Safetensors.

## 2026-07-15 - M4.2-01 selected full-model tensor values

Date: 2026-07-15

Completed task: M4.2-01. Added a deterministic, standard-library validator that
binds the stable canonical-root registry, pinned source provenance, dense and
expert source plans, component manifests, and a versioned tensor selection.

Validated 88 exact BF16-to-F32 bit samples across 13 dense tensors and 9
gate/up/down projections for three experts spanning layers 0, 24, and 47.
Selected source shards 0, 7, 14, and 15 were hash-verified across
13,084,683,552 bytes before payload sampling. No value, shape, offset, dtype,
orientation, or layout mismatch occurred.

Two real runs produced identical 34,479-byte evidence with SHA-256
`4c77a28a4ccc6fa764c5d7da64379f737278924caf65f295eb71930127818068`;
the second run left the existing evidence unchanged.

Next task: M4.2-02 - validate selected layer router IDs against Transformers.
