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

## 2026-07-16 - M4.2-02 Layer-24 router validation

Date: 2026-07-16

Completed the reviewed Layer-24 subtask of M4.2-02. The genuine Rust path now
executes the embedding and complete streaming Layers 0-23, then executes Layer
24 only through router selection. It loads only selected experts through the
normal `ExpertStore`; Layer-24 experts and later operations remain unreachable.

Transformers F32 and Rust selected identical experts for every layer 0-24. All
four Layer-24 F32 classifications are `exact_match_safe`; all four independent
BF16 classifications are `numerically_ambiguous`. The Layer-24 router maximum
error is `3.719329833984375e-5`, and the minimum F32 boundary margin is
`0.01013946533203125`, safely above its `0.0000743865966796875` required
margin.

The first scalar-only combined-MoE guard stopped at Layer 1. A focused
same-Rust-input diagnostic reduced the maximum from `3.509521484375e-4` to
`8.392333984375e-5` and produced zero scalar-contract failures, classifying the
original one-element failure as accumulated incoming drift rather than a local
expert implementation mismatch. ADR 0022 freezes per-layer propagated budgets;
RMSNorm, router ordering, expert arithmetic, and global tolerances are
unchanged.

Rust loaded 546 unique layer/expert keys for 768 occurrences, with 0 hits, 546
misses/loads, 545 evictions, and `18,874,368` peak expert-resident bytes. Dense,
expert, and total artifact reads were `1,914,119,168`, `10,305,404,928`, and
`12,219,524,096` bytes. Modeled peak explicit Rust memory was `126,140,262`
bytes; maximum Python peak working set was `660,402,176` bytes.

Evidence: two reference runs and two final Rust runs were byte-identical. All
temporary validation runs were removed. The canonical artifact and pinned
source remained read-only, and no model payload was copied.

Verification: `cargo fmt --all --check`, `cargo check --workspace`, all 123
workspace tests, workspace Clippy with warnings denied, CLI bootstrap,
feature-gated Clippy, Python compilation, four router-policy tests, and the
focused optimized Layer-24 validation all passed. The focused test reported 1
passed, 0 failed, and 73 filtered tests.

Open issue: ADR 0022 budgets are provisional for this frozen Layer-24 path and
must not be reused for Layer 47.

Next subtask after review: M4.2-02 Layer-47 router validation. Do not begin it
without the requested review.

## 2026-07-16 - M4.2-02 Layer-47 router validation and task closure

Date: 2026-07-16

Completed Layer-47 router validation and M4.2-02. The genuine Rust F32 path
executed embedding and complete streaming Layers 0-46, then Layer 47 only
through pre-router and router selection. Layer-47 experts, the Layer-47 block,
final norm, LM head, logits, sampling, and generation remained unreachable.

Transformers F32 and Rust selected identical experts for all four tokens at
every layer 0-47. All Layer-47 F32 classifications are `exact_match_safe`; all
independent BF16 classifications are `numerically_ambiguous`. The smallest F32
margin is `0.04711651802062988`, safely above its `0.00001811981201171875`
required margin.

ADR 0023 freezes a new Layer-47 propagated budget from the measured Layers
24-47 components. The largest block drift is `2.3193359375e-3`, first reached
at Layer 3. Layers 4-45 remain flat, and Layer 46 reduces the error to
`9.765625e-4`; no anomalous late increase or out-of-budget checkpoint occurred.
No isolated diagnostic was required and no arithmetic or numerical contract
changed.

Rust executed 1,504 expert occurrences and loaded 1,045 unique layer/expert
keys. Cache metrics were 0 hits, 1,045 misses/loads, 1,044 evictions, and
`18,874,368` peak resident expert bytes. Dense, expert, and total artifact reads
were `3,675,078,656`, `19,723,714,560`, and `23,398,793,216` bytes. Modeled
peak explicit Rust memory was `136,869,670` bytes; maximum Python peak working
set was `693,604,352` bytes.

Two reference exports and two final Rust evidence runs were byte-identical. All
Layer-47 run directories were removed; the canonical artifact and pinned source
remained read-only. Eleven compact files totaling `24,009,529` logical bytes
were promoted.

Verification passed: formatting, workspace check, all 123 workspace tests,
warning-free workspace and feature Clippy, CLI bootstrap, Python compilation,
four router-policy tests, the Layer-24 feature regression, and both final
Layer-47 feature runs.

Next task after review: M4.2-03 - validate selected intermediate outputs. It was
not started in this session.

## 2026-07-16 - M4.2-03 selected intermediate validation

Date: 2026-07-16

Completed M4.2-03 with eight deterministic expert cases at Layers 0, 1, 24,
and 47. The genuine Rust path completed all 48 blocks through the normal
storage-aware expert path and stopped before final model normalization. Gate,
up, SiLU, product, down, routing-weighted output, aggregation, residual, and
block checkpoints all passed the per-stage ADR 0024 budgets.

The reference exporter preserves Transformers occurrence batching. Its initial
single-row Layer-0 recomputation stopped; a same-input diagnostic proved
separate and concatenated gate/up calls bit-identical, and restoring genuine
occurrence batching reproduced the frozen aggregate exactly. The Rust trace's
down output is bit-identical to the unchanged normal expert output in every
case, so no compensating arithmetic or local implementation defect appeared.

Rust executed 1,536 expert occurrences and loaded 1,066 unique layer/expert
keys with zero hits, 1,066 misses/loads, 1,065 evictions, and `18,874,368`
peak resident expert bytes. Dense, expert, and total artifact reads were
`3,675,078,656`, `20,120,076,288`, and `23,795,154,944` bytes. Modeled peak
explicit memory was `126,685,121` bytes; Python peaked at `761,298,944` bytes.

Two reference exports and two final Rust evidence runs were byte-identical.
The canonical artifact and pinned source remained read-only. RMSNorm, routing,
activation, expert layout, dtype, and cache contracts were unchanged.

Verification passed: formatting, workspace check, all 123 workspace tests,
warning-free workspace and feature Clippy, CLI bootstrap, Python compilation,
11 focused Python tests, and both optimized intermediate evidence runs. The
single 10,457-byte characterization run was removed; no temporary run remains.

Next task after review: M4.2-04 - run a short deterministic token sequence. It
was not started in this session.

## 2026-07-16 - M4.2-04 short deterministic cached sequence

Date: 2026-07-16

Completed the full unquantized path through final RMSNorm, streamed LM head,
vocabulary logits, deterministic greedy selection, and two genuine cached
decode steps. Prompt `[9707, 11, 1879, 0]` generated `[1096, 374]`; both F32
selections are `exact_match_safe` and all top-20 ranks agree.

The fixed 48-layer KV cache reached length 6 with `1,179,648` payload bytes.
All previous K/V prefixes remained byte-identical after append and allocation
capacities never changed. Transformers incremental and full recomputation also
agree on argmax through the processed final token.

Rust read `29,518,290,944` dense and `43,486,543,872` expert bytes. The 2,304
selected-expert loads produced zero hits and 2,303 evictions with `18,874,368`
peak expert residency. Modeled peak explicit memory was `127,823,000` bytes.

Two final reference passes and two final Rust passes were byte-identical. The
two successful temporary directories, including both full vocabulary-row
files, were removed. No canonical or pinned source artifact changed.

Verification passed: formatting, workspace check, all 123 workspace tests,
warning-free workspace and feature Clippy, CLI bootstrap, Python compilation,
11 focused Python tests, and both optimized generation evidence runs.

Next task after review: M4.2-05 - record peak resident bytes, bytes read, and
cache metrics. It was not started in this session.

## 2026-07-16 - M4.2-05 resource and I/O baseline

Date: 2026-07-16

Completed three fresh-process measurements of the unchanged M4.2-04 scalar
full-model fixture. Every run preserved prompt `[9707, 11, 1879, 0]`, generated
`[1096, 374]`, both F32 `exact_match_safe` classifications, and the fixed
48-layer KV cache with 1,179,648 bytes and no allocation growth or overwrite.

Rust inference-only wall time ranged from 393.820 to 437.023 seconds. Process
peak working set ranged from 145,424,384 to 148,066,304 bytes, while modeled
explicit memory remained 127,823,000 bytes. The three runs read
29,518,290,944 logical dense bytes and 43,486,543,872 logical expert bytes for
73,004,834,816 total logical artifact bytes. These are application range reads,
not physical disk I/O; no cold-cache claim was made.

The one-expert 18,874,368-byte cache handled 2,304 occurrences across 1,332
unique layer/expert keys. It recorded zero hits, 2,304 misses/loads, 2,303
evictions, and 972 cross-token reuses. Overall read amplification was 2.428603x
and expert amplification was 1.729730x. Zero hits did not trigger optimization.

All non-timing deterministic metrics matched across the three processes.
Verification passed all five standard Cargo commands, 123 workspace tests,
warning-free workspace and feature Clippy, Python compilation, 20 Python unit
tests, the focused reconciliation test, and the M4.2-04 correctness regression
in every measured run.

Policy cleanup removed five flat run directories, 18 files, and 7,511,832
logical bytes after a matching dry run. No temporary run remains. The canonical
artifact and pinned source stayed read-only.

Next task after review: M4.2-06 - document failures or tolerance differences
before optimization. It was not started in this session.

## 2026-07-16 - M4.2-06 correctness and variance closure

Date: 2026-07-16

Closed M4.2 with the verdict `completed with documented numerical variance`.
The closure classifies every observed issue, keeps performance limitations
separate from correctness failures, records remaining coverage risks, and adds
an authoritative machine-readable tolerance registry plus optimization
invariants.

No Rust runtime implementation defect remains in the validated F32 path. The
initial post-RMSNorm stop was PyTorch reduction-order variance; ordered Python
F32 and Rust remain bit-exact. The Layer-1 scalar stop was accumulated incoming
drift, and the selected-intermediate discrepancy was a reference occurrence-
batching issue. Router ties retain higher-score/lower-ID ordering. BF16 remains
an independent model-behavior reference.

The registry covers 24 checkpoint/selection contracts and preserves their
distinct scopes: exact internal, provisional cross-runtime, layer-specific,
fixture-specific, diagnostic-only, and semantic-margin. Four focused tests
validate checkpoint completeness, scope rules, supporting documents, frozen
fixture invariants, and 12 content-addressed evidence references.

The frozen optimized M4.2-04 regression regenerated exact reference hashes,
passed all budgets and KV invariants, generated `[1096, 374]`, and retained the
existing Rust evidence byte-for-byte. Policy cleanup removed the four temporary
full-logit files and retained no M4.2-06 run.

Verification passed all five standard Cargo commands, 123 workspace tests,
warning-free workspace and feature Clippy, Python compilation, all 24 Python
tests, registry consistency, evidence-reference validation, and the frozen
generation regression.

Next ordered task after review: M4.3-01 - establish an unquantized or higher-
precision correctness baseline. No optimization or M4.3 implementation began
in this session.

## 2026-07-16 - M4.3-01 frozen F32 reference baseline

Date: 2026-07-16

Froze the existing ordered scalar Rust F32 full-model path as the authoritative
baseline for future M4.3 variants. Added a canonical baseline manifest,
Tier A/B/C fixture hierarchy, selected diagnostic-only F64 records, a stable
future-comparison schema, ADR 0028, and an optimization-invariants checklist.
No runtime arithmetic, artifact, cache, I/O, dtype, or execution policy changed.

Tier A reuses the approved prompt `[9707, 11, 1879, 0]` and generated IDs
`[1096, 374]`. Tier B executed six complete-forward fixtures totaling 11
positions and covering low/high token IDs, English, Thai, code/newline,
repeated text, and the end-of-text special token. All guard router IDs,
top-20 ranks, argmax IDs, finite-output checks, KV allocations, and cache
counters passed. Tier C references the existing focused M4.2 operation evidence.

The `single_low_token` compact logit difference (`2.8419495e-4`) exceeded the
largest prior M4.2 fixture-specific observation. A required same-input
diagnostic measured only `1.9073486e-5` all-logit local LM-head difference and
showed incoming-state effects up to `2.7370453e-4`; this is accumulated drift,
not a local LM-head defect. New budgets are scoped per Tier B fixture using the
existing `3 * observed + guard` model and do not change the M4.2 registry.

The clean final Rust Tier B regression passed in 926.71 seconds. Its TSV was
byte-identical to the generated tracked evidence with SHA-256
`5632f63acd29b4af09709904d0ffcef12336628a9af51c1b4a8514c857d976f2`.
The canonical baseline manifest SHA-256 is
`5e36071de4f4385f8d4ea3310b3beef138ab65ac4627ea53a575ab5e627b71b4`.

Verification passed all five standard Cargo commands, all 123 workspace tests,
warning-free workspace and feature Clippy, the feature-gated Tier B schema
test, Python compilation, all 29 Python tests, two byte-identical bundle
generations, evidence-reference checks, and the final full Tier B regression.
The M4.2-05 performance baseline remains frozen and was not rerun.

Cleanup dry-run and apply validated and removed two flat run directories with
seven files and 663,661 logical bytes. No M4.3-01 temporary run remains, and
the canonical artifact and pinned source roots remained protected.

Next ordered task after review: M4.3-02 - define the first candidate expert
quantization format. No quantization or performance implementation began.

## 2026-07-17 - M4.3-02 first expert quantization format

Defined and evaluated three representative INT8 formats without modifying the
runtime or creating a quantized artifact. The experiment read 24 canonical F32
projection matrices from Layers 0, 1, 24, and 47 for experts 62, 91, 68, 127,
85, 8, 54, and 36. It evaluated 72 matrix quantizations and 24 same-input
gate/up/activation/product/down/weighted chains.

Per-tensor INT8 was rejected. Per-output-channel INT8 was selected for the
first future implementation: its maximum weighted expert error was 0.1043549,
modeled expert size 4,733,280 bytes, 4.200218x compression, and 226 experts
under a 1 GiB binary cache. Input-group-128 was numerically better at 0.0820999
but costs 2.81% more bytes and remains promising rather than selected.

The selected format is symmetric INT8, F32 per-output-row scales, nearest-even
rounding, saturation to [-127,127], F32 activations and accumulation, and
little-endian 64-byte-aligned deterministic serialization. The draft runtime
contract dequantizes one complete projection to F32 before using the existing
scalar operation. The additive artifact schema is versioned separately from
the canonical F32 artifact.

Evidence was generated twice with identical hashes:

- JSON: `fe8b7d06d013227952f6387969d705c433913f4c8a95db56f7abda1324f5ddf1`
- TSV: `b2bb78d52c96fa8dec4c35cb9d80daabb48576a864b2f8f1689845b77cd2208b`

Added the candidate report, ADR 0029, format specification, additive artifact
schema, runtime kernel contract, provisional correctness gates, and five
deterministic schema/evidence tests. No F32 tolerance or runtime behavior was
changed.

Next ordered task after review: M4.3-03 - keep router and sensitive dense
tensors at measured safe precision.

## 2026-07-17 - M4.3-03 sensitive dense precision policy

Measured embedding, attention Q/K/V/O, normalization, Q/K norm, router,
final-norm, and LM-head weight groups at Layers 0, 1, 24, and 47 using F32,
BF16-rounded-to-F32, and offline per-output-channel INT8 diagnostics. The
canonical F32 artifact remained read-only; no mixed artifact, runtime kernel,
cache change, or arithmetic change was made.

The canonical artifact is BF16-derived, so BF16-rounded weights were exact in
the controlled weight-only experiment. This does not authorize BF16 activation
or kernel arithmetic. INT8 router changed Layer-0 IDs despite a positive safe
margin (`0.0468793`) and is rejected. Attention O INT8 local output error
reached `0.2976232`; dense INT8 remains diagnostic-only. Router, RMSNorm, Q/K
norm, final norm, routing weights, residuals, activations, and accumulations
remain F32. Embedding, attention Q/K/V/O, and LM-head weights are future BF16
storage candidates requiring Tier C/B/A evidence.

The deterministic evaluator produced 117 records and 12 router records twice
with JSON SHA-256
`1387addd232a80e970af00d7c86dc1a747085589fff14663b2f909ab3b38db81`.
Added the precision-sensitivity report, machine-readable evidence, tensor
registry, mixed-precision policy draft, ADR 0030, and focused Python tests.

Verification completed: Python compilation, four policy/evidence tests, and
repeated deterministic evidence generation. The standard Cargo verification
commands are run before the review commit.

Next ordered task after review: M4.3-04 - compare output degradation against
the frozen F32 baseline.

## 2026-07-17 - M4.3-04 expert INT8 degradation study

Ran the selected symmetric INT8 per-output-channel expert simulation with F32
scales, F32 expert activations/accumulation, and all non-expert tensors at F32.
The simulation quantized and dequantized selected projections on demand and
did not create a quantized artifact or modify production runtime behavior.

Tier C covered eight representative expert cases and passed all provisional
M4.3-02 gates. Tier B covered all six frozen fixtures. No safe-margin router
true mismatch occurred, but propagated drift became material at Layer 1 and
reached a maximum Layer-47 final-block error of `152.1300659`. The Thai fixture
argmax changed from `7360` to `16222` under a numerically ambiguous margin.
Tier A retained generated IDs `[1096, 374]`, with several ambiguous vocabulary
and router classifications.

The candidate is classified `quality_risk`, not accepted for runtime prototype.
The F32 tolerance registry, canonical artifact, router policy, norms, cache,
and production runtime remain unchanged. Evidence was generated with SHA-256
`0a2f5c85087de32a23b975bc206ed98b007e353dbc897fb71317fcef6568e140`.

Verification for this task includes deterministic Tier C/B/A simulation and
the focused degradation tests; standard Cargo and full Python verification are
run before the review commit.

Next ordered task after review: M4.3-05 - compare memory/I/O and speed against
ik_llama.cpp where formats and hardware permit.

## 2026-07-17 - M4.3-05 ik_llama comparison

Pinned `ik_llama.cpp` at `1fddd12ba861c4815a8633f14d9c5670692099cc` with a
clean external checkout and CPU-only Release build. The exact Qwen3-30B-A3B
Q4_K_M artifact was downloaded read-only into one flat temporary run, verified
at 18,556,685,824 bytes and SHA-256
`0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`.

Four fresh-process short runs and one 32-token decode completed with the exact
fixture `Hello, world!` -> `[9707, 11, 1879, 0]`. Short CLI decode was
`5.276-5.863` tok/s with 4.36-4.41 GB peak working set; the long decode was
`3.511` tok/s with 9.03 GB peak working set. Three `llama-bench` repetitions
reported `3.968 +/- 0.875` prompt tok/s and `2.819 +/- 0.099` decode tok/s.
The result is directional because Q4_K_M, mmap, fused MoE, SIMD, graph reuse,
and optimized kernels differ from colibri's F32 scalar path. Physical I/O was
not claimed and no runtime code changed.

Added the environment manifest, compact run metrics, comparison report, ADR
0032, and the reproducible external benchmark script. Successful temporary
model files are removed after evidence review.

Next ordered task after review: M4.3-06 - select or reject the candidate based
on recorded evidence.

## 2026-07-17 - M4.3-06 candidate decision and phase closure

Formally rejected symmetric INT8 per-output-channel expert weights as a
full-model production candidate. The format remains retained for diagnostics
and possible selective-layer/selective-projection investigation only with new
evidence. Local Tier-C gates passed, but propagated risk began at Layer 1
(`0.1724930` final-block error), reached Layer 47 (`152.1300659`), changed the
Thai Tier-B argmax (`7360` to `16222`), and reduced Top-20 overlap to `0.65`.
Tier-A IDs `[1096, 374]` matched but included ambiguous steps and are not a
sufficient acceptance criterion.

M4.3 phase verdict: completed with documented quantization rejection and
optimization pivot. F32 is authoritative; per-tensor INT8 is rejected;
group-128 is promising but insufficiently evidenced; router INT8 is rejected;
dense BF16 is a future kernel/activation candidate; ik_llama Q4_K_M is an
external performance reference only.

The project continues, pivoting from immediate quantized-runtime work to the
simulation-only `F32 memory-hierarchy study: resident dense weights plus
trace-driven expert-cache sizing` at 1/2/4/8/16/24/32 binary GiB. No runtime
change is authorized before simulation review.

Added candidate status and decision registries, M4.3 closure report, memory
hierarchy roadmap, ADR 0033, and four registry/evidence tests.

Exact next task after review: M4.4-01 - emit versioned baseline JSON.

## 2026-07-17 - M4.4-01 versioned performance baseline index

Created the canonical deterministic baseline index
`models/qwen3-30b-a3b/m4.4-performance-baseline-v1.json` with stable ID
`qwen3-30b-a3b-colibri-f32-windows-x64-v1`. It references 31 existing model,
correctness, resource, quantization, invariant, roadmap, and ik_llama evidence
files by canonical relative path, byte count, and SHA-256; no model payload or
large checkpoint evidence was duplicated.

The generator validates canonical model revision and artifact inventory, F32
authority, known evidence schema versions, frozen fixture IDs, performance and
cache reconciliation, unique baseline IDs, and rejection of production
quantization candidates. Serialization is sorted-key compact UTF-8 JSON with a
trailing newline and no timestamp. Repeated generation is byte-identical.

Added focused baseline-schema/reference/hash/reproducibility tests and the
human-readable M4.4-01 report. Runtime behavior, artifacts, cache capacity,
and numerical arithmetic were unchanged; RAM/cache simulation has not started.

Next ordered task after review: M4.4-02 - record runtime/model commits and
artifact version.

## 2026-07-17 - M4.4-02 M4 release provenance and closure

Created the authoritative release record
`models/qwen3-30b-a3b/m4-release-provenance-v1.json` with release ID
`colibri-lite-rs-m4-qwen3-30b-a3b-f32-v1`. It pins runtime commit
`a230074959fc3b55ff73e8f4eb24e377a0a6b79f`, parent M4.4 baseline commit
`80099f05246a4450ded6f42baf6b8db5a4b2e623`, the Qwen3 revision,
source/tokenizer identity, canonical artifact hashes, F32 contracts, M4.4
resource evidence, M4.3 decisions, and the directional ik_llama reference.

Added provenance validation and deterministic regeneration tests, release notes,
README status, and M4/M5 task tracking. Validation rejects changed hashes or
identities, accepted rejected candidates, duplicate release IDs, and any M5
task marked started. No runtime, artifact, cache, numerical, or performance
implementation changed; no RAM/cache simulation ran.

The approved tag is `m4-full-qwen3-baseline-v1` and must point to the final
clean closure commit. Exact next task: M5.1-01 trace-driven memory hierarchy
simulation.

## 2026-07-17 - M5.1-00 authoritative ordered expert trace

Added behavior-preserving request instrumentation at the `ExpertStore` load
boundary and replayed the frozen M4 Tier-A configuration. The canonical trace
contains 2,304 actual ordered expert occurrences and 1,332 unique
layer/expert keys. Validation reproduced 0 hits, 2,304 misses/loads, 2,303
evictions, 43,486,543,872 expert logical bytes, and the frozen reuse-distance
distribution (379/384/1,924; 525/298/149 buckets).

The canonical trace and an independent repeat are byte-identical at
SHA-256 `f3f87f05d15424030c9261cdf3e93bd72e9c006a55303bc0c28a92a4fb3ff2d0`.
Both replays preserved generated IDs `[1096, 374]`, guard router IDs,
finite outputs, selected checkpoints, and KV-cache invariants. The replay is
explicitly a measurement supplement; temporary full-vocabulary logits were not
retained, so no new full-logit claim is made.

Added trace schema/manifest, deterministic capture and validation scripts,
aggregate evidence, report, and ADR 0034. No cache simulation or runtime
memory-hierarchy prototype has started.

Exact next task after review: M5.1-01 - trace-driven memory hierarchy
simulation.

## 2026-07-17 - M5.1-01 trace-driven memory hierarchy simulation

Implemented a deterministic Python simulator over the authoritative M5.1-00
ordered trace. The input and result artifacts validate the trace SHA-256,
frozen M4 baseline/provenance identity, key ranges, payload accounting, and
all scenario identities. Simulated binary 1/2/4/8/16/24/32 GiB budgets for
streamed-dense and resident-dense configurations under global LRU,
layer-aware LRU, observed-frequency LFU, and a clearly theoretical Belady
upper bound.

Global LRU requires exactly 379 charged entries (`7,154,937,856` bytes) for
the first hit; the first fixed total-RAM point with a hit is 8 GiB. At 8 GiB
streamed-dense LRU, expert-byte hit rate is 31.21% and modeled total logical
reads are reduced 18.59%. Full unique-key residency requires 25,146,114,048
bytes including entry charge. Dense residency is infeasible below 8 GiB and
competes with the expert working set at 8/16 GiB.

Selected the configurable larger expert cache as the first runtime prototype
for a future task. This task made no Rust runtime, cache-policy, artifact,
quantization, mmap, prefetch, SIMD, threading, GPU, or numerical changes.
Added the simulator, focused synthetic-policy tests, input manifest, result
matrix, report, and ADR 0035. Deterministic regeneration and accounting tests
pass.

Exact next task after review: M5.1-02 (or the next task specified by the
approved roadmap), before implementing the selected prototype.

## 2026-07-17 - M5.1-02 configurable expert-cache prototype

Promoted the existing safe Rust byte-budgeted `ExpertCache` into the reviewed
configurable prototype contract. The API continues to accept an explicit
payload-byte budget through `ExpertStore::new`, preserving the one-expert
default. Added configured-budget, resident/peak entry, bytes-served/avoided,
oversized-entry, and blocked-eviction metrics. Strict global LRU, deterministic
tie-breaking, lease pinning, bounded payload residency, and oversized-entry
rejection remain unchanged.

Added synthetic cache tests and a Rust trace-replay example with a Python
adapter. Replay of the authoritative M5.1-00 order matched M5.1-01 global-LRU
counters at the 8 GiB and 16 GiB modeled operating points. Payload residency
stayed within budget; metadata/alignment deltas were reported separately.

Full-model correctness and timing were not run because this checkout lacks the
canonical dense/expert payload directories and `COLIBRI_ARTIFACT_ROOT`. No
timing or physical-I/O claim is made. Classification is
`accepted_with_limitations`; next candidate is a broader representative trace
corpus, recorded in ADR 0036.

Exact next task after review: M5.2-01 Capture broader representative expert
traces.

## 2026-07-17 - M5.1-03 canonical full-model cache validation

Validated the configurable F32 strict-global-LRU cache against the canonical
Qwen3-30B-A3B artifact at the one-expert baseline and exact nominal 8 GiB and
16 GiB payload budgets. The artifact manifest and all 57 payload files matched
the pinned root hash. Every run preserved generated IDs `[1096, 374]`, retained
F32 checkpoints, deterministic routing, finite outputs, KV-cache invariants,
and bounded payload residency.

Exact-budget runtime counters matched independent trace replay: 8 GiB had
719 hits, 1,585 loads, 1,130 evictions, and 13,570,670,592 bytes avoided; 16
GiB had 931 hits, 1,373 loads, 463 evictions, and 17,572,036,608 bytes avoided.
The different M5.1-02 counters are explained by its usable-budget overhead
accounting. Logical reads fell 18.59% and 24.07% overall. Timing is directional
because filesystem cache state was uncontrolled and one sample was collected
per mode. Process working-set sampling and full-vocabulary logits remain open
limitations. Classification remains `accepted_with_limitations`.

Added the machine-readable result, validation report, and ADR 0037. Exact next
task: M5.2-01 Capture broader representative expert traces.

## 2026-07-18 - M5.2-01 representative expert traces

Completed:

- M5.2-01 representative corpus capture and validation.
- Eight accepted workload cases, each captured twice; the frozen Tier-A
  control was reproduced from the canonical M5.1-00 trace.
- M5.1-03 counter discrepancy follow-up, resolved as budget-accounting
  semantics with no runtime change.

Changed:

- Added v2 ordered-trace schema, fixture manifest, corpus manifest, seven new
  individual traces, repeatability and aggregate evidence.
- Added deterministic one-fixture-at-a-time capture and analysis tooling.
- Added corpus regression/record-adapter tests, ADR 0038, task status, and the
  M5.2-01 report.

Evidence:

- 11,520 total expert occurrences; 3,148 unique layer/expert keys;
  `217,432,719,360` requested payload bytes.
- Every trace has two byte-identical canonical repeats and stable generated
  IDs. New-trace SHA-256 values are pinned in the corpus manifest.
- All schema, ordinal/range, payload, KV-cache, cache-accounting, finite,
  deterministic, and no-oversized/no-blocked-eviction checks passed.
- Existing simulator record-key compatibility passed without running cache
  simulation on the corpus.
- Descriptive 8 GiB classification is `inconclusive`; no policy recommendation
  was made.

Open issues:

- The corpus is intentionally small and synthetic/short; `single_low_token`
  Tier-B was not included because it adds little diversity.
- Long-context and long-decode cases do not claim independent Transformers
  equivalence. Corpus-wide cache-policy behavior remains unmeasured.

Next:

- M5.2-02 Simulate cache policies and RAM budgets across the representative
  trace corpus. Do not start until corpus review is complete.

## 2026-07-18 - M5.2-02 representative corpus cache simulation

Completed:

- M5.2-02 deterministic cache-policy and expert-payload-budget simulation.
- Validated all eight M5.2-01 traces, hashes, boundaries, ordinals, ranges,
  payloads, canonical artifact identity, M4 baseline/provenance references,
  and M5.1 simulator key compatibility before replay.
- Simulated cold per-session and persistent manifest/reverse fixture orders for
  global LRU, architecture-only layer LRU, calibrated layer LRU, observed LFU,
  segmented LRU, and offline Belady across 1/2/4/6/8/12/16/24/32/48 GiB.

Changed:

- Added `scripts/simulate_m5_2_corpus_cache.py` and synthetic/unit regression
  tests in `scripts/test_simulate_m5_2_corpus_cache.py`.
- Added the simulation input manifest, result matrix, corpus aggregate
  statistics, threshold analysis, persistent-cache evidence, report, ADR
  0039, and task status.
- Kept the payload-only ExpertCache budget contract; metadata/alignment remain
  separate accounting context. No Rust runtime, cache policy, artifact, or
  numerical execution changed.

Evidence:

- Cold global LRU at 8 GiB: 2,808 hits, 8,712 misses/loads, 24.3750% micro
  byte hit rate and 15.5660% macro byte hit rate; 2/8 fixtures have zero hits.
- Cold global LRU at 16 GiB: 31.9444% micro and 19.8707% macro byte hit rate.
- 8 GiB classification: `useful_for_selected_workloads`.
- Selected policy: retain strict global LRU. Next runtime matrix: 8 versus
  16 GiB global LRU; not executed in M5.2-02.
- Canonical result SHA-256:
  `cc76873de24cc29eb8fbfa1580fafa721617bc4b6c5b64f4dd04079048378949`.
- Canonical report SHA-256:
  `925c7e87b7eae0785edc7781b2caa4d0b1224633dc84c72588c565f8f84cefcb`.
- Input manifest SHA-256:
  `d040e505c9ab87b65935f11b68e8fc65aa4b496bb02f3d10832b98eadaf80b5b`.
- Regeneration was byte-identical for both result and report.

Open issues:

- The corpus remains eight deterministic workloads; it is not production
  traffic coverage. Persistent-cache results are order-sensitive, calibrated
  layer policies are diagnostics, and Belady is offline-only.
- No throughput, latency, physical-I/O, or production-readiness claim is made.

Next:

- M5.2-03. Stop before running the selected full-runtime validation matrix.

## 2026-07-18 - M5.2-03 representative full-runtime cache validation

Completed:

- M5.2-03 exact 8 GiB versus 16 GiB strict global-LRU validation.
- Six representative full-model fixtures, 18 runs total; Tier-A,
  long-context, and long-decode repeated twice per budget.
- Exact simulation/runtime counter comparison, process-memory sampling,
  logical-read accounting, timing capture, deterministic trace checks, and
  correctness/invariant validation.

Changed:

- Added runtime-only timing, phase, process-memory, and machine-metrics
  instrumentation around the existing full-model validation path.
- Added deterministic one-fixture-at-a-time harness
  `scripts/capture_m5_2_03_runtime_validation.py`.
- Added runtime evidence, results JSON, report, ADR 0040, task status, and
  this work-log entry. No cache policy, cache semantics, artifact, or
  numerical execution changed.

Evidence:

- All 18 runs matched M5.2-02 exact-budget simulation fields: requests, hits,
  misses, loads, evictions, expert bytes loaded/avoided, and peak resident
  payload bytes.
- All generated IDs, router/request traces, finite-output, KV-cache, and
  bounded-residency checks passed. Oversized-entry and blocked-eviction events
  were zero.
- Selected-subset macro/micro byte hit rates were 19.1488%/27.3838% at 8 GiB
  and 24.8884%/36.1178% at 16 GiB; Thai and special-token workloads had zero
  hits at both budgets.
- Canonical artifact root was revalidated at
  `f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`.
- Evidence hashes: results JSON
  `0a0b964eaca9de55f3244f45b275b8d386b66b448a701a35377fbf85631ae870`,
  report `18cdbe6c9b1868050d87dcfe14858f1fa1c6490d85b5e1010b2f427da823772`,
  ADR `d587f2b5fa8205103ad3a82aa49518fc8aed19772d69a8c0f6ef3f434ebdc127`,
  and harness
  `9934dc3f8e0541863bacc4d4074df8a997e2d18f94ebdd7b6ebc1d0930d1dce6`.
- `cargo fmt --all --check`, `cargo check --workspace`, `cargo test --workspace`
  (126 tests), workspace Clippy, feature-gated Clippy, CLI smoke, Python
  compilation, Python reference tests (60), M5.1 control validation, and
  simulator unit tests (8) passed.

Open issues:

- Filesystem cache state was uncontrolled; timing remains directional and no
  physical-I/O or throughput claim is made.
- Short English and repeated-pattern fixtures were omitted from full-model
  execution but remain in the eight-fixture simulation corpus.
- Persistent cache was not run in full runtime; M5.2-02 persistent results are
  order-sensitive simulation evidence.

Next:

- Exact next task after review: mmap/coalesced expert access study. Do not start
  it in this task.

## 2026-07-18 - M5.3-01 expert access study

Completed:

- M5.3-01 artifact-layout, current-reader instrumentation, range-grouping, mmap
  feasibility, and controlled storage microbenchmark study.
- Validated the canonical artifact root, 48 expert shards, 6,144 expert ranges,
  contiguous gate/up/down payload layout, selected M5.2 trace hashes, ordinals,
  layer/expert ranges, and payload sizes.
- Selected reusable aligned read buffers as the next isolated storage-access
  prototype. It was not implemented in this task.

Changed:

- Added feature-gated `m5-3-instrumentation` reader counters/timings and
  `ExpertPathMetrics` in `clr-storage`; default runtime behavior is unchanged.
- Added `crates/clr-storage/examples/m5_3_expert_access_bench.rs` with exact
  payload-hash checks and isolated current/persistent-handle/buffer loops.
- Added `scripts/analyze_m5_3_expert_ranges.py` and its three synthetic tests,
  deterministic range results, and the layer-47 miss-sequence evidence.
- Added the M5.3-01 report and ADR 0041; updated task status.

Evidence:

- Canonical artifact root SHA-256:
  `f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`.
- Deterministic range results SHA-256:
  `2d7cab4e69d6063bebbd9c392c5635aa56183ee8a2055ad6d28e8e7a210f0ca0`.
- Storage benchmark evidence SHA-256:
  `59fcc85be74158497492d4c05334a490e19fbbbb5cec1b0da2c2651ec67119c`.
- The 64-request current-reader miss subset performed 64 opens, seeks, read
  calls, allocations, and 1,207,959,552 payload bytes; measured wall time was
  6.410 seconds, including 5.748 seconds of SHA-256 verification. Persistent
  handle/fresh buffer was 6.367 seconds and reusable buffer was 5.996 seconds
  in the same uncontrolled-cache run.
- Exact-adjacent grouping reduced simulated operations by 3.5–6.5%; 1 MiB
  bounded grouping added no further grouping. One-layer batching amplified
  bytes by approximately 12.3–15.8x.
- Python compilation, range tests, feature-gated storage tests, benchmark
  compilation, and payload hash checks passed.

Open issues:

- Physical I/O and cold-cache behavior remain unmeasured.
- Full-model matrix-compute timing was not isolated; storage-vs-compute
  dominance remains unknown.
- The selected reusable-buffer prototype needs repeated controlled comparison.

Next:

- Exact next task: implement and independently benchmark reusable aligned read
  buffers. Do not start it in this task.

## 2026-07-18 - M5.3-02 reusable aligned read-buffer prototype

Completed:

- Implemented feature-gated `Reference` and `ReusableAlignedBuffer` reader
  modes in `clr-storage` with one safe reusable staging buffer, exact-range
  hashing, owned `Arc<[u8]>` handoff, variable-size growth, and no unsafe code.
- Extended deterministic reader/path metrics for allocation, growth, reuse,
  read, copy, alignment, fallback, and active reader mode.
- Added byte-equivalence, variable-size lifecycle, truncation/recovery, and
  ExpertStore accounting tests.
- Added deterministic full-runtime capture tooling and validated 24 runs:
  six required fixtures × 8/16 GiB × reference/reusable reader. All rows match
  M5.2-02 simulation counters and committed request traces.
- Added M5.3-02 report and ADR 0042; updated task status.

Changed:

- `crates/clr-storage/src/reader.rs`, `crates/clr-storage/src/expert.rs`, and
  feature declarations in storage/Qwen crates.
- `crates/clr-qwen3-moe/src/full_model_validation_tests.rs` and
  `m5_2_trace_capture.rs` for explicit reader mode and storage evidence.
- `crates/clr-storage/examples/m5_3_expert_access_bench.rs` and
  `scripts/capture_m5_3_02_runtime_validation.py`.
- Machine-readable benchmark/results and 72 per-run runtime evidence files.

Evidence:

- Canonical artifact root SHA-256:
  `f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`.
- M5.3-02 runtime results SHA-256:
  `69121543607046c2c88bf312cae8c506840e74832cad4ac2d328c2658a97641a`.
- M5.3-02 storage benchmark SHA-256:
  `f1a20dfad10da22af89c3b535155f7d2896faa28f8eef81761651a3ae515ebc8`.
- All 24 runtime rows have exact simulation comparison, finite outputs,
  deterministic generated IDs/traces, bounded residency, zero fallback and
  alignment failures, and one reusable allocation per run.
- Isolated 64-request benchmark: reference 64 allocations/1,207,959,552
  allocated bytes versus reusable 1 allocation/18,874,368 staging bytes;
  reusable mean wall 5.589 s versus 5.800 s, with an added full payload copy.
- Full-model simple mean total time: reference 150.41 s, reusable 187.11 s;
  reusable was slower in 9/12 matched rows. Filesystem cache state was
  uncontrolled, so timing is directional and no throughput claim is made.
- `cargo test -p clr-storage --features m5-3-reusable-buffer`: 23 passed.

Open issues:

- The reusable path has microbenchmark-only value and remains non-default.
- Hardware WMI queries were denied in the restricted shell; host details are
  inherited from the M4 baseline.
- Full-model timing has one sample per configuration and no cold-cache control.
- Short English and repeated-pattern remain simulation-only for this task.

Next:

- Exact next task after review: `M5.3-03 Compute profiling`. Do not start it.

## 2026-07-18 - M5.3-03 compute profiling

Completed:

- Investigated and corrected the historical M4 repeated-build guard lifecycle.
- Added a feature-gated hierarchical compute profiler and deterministic
  capture/aggregation tooling.
- Profiled Tier-A control, code, long-context, and long-decode fixtures at
  exact 8 and 16 GiB reference-reader/global-LRU configurations, plus
  disabled/coarse/detailed Tier-A overhead modes.
- Selected an isolated read-only mmap expert-access prototype as the next
  task; it was not implemented.

Changed:

- Added `m5-3-compute-profiling` instrumentation in `clr-qwen3-moe` for model,
  phase, layer, attention, routing, expert load, expert MLP, and LM-head
  scopes, with matrix dimensions and estimated FLOPs.
- Added `scripts/capture_m5_3_03_compute_profile.py` and
  `scripts/analyze_m5_3_compute_profile.py`.
- Added the machine-readable profile results and aggregate, report, ADR 0043,
  task status, and this work-log entry. The reference reader remains default;
  no cache policy, artifact, numerical kernel, or runtime optimization changed.
- Corrected the M4 provenance test to use an explicit historical task snapshot
  for the repeated-build test while retaining a negative test for current M5
  progress.

Evidence:

- All 10 profile rows passed correctness, deterministic trace/output identity,
  bounded residency, zero blocked/oversized events, and exact simulation
  comparison. Eight detailed rows were included in the aggregate.
- The cache lookup/expert-load path measured 71.6--76.4% of model profile
  time; expert MLP measured 4.1--5.5%, LM head 2.8--3.8%, and attention
  approximately 2.0--2.8%.
- Results JSON SHA-256:
  `25036c06623f16cb84cfa681e9697f4ef291951eea89b7b92e3d1a8017aae9c1`.
- Aggregate JSON SHA-256:
  `9800aa25181e843e53fd3989f8a4edec315cab33ae68c52ccb75cba05d89390b`.
- Canonical artifact root SHA-256:
  `f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`.
- `cargo fmt --all --check`, workspace check/test/Clippy, feature-gated
  Clippy, CLI smoke, explicit Python reference tests (61), historical guard
  tests, profiler unit tests, artifact/schema/hash validation, and deterministic
  evidence validation passed. Workspace tests reported 126 passed.

Open issues:

- Filesystem cache state was uncontrolled, so timing is directional and no
  physical-I/O or throughput claim is made.
- General allocator/copy percentages and dense-load-versus-dense-compute
  percentages were not isolated by the current instrumentation; they remain
  explicitly unknown rather than inferred.
- The full feature test binary requires per-run artifact and fixture
  environment variables; the capture harness supplied them for every accepted
  row.
- A redundant post-capture full-root rehash exceeded the 120-second command
  limit because of artifact size; the completed capture preflight remains the
  authoritative artifact validation.

Next:

- Exact next task after review: `M5.3-04 Isolated read-only mmap expert-access
  prototype`. Do not start it in this task.

## 2026-07-18 - M5.3-04 isolated read-only mmap expert access

Completed:

- Added isolated `clr-mmap` using `memmap2 = 0.9.11`, with lazy complete-shard
  read-only mappings and explicit file/map ownership.
- Kept the reference reader default and exposed mmap only through the
  `m5-3-mmap` feature and explicit reader mode.
- Added byte-equivalence, boundary, truncation, missing-file, two-shard,
  repeated-access, and Windows cleanup tests.
- Extended storage evidence with mapping, virtual-byte, first-touch, access,
  copy, and reuse counters; added deterministic benchmark and full-runtime
  capture tooling.

Changed:

- `crates/clr-mmap/`, `crates/clr-storage/src/mmap.rs`, and feature-gated
  reader integration in `clr-storage`.
- `crates/clr-storage/examples/m5_3_expert_access_bench.rs` compares reference,
  reusable staging, and mmap across same-shard and cross-shard scenarios.
- `scripts/capture_m5_3_04_mmap.py`, ADR 0044, and the M5.3-04 report.

Evidence:

- Canonical artifact root SHA-256:
  `f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`.
- Full runtime results SHA-256:
  `05a5aff20b5ce7698825ff1cb50bddc7394d02d54a7673e89382a9a31547af64`.
- Storage benchmark SHA-256:
  `87bfdbdd44975096e20a7d59c3fc6b584e820aed4370d88d62d1ac335eb2b1cb`.
- All 16 reference/mmap rows passed correctness, exact simulation, trace,
  cache, KV, and bounded-payload checks. Mmap was slower in all 8 paired
  comparisons; median total-runtime change was `+5.92%`.
- Mmap mapped 108 GiB virtual shard space in full runtime and measured
  29.46--39.00 GiB peak working set versus 8.72/17.31 GiB for reference at
  8/16 GiB cache budgets. No physical-I/O claim was made.
- `cargo test -p clr-mmap`: 2 passed; `cargo test -p clr-storage
  --features m5-3-mmap`: 24 passed; mmap-enabled feature Clippy passed.

Open issues:

- Mmap is technically correct but has insufficient runtime value and remains
  non-default, non-production, and outside the normal runtime configuration.
- Filesystem cache and page-fault state were uncontrolled; mapped virtual bytes
  are not resident-RAM claims.

Next:

- Exact next task after review: stop the current storage-access optimization
  path due insufficient runtime value. Do not start it.

## 2026-07-18 - M5.4-02 resident-dense runtime prototype measurement closure

Completed:

- Re-verified the read-only canonical Qwen3-30B-A3B artifact root and captured
  all 24 paired rows: six streamed and six resident-dense fixtures at 8 GiB,
  then the same matrix at 16 GiB.
- Confirmed exact generated IDs, trace hashes, router/expert ordering,
  intermediate F32 checkpoints, finite outputs, KV invariants, strict global
  LRU accounting, and total-budget accounting for every passed row.
- Marked M5.4-02 complete for review only. It remains measurement-only.

Changed:

- Updated the M5.4-02 report, task status, and machine-readable results with
  runtime evidence from source commit `6207fef2c6c1acbcafd525379f29da4bf023e5c0`.

Evidence:

- Canonical root SHA-256:
  `f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`;
  58 files and 122,147,666,917 component bytes.
- The 8 GiB resident maximum accounted peak was 8,580,189,336 <=
  8,589,934,592 bytes; the 16 GiB resident maximum was 17,168,026,776 <=
  17,179,869,184 bytes.
- Retained bounded diagnostics were caused by a harness cache-hit assertion and
  an output filename collision, then passed on rerun. They were not memory,
  allocation, numerical, router, KV, or cache-policy failures.

Open issues:

- Timing is directional and uncontrolled. Physical I/O and page-cache behavior
  were not measured, so no latency, throughput, or production-performance claim
  is supported. The two fixtures lacking M5.2 full-runtime dense-read evidence
  remain unavailable.

Next:

- Review the measurement-only evidence. No production/default adoption is
  authorized; decision classification remains `prototype_insufficient_runtime_value`.

## 2026-07-18 - M5.4-02 resident-dense runtime prototype implementation

Completed:

- Added a test-only, opt-in resident dense source guarded by the
  `m5-4-resident-dense` feature and `COLIBRI_DENSE_RESIDENCY_MODE`.
- Added focused validation for one-time loading, range rejection, explicit
  over-budget rejection, and file release.
- Preserved the streamed `File` path as the default and retained the existing
  strict global-LRU `ExpertStore` for all experts.

Changed:

- `crates/clr-qwen3-moe/src/m5_4_resident_dense.rs` and the full-model
  validation harness.
- Added the M5.4-02 report and result record with unavailable runtime fields.

Evidence:

- `cargo test -p clr-qwen3-moe --features full-model-validation,m5-4-resident-dense m5_4_resident_dense`: 3 passed.
- The full-model validation harness compiles with the prototype feature.
- The canonical 122 GB artifact is not mounted in this workspace, so no
  full-model measurement was executed and no timing or I/O claim was made.

Open issues:

- The six-fixture 8/16 GiB capture remains required on the registered canonical
  artifact. The two M5.2 fixtures without dense-read evidence remain unavailable.

Next:

- Run the M5.4-02 six-fixture baseline/candidate matrix under the registered
  canonical artifact; do not treat this measurement-only prototype as a
  production adoption decision.

## 2026-07-18 - M5.4-01 resident-dense candidate simulation

Completed:

- Ran a deterministic simulation-only resident-dense plus strict global-LRU
  study over the validated M5.2 corpus.
- Validated all eight corpus inputs and used the six fixtures with recorded
  full-runtime dense-read evidence for the candidate matrix.
- Modeled total-RAM budgets at 8, 12, 16, 24, 32, and 48 binary GiB by
  reserving dense/runtime components before assigning the remaining bytes to
  expert payload cache.
- Added focused simulator invariants for fixed-component accounting and
  configuration validation.

Changed:

- Added `scripts/simulate_m5_4_resident_dense.py` and
  `scripts/test_simulate_m5_4_resident_dense.py`.
- Added `models/qwen3-30b-a3b/m5.4-01-resident-dense-simulation-v1.json` and
  `docs/reports/m5.4-01-resident-dense-simulation.md`.
- Added ADR 0045 and promoted resident dense plus strict global LRU from a
  deferred idea to a separately reviewed candidate direction.
- Updated the implementation plan, task tracker, README, and backlog.
- No Rust runtime, artifact, cache policy, numerical path, dependency, or
  production default changed.

Evidence:

- Six-fixture aggregate modeled total logical-read reduction: resident dense
  40.43% at 8 GiB and 56.90% at 16 GiB; streamed dense was 16.31% and 21.36%
  under the same total-RAM accounting.
- At 8 GiB resident dense leaves 1.981 GiB for experts and produces zero
  simulated expert-byte hits; at 16 GiB it retains 27.64% expert-byte hits.
- `python -m unittest scripts.test_simulate_m5_4_resident_dense`: 3 passed.
- The simulation validated the canonical artifact identity and M5.2 input and
  runtime evidence hashes before replay.

Open issues:

- Resident-dense runtime behavior, latency, throughput, physical I/O,
  allocator overhead, working-set behavior, and concurrent behavior remain
  unmeasured.
- Two M5.2 corpus traces were not included because their full-runtime
  dense-read evidence was not recorded.
- The candidate is not a production preset and does not authorize runtime
  implementation by itself.

Next:

- Stop for separate review of `M5.4-02 Measurement-only resident-dense runtime
  prototype`. Do not implement it until explicitly approved.
