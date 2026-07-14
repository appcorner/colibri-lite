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
