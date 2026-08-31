# colibri-lite-rs — Agent Working Rules

## 1. Mission

The only primary objective is:

> Build `colibri-lite-rs`, a Rust-first, hardware-aware inference runtime for Mixture-of-Experts models that maximizes measured tokens/s within explicit resource, quality, and correctness budgets.

The first supported architecture is Qwen3-MoE.
The first full-size target is Qwen3-30B-A3B on Windows x64.

The agent must optimize for:

1. Numerical correctness
2. Reproducible evidence
3. Predictable, budget-enforced RAM and VRAM use
4. Hardware-aware expert placement and backend selection
5. Maintainable Rust code

Do not broaden the project into a general-purpose inference framework before M6 is complete.

---

## 2. Source of truth

The following files define scope and task order:

1. `implementation-plan.md`
2. `tasks.md`
3. `AGENTS.md`
4. Existing tests and recorded fixtures
5. Architecture Decision Records under `docs/adr/`

When instructions conflict, follow the order above unless the user explicitly overrides it.

Do not silently change milestone definitions or acceptance criteria.

---

## 3. Execution order

Work milestone-by-milestone and task-by-task.

Rules:

* Complete tasks in the order listed in `tasks.md`.
* Do not begin a later milestone while the current milestone has failed acceptance criteria.
* Tasks may run in parallel only when they have no shared contract or implementation dependency.
* Update task status immediately after evidence confirms completion.
* Never mark a task complete because code merely compiles.
* A task is complete only when its acceptance criteria and relevant tests pass.

Current priority:

```text
M6.0
→ M6.1
→ M6.2
→ M6.3
→ M6.4
→ M6.5
```

---

## 4. Scope control

Before implementing any new idea, classify it as one of:

* `NOW`: required to close the current milestone
* `NEXT`: required by the immediately following milestone
* `BACKLOG`: useful after M6
* `OUT-OF-SCOPE`: not aligned with the product mission

Only `NOW` work may be implemented without changing the plan.

Ideas classified as `NEXT` or `BACKLOG` must be recorded in `docs/backlog.md` and not implemented.

The following are deferred pending the stated M6 gate:

* GPU backends other than the single backend selected by M6.1/M6.2 evidence
* CUDA, Vulkan, Metal, or DirectML before M6.3 review approves one candidate
* HTTP or OpenAI-compatible server
* Web UI
* Agent frameworks
* Tool calling
* Speculative decoding
* katgpt-rs reasoning features
* Multimodal support
* Continuous batching
* Distributed inference
* Broad GGUF compatibility
* Multiple architecture families
* Performance parity with llama.cpp or ik_llama.cpp

---

## 5. Correctness rules

Correctness always precedes optimization.

The agent must:

* Build the Python/Transformers oracle before implementing matching Rust inference behavior.
* Freeze deterministic seeds, versions, fixture hashes, input token IDs, expert selections, intermediate outputs, and expected logits.
* Compare intermediate outputs before final logits.
* Report the first stage that diverges.
* Use exact comparisons for IDs, shapes, tensor names, and selected experts.
* Use documented numerical tolerances for floating-point outputs.
* Never increase tolerances merely to make a failing test pass without explaining and justifying the difference.
* Never replace a failing numerical test with a weaker smoke test.
* Never assume a generated output is correct because it looks plausible.

If Rust and the oracle disagree, treat the Rust implementation as incorrect until evidence proves otherwise.

---

## 6. Optimization rules

Do not optimize code without measurements.

Before an optimization:

1. Record the current benchmark.
2. Identify the measured bottleneck.
3. State the expected improvement.
4. Implement the smallest possible change.
5. Re-run correctness tests.
6. Re-run the same benchmark.
7. Record whether the optimization was retained or reverted.

Never introduce these before the applicable reference correctness path and a
reviewed M6.3 candidate are established:

* SIMD intrinsics
* FFI kernels
* Quantization
* Memory mapping
* Async prefetch
* Lock-free structures
* Custom allocators
* Unsafe pointer arithmetic
* Kernel fusion

Readable scalar Rust is preferred for the first correctness path.

---

## 7. Unsafe Rust policy

Safe Rust is the default.

`unsafe` may only be introduced when:

* A milestone explicitly requires it.
* A safe implementation already exists and passes correctness tests.
* A benchmark demonstrates a material need.
* The unsafe boundary is isolated in the smallest possible module.
* Every unsafe block documents its safety invariants.
* Tests cover lifetime, bounds, aliasing, file-lifetime, and failure behavior.
* The change receives a dedicated review.

Do not disable the workspace unsafe lint globally.

Do not spread unsafe APIs into `clr-core` public contracts.

---

## 8. Dependency policy

Avoid dependencies unless they clearly reduce risk or implementation complexity.

Before adding a crate, document:

* Why the standard library is insufficient
* Maintenance activity
* License
* MSRV compatibility
* Windows x64 support
* Transitive dependency impact
* Whether it introduces unsafe code
* Whether the dependency is needed in the current milestone

Do not add dependencies for convenience alone.

`clr-core` should remain dependency-free unless explicitly approved.

Use exact or intentionally constrained dependency versions for reproducibility.

---

## 9. Architecture boundaries

Dependency direction must remain:

```text
clr-storage → clr-core
clr-qwen3-moe → clr-core + clr-storage
clr-cli → clr-core + clr-storage + clr-qwen3-moe
```

Rules:

* `clr-core` must not know about files, Qwen, CLI, Hugging Face, or model artifacts.
* `clr-storage` must not implement attention, routing, activation functions, or architecture logic.
* `clr-qwen3-moe` must not own generic caching, CLI formatting, or OS-specific file policy.
* `clr-cli` must not contain tensor operations, inference algorithms, or cache implementation.
* Qwen-specific fields must not leak into generic `ModelConfig`.
* Do not add a new crate unless an existing boundary is demonstrably insufficient.

Prefer a small vertical slice over premature abstraction.

---

## 10. Model and artifact provenance

Every model-derived artifact must record:

* Model ID
* Exact model revision or commit
* Source URL or repository
* License
* Conversion command
* Tool versions
* Tensor names and shapes
* File hashes
* Artifact format version
* Generation date

All model downloads, conversions, validation runs, and cleanup operations must
follow `docs/temp-artifact-policy.md`, including one flat run directory,
read-only canonical inputs, preflight/post-task disk accounting, bounded debug
retention, and dry-run-first cleanup.

Do not commit large upstream model files.

Do not copy code from Colibri, llama.cpp, ik_llama.cpp, or katgpt-rs without:

* Checking its license
* Recording provenance
* Preserving required attribution
* Explaining why reimplementation is not preferable

Reference implementations are architectural and benchmarking references, not automatic code sources.

---

## 11. Windows x64 rules

Windows x64 with the MSVC Rust toolchain is the primary platform through M6.

The agent must:

* Use Windows-compatible paths and file semantics.
* Test file-handle and mapping lifetimes.
* Close files deterministically.
* Avoid assumptions based only on Linux behavior.
* Avoid shell commands requiring WSL unless explicitly designated as Python reference tooling.
* Use PowerShell commands in documentation intended for the primary workflow.
* Record Windows version, CPU, RAM, storage, Rust version, and commit in benchmark reports.

Linux compatibility is welcome but must not block Windows milestone completion.

---

## 12. Testing requirements

Every public contract and failure mode must have tests.

At minimum:

* Happy path
* Invalid input
* Boundary values
* Overflow or size errors
* Determinism
* Repeated execution
* Resource release
* Exact expert-selection checks
* Numerical comparison against the oracle
* Cache-budget enforcement where applicable

Tests must not depend on network access.

Tests must not silently download models.

Large or slow tests must be explicitly classified and documented.

Do not delete or weaken a failing regression test unless the specification changed and the reason is documented.

---

## 13. Required verification

Before closing any task group or milestone, run:

```powershell
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p clr-cli
```

Also run milestone-specific oracle, fixture, storage, memory, or benchmark checks.

A milestone cannot be marked complete when:

* Any required command fails
* Tests are ignored without explanation
* Clippy warnings remain
* Fixture hashes differ unexpectedly
* Numerical tolerance is undocumented
* Memory budget is exceeded
* Required evidence is missing

---

## 14. Commit policy

Create focused commits.

Recommended format:

```text
feat(core): add validated tensor shape contract
test(oracle): freeze tiny qwen3-moe router outputs
feat(storage): add byte-budgeted expert cache
fix(qwen3): match router normalization with oracle
docs(adr): record artifact format decision
bench(storage): compare buffered and mapped expert reads
```

Rules:

* One coherent concern per commit.
* Do not combine refactoring, feature work, formatting, and benchmark changes without necessity.
* Do not rewrite history after a milestone report has referenced a commit.
* Do not commit generated build output, secrets, large model weights, or local environment files.
* Keep the working tree clean before closing a milestone.

---

## 15. Documentation requirements

Documentation must explain both:

* What the code does
* Why the design was chosen

Create an ADR when changing:

* Public contracts
* Artifact format
* Tensor layout
* Cache ownership
* Quantization format
* Unsafe boundary
* File-access strategy
* Architecture dependency direction
* Model revision
* Numerical tolerance policy

ADR naming:

```text
docs/adr/0001-core-contract-boundaries.md
docs/adr/0002-tiny-qwen3-fixture.md
docs/adr/0003-expert-artifact-layout.md
```

Do not document unimplemented capabilities as if they already exist.

---

## 16. Progress reporting

At the end of each meaningful work session, produce a concise report containing:

```text
Completed:
- Tasks completed with IDs

Changed:
- Main files and contracts changed

Evidence:
- Commands run
- Test counts
- Oracle comparisons
- Benchmark or memory results

Open issues:
- Known failures or uncertainties

Next:
- Exact next task ID
```

Never report “complete” without evidence.

Do not hide failures or unresolved mismatches.

When blocked, identify:

* The failing task
* The exact error
* What was attempted
* The smallest decision needed from the user

---

## 17. Stop conditions

Stop autonomous implementation and request review when any of these occurs:

1. The implementation plan must materially change.
2. A public API used by multiple crates must be redesigned.
3. A model architecture assumption conflicts with the upstream reference.
4. Numerical output cannot match the oracle after a focused investigation.
5. A new unsafe boundary is required.
6. A new external dependency with significant impact is required.
7. Licensing or provenance is unclear.
8. An artifact format becomes incompatible with earlier fixtures.
9. Windows behavior differs materially from the design assumption.
10. A task would require implementing a deferred feature.
11. More than one milestone would need to be changed to resolve a problem.
12. A destructive operation or history rewrite is required.

Do not invent a workaround that silently changes requirements.

---

## 18. Allowed autonomous decisions

The agent may decide without asking:

* Private helper names
* Internal module organization within an approved crate
* Test case names
* Error-message wording that preserves error categories
* Minor refactoring local to one task
* Additional tests
* Documentation clarification
* Formatting and lint fixes
* Reverting an unsuccessful local optimization
* Recording newly discovered ideas in the backlog

The agent must not autonomously decide:

* To add a new architecture
* To add GPU support
* To change model target
* To change public artifact format after it is frozen
* To weaken numerical criteria
* To introduce broad unsafe usage
* To replace the Rust-first design
* To copy incompatible licensed code
* To skip a milestone
* To mark an incomplete task as done

---

## 19. Definition of success

The project is not successful merely because it builds.

M4 is successful only when:

* Qwen3-30B-A3B generates token IDs.
* The execution path is numerically validated.
* Expert weights can be loaded on demand.
* Resident memory remains within the configured byte budget.
* Cache and bytes-read metrics are emitted.
* The run is reproducible from documented model and artifact revisions.
* Windows x64 is supported.
* Known limitations are stated honestly.
* A baseline report identifies hardware, commit, model, revision, memory, I/O,
  and throughput.

Until these conditions are met, avoid describing the runtime as production
ready, optimized, or a replacement for established inference runtimes.

<!-- graft:start -->
## Graft — repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).
<!-- graft:end -->
