# Codex Handover: M6.3-R2.3b Per-Layer Characterization

## Mission

Continue `colibri-lite-rs` from closed R2.3a into **R2.3b only**. Build and
execute the frozen per-layer characterization workflow for the 45 unmeasured
MoE layers without extrapolating from Layer 0, 24, or 47.

Start from repository commit:
`82327280a6fb8ac6b15684c1e695e0bdd6f46238` or a later commit containing only
documentation/handover changes. Confirm the worktree is clean before changing
implementation files.

## Mandatory reading before editing

Read, in this order:

1. `docs/m6-hardware-aware-handoff/12-r2-3-current-state.md`
2. `models/qwen3-30b-a3b/m6.3-r2-3-implementation-contract-v1.json`
3. `docs/reports/m6.3-r2-3-implementation-protocol.md`
4. `models/qwen3-30b-a3b/m6.3-r2-3a-f32-four-token-reference-v1.tsv`
5. `docs/adr/0121-m6-3-r2-3a-four-token-f32-reference-pass.md`
6. `crates/clr-qwen3-moe/src/r2_2_d3_tests.rs`
7. `crates/clr-qwen3-moe/src/r2_2_d4d3_tests.rs`
8. `scripts/convert_m6_3_r2_2_d2_grouped_layer.py`
9. `docs/tasks.md` and `docs/implementation-plan.md`

Treat the JSON implementation contract as authoritative if prose and memory
appear to disagree.

## Frozen R2.3b scope

Layers: `1-23,25-46` only. Layer 0 and Layer 24 already have accepted policies;
Layer 47 is F32-locked.

Per layer, evaluate all 21 candidates:

- groups: `32`, `16`, `8`;
- subsets, in frozen tie-break order:
  `gate_up_down`, `gate_up`, `gate_down`, `up_down`, `gate`, `up`, `down`.

For every candidate require:

1. canonical-F32 control;
2. same-input local max-abs `<= 0.001`;
3. sequence-aware max-abs `<= 0.001` over both held-out fixtures with exactly
   one generated token of propagation;
4. finite outputs and deterministic identities;
5. exact artifact/source/layout/hash provenance.

Candidate selection per layer:

1. among candidates passing every local and sequence gate, maximize logical
   bytes saved;
2. tie -> larger group size;
3. tie -> frozen projection-subset order above;
4. no passing candidate -> canonical F32.

No automatic retry.

## Required implementation strategy

Preserve all accepted R2.2/D5/R2.3a harnesses. Prefer adding new files named for
R2.3b rather than editing historical evidence code.

Recommended decomposition:

### R2.3b-1 — machinery

- Add a deterministic R2.3b artifact-generation/orchestration layer that reuses
  the proven grouped-layer conversion grammar.
- Add an R2.3b Rust characterization harness that can select explicit
  `(layer, group, subset)` candidates and compare against canonical F32.
- Add a validator that enforces the frozen 21-candidate grid, both fixtures,
  limits, deterministic selection, and F32 fallback.
- Add focused synthetic/unit tests before any official candidate run.
- Run Clippy with `-D warnings`.

### R2.3b-2 — freeze execution identity

Before official characterization:

- freeze harness SHA-256;
- freeze builder/orchestrator SHA-256;
- freeze validator SHA-256;
- build one release test binary and freeze its bytes/SHA-256;
- bind canonical artifact root manifest SHA-256
  `f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`;
- bind R2.3a reference SHA-256
  `9f8de6841ff883c062c2ed8387dba93631c9de0565943f1de2bb1a27f0fada1e`;
- create and commit an execution manifest before official results are seen.

### R2.3b-3 — artifact/evidence execution

- Generate only the artifacts required by the frozen grid.
- Verify source length/SHA and output length/SHA.
- Rejected artifacts may be deleted only after builder/layout/source/output
  identity evidence is frozen as permitted by the contract.
- Record each candidate once; do not rerun because its result is unfavorable.
- If transport fails before a scientifically valid result exists, stop and
  document the failure before considering any explicit recovery.

### R2.3b-4 — result and closure

Produce a machine-readable result with exactly 45 layer decisions and evidence
for all 945 candidates or an explicit scientifically valid early-stop/failure
record if the contract requires stopping.

For each layer record at minimum:

- selected policy or `canonical_f32`;
- candidates evaluated/passed/failed;
- worst same-input error;
- worst sequence-aware error;
- logical bytes saved for selected candidate;
- selected artifact/layout identities;
- deterministic tie-break evidence.

Update:

- `docs/tasks.md`;
- `docs/implementation-plan.md`;
- `docs/work-log.md`;
- a new ADR and result report.

R2.3b completion may authorize **R2.3c only** if the frozen contract says so.
Do not begin R2.3c in the same change unless a separately committed closure
explicitly authorizes it.

## Prohibited work

Do not:

- reuse historical R2.2c/R2.2d;
- extrapolate policy by depth, neighborhood, or sentinel behavior;
- tune `0.001` after observing candidates;
- drop a difficult layer from the 45-layer matrix;
- reorder the frozen candidate tie-break to favor a result;
- add INT4/new precision families;
- implement native group16 or group8 kernels;
- alter cache policy;
- change the accepted D5 policy;
- claim physical SSD reductions;
- implement R2.3c/d/e/f or M6.4 before authorization.

## Verification expectations

At minimum before freezing official execution:

```powershell
cargo test -p clr-qwen3-moe --features full-model-validation,m6-3-r2-native --no-run --release
cargo clippy -p clr-qwen3-moe --all-targets -- -D warnings
git diff --check
```

Use narrower focused tests as they are added. Do not run a long official matrix
from a transient/uncommitted harness.

## Evidence discipline

- Hash every frozen contract, harness, builder, validator, binary, artifact, and
  result relevant to the decision.
- Do not rewrite accepted historical evidence files.
- Keep temporary artifacts outside the repo unless the contract says otherwise.
- Windows CRLF/mixed-EOL files exist in this repository. Avoid whole-file
  normalization of `docs/tasks.md` and `docs/implementation-plan.md`; use
  surgical edits and verify with `git diff --check`.
- Long official work should use durable execution and be polled from the same
  run root. Never silently relaunch a no-retry run.

## Ready-to-paste Codex prompt

```text
Work on colibri-lite M6.3-R2.3b only.

First read:
- docs/m6-hardware-aware-handoff/12-r2-3-current-state.md
- docs/m6-hardware-aware-handoff/13-r2-3b-codex-handover.md
- models/qwen3-30b-a3b/m6.3-r2-3-implementation-contract-v1.json
- models/qwen3-30b-a3b/m6.3-r2-3a-f32-four-token-reference-v1.tsv
- docs/tasks.md
- docs/implementation-plan.md

Treat the implementation contract as authoritative. Preserve accepted R2.2,
D5, and R2.3a evidence paths. Do not extrapolate layer policies.

Implement R2.3b per-layer characterization for layers 1-23 and 25-46. Every
layer must evaluate the frozen 21 candidates: groups 32/16/8 crossed with
subsets gate_up_down, gate_up, gate_down, up_down, gate, up, down. Require
same-input <=0.001 and sequence-aware bilingual <=0.001. Select maximum logical
bytes saved among candidates passing every gate; tie by larger group size, then
the frozen subset order. Otherwise select canonical F32. No automatic retry.

Reuse the existing R2.2 grouped-artifact builder and D3/D4D3 characterization
semantics where possible, but add R2.3b-specific files instead of modifying
frozen historical harnesses. Build focused tests and a strict validator first.
Do not launch official characterization until harness/builder/validator/release
binary hashes and an execution manifest are committed.

Before official execution run focused tests, release no-run build, Clippy -D
warnings, and git diff --check. Keep long runs durable and never relaunch a
no-retry scientific run without an explicit documented recovery decision.

At completion produce machine-readable evidence, one deterministic decision for
all 45 layers, ADR/report, task/implementation-plan/work-log updates, hashes,
commands/tests, open issues, and the exact next authorized task. Do not start
R2.3c or M6.4 unless a separately committed R2.3b closure authorizes it.
```

## Handover completion format expected from Codex

Return a concise report containing:

- status: PASS / FAIL / transport-blocked / partial;
- commits created;
- files changed;
- frozen SHA-256 identities;
- tests/build/Clippy results;
- official candidate count completed;
- selected non-F32 vs F32 layer counts;
- any failed/locked layers;
- evidence/result file paths and hashes;
- exact next authorized task;
- explicit statement that prohibited work was not performed.
