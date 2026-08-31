# ADR 0069: M6.3-R1.1d Group-32 Admission Amendment

## Status

Accepted. This is a prospective amendment only. It does not rewrite the completed M6.3-R1.1 record, which remains `no_candidate_admitted`, and it does not authorize R1.3 or M6.4.

## Context

R1.1 rejected group-64 and group-32 because neither had Layer-0 routed-expert characterization, checkpoint/first-divergence evidence, router execution evidence, or a justified logit envelope. R1.1a was later approved as append-only remediation. ADR 0060 bound that remediation to the frozen `code_newline` held-in fixture, and ADRs 0064-0067 fixed the fixture-scoped tolerance and telemetry contracts before accepted candidate runs.

Both candidates subsequently completed three valid deterministic characterization runs. ADR 0068 applied the locked ADR 0059 ranking and selected group-32 as the characterization winner because its fixed-logit maximum absolute error `2.36730575561523438e-2` is lower than group-64's `2.37542390823364258e-2`.

## Decision

Admit `cpu-safe-rust-int8-group32-layer0-r1-1a` prospectively for M6.3-R1.2 only.

The admission envelope remains the already pre-registered `0.05` fixed-logit maximum-absolute cap. It is not tightened or loosened after characterization. Group-32's observed held-in error is below this cap. Exact Layer-0 router IDs, named finite checkpoints, first-divergence localization, direct packed consumption, zero complete F32 weight materializations, and three-run deterministic repetition are required evidence and are satisfied by the closed R1.1a set.

The historical R1.1 JSON and decision remain immutable. This ADR creates a new amendment record rather than changing the old selection outcome.

## R1.2 boundary

R1.2 is now authorized to evaluate only the already selected group-32 candidate against `reference-f32-v1`. It must use the held-out `short_english` and `short_thai` fixtures and the existing stage-level, safe-margin router, fixed top-20, greedy, multi-token, and repeatability gates. R1.2 must not compare layouts or reselect a candidate.

R1.3 remains blocked until R1.2 closes successfully. M6.4 remains blocked regardless of R1.2 until the later re-entry review and a separately approved all-layer plan satisfy the implementation-plan gates.

## Consequences

- R1.1 historical outcome remains `no_candidate_admitted`.
- R1.1a remains an append-only characterization result.
- Group-32 becomes the single candidate admitted to R1.2.
- Group-64 is not selected by the locked ranking and is not a fallback.
- Group-128 remains permanently stopped.
- No runtime, artifact, cache, dependency, unsafe, SIMD, GPU, or production-default change is made by this ADR.
