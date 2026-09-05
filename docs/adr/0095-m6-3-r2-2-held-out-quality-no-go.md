# ADR 0095: M6.3-R2.2 held-out quality NO-GO

- Status: Accepted
- Date: 2026-09-05
- Milestone: M6.3-R2.2b

## Context

R2.2 pre-registered a three-sentinel proof on Layers 0/24/47 using the unchanged
R2.1 group32 representation and AVX2+FMA kernel. Official performance was
explicitly blocked until held-out quality passed.

Execution ordinal 4 used the frozen execution-v3 identities and corrected
durable transport. The native test exited `101` after `793.34s`, and the
failure is therefore a valid quality verdict rather than a transport loss.

## Decision

R2.2b closes **NO-GO**. The first failing gate is exact prompt top-20 ordering
for `short_thai`. The failure occurs while validating the scalar-group32 run,
before the native run is validated, so current evidence implicates the packed
representation/placement rather than the AVX2 kernel.
The observed top-20 token set is unchanged, but two adjacent rank pairs swap:
`94482 <-> 69440` and `52388 <-> 129075`. Exact two-token generation had
already passed for that scalar run before the top-20 assertion failed. Later
argmax/router/native-repeatability gates were not reached and are not inferred.

Result artifact SHA-256:
`2b8c841daf9318fcd2b9d08ffdb03dbeef541195ae9f76090ea72e2542bf96be`.
Raw ordinal-4 preflight/stdout/stderr/completion/exit evidence is preserved in
the model evidence directory with hashes recorded in the result artifact.

## Consequences

R2.2c official 108-sample performance measurement is not authorized. R2.2d,
R2.3 all-layer rollout/implementation, and M6.4 remain blocked.

The only authorized next activity is diagnostic localization of the quality
drift with the frozen group32 artifacts. The diagnostic must not relax the
original R2.2 quality contract or relabel diagnostic evidence as a quality PASS.
A new hypothesis/plan review is required before any quality re-entry attempt.
