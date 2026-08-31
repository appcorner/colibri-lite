# ADR 0070: M6.3-R1.2 Held-Out Quality Validation

## Status

Accepted before any R1.2 candidate execution on the held-out fixtures.

## Context

ADR 0069 prospectively admits exactly one candidate to R1.2:
`cpu-safe-rust-int8-group32-layer0-r1-1a`. The historical R1.1
`no_candidate_admitted` record remains unchanged. R1.2 is a quality gate only;
it may not compare or reselect layouts and it does not authorize R1.3 or M6.4.

The frozen M6.0 quality bundle already pins `short_english` and `short_thai`
prompt token IDs, final-position top-20 IDs, greedy IDs, compact logits,
finite-count expectations, final-norm identity, and safe-margin router guards.
It does not contain a multi-token generated sequence.

## Decision

Use only the held-out `short_english` and `short_thai` fixtures. Before the
candidate sees either fixture, execute the unchanged `reference-f32-v1` path and
freeze a two-generated-token greedy sequence for each fixture. Two generated
tokens are the minimum sequence length accepted as the R1.2 multi-token gate.
The reference freeze must first reproduce the already frozen prompt-level
router guards, top-20 IDs, and greedy ID, so a newly generated sequence cannot
mask a changed F32 oracle.

After the F32 reference record is frozen, evaluate the admitted group-32
candidate twice independently on each fixture. The candidate must:

1. preserve the final prompt-position safe-margin router IDs at Layers 0, 24,
   and 47 exactly;
2. preserve the frozen prompt top-20 IDs exactly and the frozen greedy ID
   exactly;
3. keep the maximum observed prompt fixed/top-20 logit error at or below both
   the unchanged admission envelope `0.05` and one quarter of the fixture's
   frozen F32 top-1 margin;
4. reproduce the two-token frozen F32 greedy sequence exactly;
5. report finite Layer-0 candidate-vs-F32 stage error and keep direct packed
   consumption with zero complete F32 weight materializations; and
6. produce byte/digit-identical discrete outputs and identical numerical hashes
   across the two candidate executions for each fixture.

The candidate layout, quantization rule, artifact SHA-256, router, norms,
attention, Layers 1-47, KV-cache semantics, final norm, and LM head remain
unchanged. No dependency, GPU backend, SIMD path, FFI, mmap, cache-policy change,
or production default is introduced.

## Stop conditions

Any router-ID, top-20, greedy, multi-token, repeatability, finite-value, or
numerical-envelope failure is an R1.2 NO-GO. A failed held-out gate may not be
repaired by changing a threshold after observing candidate results.

R1.3 remains blocked until R1.2 closes with all gates passing. M6.4 remains
blocked independently by the all-layer review requirements.
