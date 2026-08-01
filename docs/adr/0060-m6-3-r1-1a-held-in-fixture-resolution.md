# ADR 0060: M6.3-R1.1a Held-in Fixture Resolution

## Status

Accepted before R1.1a execution. This ADR supersedes only the held-in fixture
binding in ADR 0059; its candidate list, ranking order, and `0.05` cap remain
unchanged.

## Decision

Use the existing frozen Tier-B `code_newline` fixture as the sole canonical
held-in fixture for M6.3-R1.1a. Its existing input IDs, Layer-0 guard router
IDs, fixed-logit indices/values, and F32 reference evidence are the only
permitted characterization reference in this task.

`tier_a_control` is deferred to a separate task that first creates and freezes
its own reference input, router, intermediate checkpoint, and fixed-logit
contract. It is not silently mapped to a generation trace, profile label, or
another fixture.

## Consequences

R1.1a remains limited to Layer-0 characterization and may not run
`short_english`, `short_thai`, any held-out bilingual fixture, R1.2, or M6.4.
The decision does not alter the stopped R1.1 result or make a full-model
quality claim.
