# ADR 0120: M6.3-R2.3 Implementation Contract

## Status

Pre-registered before R2.3 implementation.

Contract SHA-256:
`97dd33b5cdb576d1ebcf9b3b7661f5462c912be40c7485c7e6bd7f8773d7771b`.

## Context

ADR 0119 closes the R2.3 all-layer plan review `GO` for implementation-contract design only. The review explicitly classifies all 48 layers while leaving 45 unmeasured layers in canonical F32. It also freezes the principle that precision and projection tolerance are layer-local evidence, not depth interpolation.

## Decision

R2.3 implementation is split into six ordered gates: freeze an extended canonical-F32 quality reference, characterize every unmeasured layer, cumulatively admit independently passing layers with deterministic interaction fallback, freeze production layouts and re-enter full quality, run a fresh three-path paired performance/resource proof, then hold a final decision review.

Per-layer characterization evaluates the complete 7-projection-subset x 3-group-size grid (`21` candidates) for every unmeasured layer. Group32/group16/group8 are independent hypotheses; finer grouping is never assumed to improve quality. Any layer without a passing direct candidate stays F32.

Cumulative admission ranks selected layer candidates by worst sequence-aware local error divided by logical bytes saved, tie lower layer ID, then adds fixed batches of four. A failed batch is bisected deterministically; an individually failing layer reverts to F32 and is locked out for the contract. There is no threshold retuning or alternate-precision retry after cumulative failure.

The final performance proof compares three paths: canonical F32, the already validated D5 baseline, and the final R2.3 candidate. Each fixture/cache cell runs all six path permutations once, yielding 72 fresh processes total. GO requires at least `1.0%` median wall speedup over D5 in each fixture, at least `5/6` candidate wins over D5 and over F32 in every cell, no more than `1.0%` working-set/private-byte regression versus D5, and lower logical expert bytes than D5 in every triplet.

Physical SSD attribution remains outside this contract unless a separate Windows telemetry contract is pre-registered and revalidated.

## Consequences

Once this contract is frozen, R2.3a may begin by generating and freezing the canonical-F32 four-token bilingual quality reference. No later phase may run before the immediately preceding gate closes. A final R2.3 `GO` would authorize only a separate M6.4 entry review; it would not automatically start M6.4 or production rollout.
