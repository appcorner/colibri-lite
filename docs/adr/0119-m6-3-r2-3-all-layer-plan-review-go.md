# ADR 0119: M6.3-R2.3 All-Layer Plan Review GO

## Status

Accepted. R2.3 all-layer plan review closes `GO` for implementation-contract design only.

Result SHA-256:
`c525cfc8ed536fb7d0033104f41e5e2f64a0d9a2842c4fcddba5b32dfac7e6e2`.

## Context

ADR 0118 pre-registered a design-only review after D5c proved a production-like end-to-end benefit for the validated Layer0/Layer24 policy. Existing depth evidence forbids treating Layers 0, 24, and 47 as a precision interpolation rule for the remaining 45 layers.

The review therefore starts from an explicit 48-layer map rather than a guessed all-layer quantization plan.

## Decision

The review satisfies every frozen deliverable and closes `GO`.

- Layer 0 remains the already validated full group32 native policy.
- Layer 24 remains the already validated F32 gate/up plus scalar group8 down policy.
- Layer 47 remains F32-locked by existing depth/projection evidence.
- All other 45 layers are explicitly `unmeasured_default_f32` and cannot inherit a non-F32 policy without direct layer evidence.

Future characterization uses a deterministic ladder: F32 control, complete group32 diagnostic, the full seven D3 projection subsets when needed, independent group32/group16/group8 precision checks, lowest-byte passing selection or F32, then sequence-aware validation. Finer groups are never assumed to be more accurate.

Independent layer PASS is not sufficient. Eligible layers are ranked by worst sequence-aware local error divided by logical bytes saved, tie lower layer ID, then admitted cumulatively in batches of four. A failed batch is bisected by ascending layer ID; a single layer that fails cumulative admission reverts to F32 and is locked out for that contract. No threshold or precision retry is permitted after cumulative failure.

The impact model deliberately does not invent an all-layer speedup. The measured lower anchor remains D5: `3.4380%` English and `4.0296%` Thai median wall speedup with `2,239,758,336` bytes (`1.931423611111%`) exact expert-payload storage reduction from the validated policies. A central speedup/storage projection is not identifiable before direct characterization of the 45 unmeasured layers. The `69.314236111111%` storage-only arithmetic ceiling is explicitly not a candidate, quality, or speedup projection.

## Consequences

R2.3 may proceed only to design an exact implementation contract for the reviewed characterization/admission workflow. This ADR does not authorize artifact generation, per-layer conversion, runtime implementation, official all-layer performance timing, a 48-layer rollout, or M6.4. Physical SSD claims remain blocked until Windows telemetry is separately revalidated.
