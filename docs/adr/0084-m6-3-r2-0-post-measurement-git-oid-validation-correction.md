# ADR 0084: M6.3-R2.0 Post-Measurement Git OID Validation Correction

## Status

Accepted after all 40 R2.0 localization samples completed and before any
localization classification or R2.1 authorization.

## Context

The immutable localization sample set completed successfully with SHA-256
`e38688ae1309c411c11309def37b5b2ef6e84eb634dee244bb565a41b7eb7942`.
The final validator then rejected the document with `INVALID: instrumented
commit` before inspecting/classifying the sample matrix.

The frozen execution manifest records the source commit as
`b4cb158b5cfbaf7ac6a8dae44b929d02f14ae004`, a valid 40-hex Git SHA-1 object
ID. The validator incorrectly reused its SHA-256 helper and therefore required
64 hex characters.
## Decision

Allow one post-measurement validation-only correction:

- accept a lowercase hexadecimal Git object ID of either 40 or 64 characters;
- keep every measurement sample byte-for-byte unchanged;
- keep observer controls, fixtures, pair order, timing scopes, classification
  thresholds, and all R2.0 gates unchanged;
- rerun classification only on the immutable 40-sample document;
- record both the frozen validator SHA and corrected validator SHA in closure
  evidence.

No benchmark process may be rerun to replace or improve any sample.

Machine-readable correction contract:
`models/qwen3-30b-a3b/m6.3-r2-0-validation-correction-v1.json`, SHA-256
`d847abbc918b67fe457ce76b7025c92977ea9494eae795bd0cb2b2e8dbad5981`.

## Frozen evidence

- localization samples SHA-256:
  `e38688ae1309c411c11309def37b5b2ef6e84eb634dee244bb565a41b7eb7942`
- failed-run record SHA-256:
  `1598196120dc6db7392dae1c7847c4ef55f3839df0a99515f679ad691f75789e`
- frozen validator SHA-256:
  `0b94b821d068e22129d1a0901e5a8ed0dbafb23f8e4b478d5ea8a1648b49e819`
- sample count: `40`
- classification seen before correction: `false`
- R2.1 authorization before correction: `false`
- M6.4 remains blocked.