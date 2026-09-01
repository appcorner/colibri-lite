# ADR 0075: M6.3-R1.3 Prevalidated Artifact Identities

## Status

Accepted after the rejected v2 diagnostic and before any valid official R1.3 sample.

## Context

The ETW collector historically hashes every captured artifact after inference before writing `capture.json`. For an R1.3 sample this rereads nearly the full model outside the ETW window, materially warming the OS filesystem cache before the next paired sample.

That post-sample read is measurement infrastructure, not runtime behavior, and would bias later samples even though OS cache state is classified as uncontrolled.

## Decision

R1.3 configs carry an exact `artifact_identities` set. Canonical dense/expert identities come from the frozen `model-manifest-v1.json` SHA-256 `f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`; the admitted candidate uses its frozen R1.1a identity.

The collector verifies exact path-set equality, SHA text shape, file existence, and current byte length before launch. It records those prevalidated identities after capture and does not reread payloads to hash them. Legacy configs without identities retain the old post-capture hash behavior.

No runtime, cache, numerical, threshold, pair-order, or ETW timed-boundary rule changes.