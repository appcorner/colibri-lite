# ADR 0049: Profile CLI JSON composition

## Status

Accepted for M6.1-05.

## Decision

The doctor and profile-model CLI commands compose versioned JSON profiles from
explicit, tracked evidence paths. Callers provide the output timestamp and
runtime commit, so a given set of inputs produces reproducible bytes rather
than silently sampling the host or downloading model data.

The commands use serde_json version 1.0.149 only in clr-cli. The standard
library has no general JSON parser; this maintained MIT/Apache-2.0 crate avoids
a fragile ad-hoc parser for versioned evidence. It supports the workspace MSRV
and Windows x64. clr-core remains dependency-free and neither command adds
unsafe code, a GPU backend, or a model payload dependency.

## Consequences

Doctor exposes only measured M6.1 evidence and preserves unavailable/not-run
backend states. Profile-model describes the frozen F32 model and quality
reference without authorizing quantization or a backend.
