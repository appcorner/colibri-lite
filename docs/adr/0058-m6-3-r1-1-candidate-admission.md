# ADR 0058: M6.3-R1.1 Deterministic Layer-0 Candidate Admission

## Status

Proposed. This ADR defines the admission protocol only. It does not accept a
candidate, reopen `cpu-safe-rust-int8-group128-layer0-v1`, authorize R1.2, or
authorize M6.4.

## Context

ADR 0057's group-128 candidate is permanently stopped by M6.3-06. Its English
fixed-logit drift was material and it has no candidate-specific admission rule.
It is therefore excluded by identifier and cannot be a default, a baseline, or
a fallback in this study.

The study considers two new, mutually exclusive Layer-0 layouts: symmetric
INT8 with F32 scales per output-row/input group, using input-column groups of
64 or 32. Both retain row-major little-endian packed values, 64-byte-aligned
projection offsets, round-half-away-from-zero, and saturation to `[-127,127]`.
The normal path must read packed values and scales inside the scalar F32
accumulation order; it must not reconstruct a complete F32 projection or
expert. Router, norms, routing weights, residuals, softmax, activations, and
accumulation remain F32.

## Decision

Use the versioned admission record and validator at
`models/qwen3-30b-a3b/m6.3-r1-1-candidate-admission-v1.json` and
`scripts/validate_m6_3_r1_1_candidate_admission.py`. The selection algorithm
is deterministic: validate every candidate against pre-registered gates;
admit exactly one only if it passes all gates; otherwise record
`no_candidate_admitted`.

The characterization set is fixed before conversion (`tier_a_control` and
`tier_b_code_newline`). `short_english` and `short_thai` are held out for
R1.2, not used to choose a candidate. The pre-registered logit cap is `0.05`,
which is below one quarter of the smallest frozen held-out top-1 margin
(`0.20189857482910156 / 4`). A candidate must additionally justify its actual
envelope from the characterization set. Delta-NLL, top-k overlap, and endpoint
agreement are supplementary evidence only.

Required gates are exact source/output hash, offset/alignment, tensor name,
shape, dtype, and finite checks; byte-identical independent reconversion;
three deterministic candidate executions; exact IDs at safe router margins
and explicit ambiguity at near ties; and Layer-0 first-divergence reporting.
Final top-20, greedy, and multi-token checks are predeclared gates for R1.2;
they cannot be used here to promote a format.

## Dependency, license, and unsafe review

No Cargo dependency, external code, FFI, SIMD, GPU backend, mmap, async
prefetch, thread pool, cache-policy change, or unsafe code is added. The
workspace unsafe prohibition remains active. The source remains the pinned
Apache-2.0 `Qwen/Qwen3-30B-A3B` revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`; no TurboFieldfare, Colibri,
llama.cpp, or ik_llama layout/code is used.

## Current evidence outcome

The required read-only canonical Layer-0 F32 shard is not present in this
workspace. Downloading or substituting a payload is outside this task, so no
conversion, characterization, or final fixture execution occurred. The valid,
honest result is `no_candidate_admitted`. If the canonical input is later made
available, a conversion must follow the temporary-artifact policy: disk
preflight, one flat unique run directory, `.incomplete` output, atomic
promotion, hashes/provenance/post-task accounting, and cleanup.
