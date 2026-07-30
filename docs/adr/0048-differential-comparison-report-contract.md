# ADR 0048: Differential Comparison Report Contract

## Status

Accepted for M6.0-03.

## Context

M4 provides a frozen F32 baseline and a tolerance registry with different
scopes. Candidate backends need a deterministic diagnostic that preserves
those scopes instead of substituting a single broad tolerance.

## Decision

`clr-core` provides ordered `StageComparison` inputs and a
`DifferentialReport`. Each stage owns its documented `ComparisonTolerance`
(absolute term, relative term, and source identifier). The comparator checks,
in order:

1. tensor metadata;
2. shape-derived reference and candidate value lengths;
3. finite reference and candidate values;
4. `absolute + relative * abs(reference)`.

It stops at the first divergence and records stage index/ID, failure category,
flat element index when applicable, values, absolute/allowed error when
applicable, and the exact tolerance used. Empty comparison sequences and
invalid tolerance definitions fail structurally.

## Consequences

The generic comparator has no model, artifact, storage, backend, or JSON I/O
knowledge. Callers select stages and import their exact tolerance values from
the frozen registry. Router-ID, semantic-margin, and quality comparisons remain
separate gates because they are not reducible to elementwise floating-point
comparison.
