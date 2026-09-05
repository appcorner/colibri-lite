# ADR 0096: M6.3-R2.2-D1 quality-drift localization

- Status: Accepted
- Date: 2026-09-05
- Milestone: M6.3-R2.2-D1

## Context

ADR 0095 closes R2.2b `NO-GO`. The first valid failure is `short_thai` exact
prompt top-20 ordering on the scalar group32 run with Layers 0/24/47 active.
Exact generated IDs had already matched, and the observed top-20 token set is
unchanged. Native validation was not reached.

The next question is therefore not performance and not native-kernel tuning. It
is whether one frozen group32 sentinel is sufficient to cause the reorder or
whether the reorder emerges only after errors accumulate/interact across depth.

## Decision

Run a scalar-only prompt diagnostic on `short_thai` with exactly eight frozen
states, in this order: `[]`, `[0]`, `[24]`, `[47]`, `[0,24]`, `[0,47]`,
`[24,47]`, `[0,24,47]`. `[]` is the canonical F32 control. Selected sentinel
layers use their already frozen group32 artifacts; every other layer uses
canonical F32 experts.
Each state performs one deterministic prompt forward only; no generation and no
timing are part of D1. Record prompt top-20 IDs, exact-order match, token-set
match, argmax, Layers 0/24/47 router guards, frozen fixed/top-20 logit errors,
and the four swapped-token logits/ranks (`94482`, `69440`, `52388`, `129075`).
At each active sentinel, also compute a same-input/router canonical-F32 routed
expert comparator and record scalar-group32 local max-abs before continuing the
candidate path.

A minimal-cardinality failing subset identifies the first sufficient placement.
If every singleton passes but a pair/triple fails, classify accumulation or
interaction rather than a single-layer cause. No result may alter the frozen
R2.2 gate, authorize performance, or select/requantize an artifact.

## Consequences

D1 is diagnostic evidence only. One run per state is frozen; no results-based
retry or subset reselection is allowed. After D1, a separate reviewed hypothesis
is required before any quality re-entry implementation. R2.2c/R2.2d, R2.3
all-layer work, and M6.4 remain blocked.
