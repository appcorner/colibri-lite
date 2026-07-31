# ADR 0052: Evidence-backed placement enumeration

## Status

Accepted for M6.2-03.

## Decision

`clr-core` enumerates a candidate only when every required profile status is
`measured`. Dense components may be placed in RAM or VRAM; experts may be
placed in RAM, VRAM, or SSD. A mixed RAM/VRAM placement additionally requires
measured host-to-device transfer evidence. Candidate IDs are formed
deterministically from backend ID and the two tiers.

`unavailable` and `not_run` are distinct non-admission states. The enumerator
does not reinterpret a detected device, a zero budget, or an absent benchmark
as support. It performs no budget admission, cost calculation, or ranking.

## Consequences

The M6.1 evidence currently supports only CPU RAM and SSD candidates. GPU
candidates remain absent until a reviewed backend has matching measured
compute, memory-access, and transfer evidence. M6.2-04 will evaluate the
enumerated candidates against RAM/VRAM/context budgets.
