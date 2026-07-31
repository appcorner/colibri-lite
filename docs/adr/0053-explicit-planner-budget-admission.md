# ADR 0053: Explicit planner budget admission

## Status

Accepted for M6.2-04.

## Decision

`clr-core` admits each enumerated candidate against the caller's exact RAM and
VRAM byte budgets plus requested context length. It compares calculated RAM and
VRAM requirements using inclusive limits, and compares requested context to the
candidate's maximum context capacity.

Admission returns either `Admitted` or every exceeded constraint in the stable
order RAM, VRAM, context. Each rejection contains the candidate's stable plan
ID, a machine-readable code, calculated requirement, and applicable limit.
No profile recommendation may override these requested limits.

## Consequences

M6.2-05 can serialize these rejections directly into the planner-result
contract. Exact-boundary plans remain feasible; a zero budget explicitly
forbids non-zero demand in that tier. This layer performs no candidate
enumeration, throughput ranking, or environment observation.
