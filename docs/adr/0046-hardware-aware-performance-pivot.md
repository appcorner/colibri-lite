# ADR 0046: Hardware-Aware Tokens-per-Second Pivot

## Status

Accepted. M6.0 is the next authorized task group.

## Context

M4 established a reproducible, correctness-valid F32 Qwen3-30B-A3B runtime.
M5 showed that storage-access variations did not establish repeatable
end-to-end runtime value. The project must no longer optimize for minimum RAM
alone.

## Decision

Optimize the supported runtime plans for measured tokens/s subject to explicit
RAM, VRAM, storage-I/O, context, stability, numerical-correctness, and model
quality constraints. Retain the frozen F32 execution as `reference-f32-v1`.
It remains executable and is the differential oracle; it is not the default
performance target.

The first implementation sequence is M6.0 reference/contracts, M6.1 measured
profiles, M6.2 planning, and exactly one M6.3 quantized vertical slice. GPU
and optimized backends are candidate implementations, not assumptions.

## Consequences

- Plans must state resource budgets, estimates, confidence, and rejection
  reasons.
- Hardware claims must use local measurement, not device names or datasheets.
- No all-layer quantized or accelerator runtime may begin before the one-layer
  slice passes its correctness, quality, and repeated end-to-end performance
  gates.
- M5 prototype decisions remain historical evidence and are not silently
  promoted to defaults.

## References

- `docs/m6-hardware-aware-handoff/`
- `docs/implementation-plan.md`
- `docs/tasks.md`
- `docs/adr/0033-m4.3-06-candidate-rejection-and-memory-pivot.md`
- `docs/adr/0041-m5.3-expert-access-prototype-selection.md`
