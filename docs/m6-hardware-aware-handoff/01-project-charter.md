# M6 Project Charter

## Current status pointer

M6.3 is active. D5 is closed `GO`, the R2.3 all-layer plan review and
implementation contract are frozen, and R2.3a is closed `PASS`. The exact next
task is `M6.3-R2.3b` per-layer characterization. See
`12-r2-3-current-state.md` and `13-r2-3b-codex-handover.md` before continuing
implementation.

## Objective

Maximize measured decode tokens/s for a supported Qwen3-MoE plan while staying
within declared RAM, VRAM, I/O, context, quality, and correctness limits.

## Non-negotiables

- `reference-f32-v1` remains executable and authoritative for differential
  validation.
- A low-memory plan is a compatibility option, not the automatic optimum.
- Resource budgets are enforced, observable, and reported separately from OS
  working-set measurements.
- No speed claim is accepted without repeatable end-to-end evidence.

## First success gates

| Gate | Minimum result |
| --- | --- |
| Correctness | Exact applicable router IDs and documented checkpoint/logit tolerances |
| Quality | Thai and English fixtures meet the approved comparison policy |
| Stability | No swap dependence; RAM/VRAM budget stays enforced |
| Performance | At least 0.5 decode tok/s for the first approved interactive plan |

`0.5 tok/s` is a product threshold, not permission to skip M6.3 gates.
