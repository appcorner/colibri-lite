# Codex Implementation Prompts

Use a focused prompt per task: state the exact task ID, frozen reference
artifacts, allowed scope, acceptance criteria, required commands, evidence
outputs, and explicit stop conditions. Require a concise evidence report with
completed work, changed contracts, commands/tests, hashes, open issues, and the
exact next task.

For the current phase, Codex must read these files before editing:

1. `docs/m6-hardware-aware-handoff/12-r2-3-current-state.md`
2. `docs/m6-hardware-aware-handoff/13-r2-3b-codex-handover.md`
3. `models/qwen3-30b-a3b/m6.3-r2-3-implementation-contract-v1.json`
4. `models/qwen3-30b-a3b/m6.3-r2-3a-f32-four-token-reference-v1.tsv`
5. `docs/tasks.md` and `docs/implementation-plan.md`

## Current prompt

The ready-to-paste Codex prompt is maintained in
`13-r2-3b-codex-handover.md`. Do not shorten away the prohibitions or replace
the frozen R2.3b selection rule with heuristic layer/depth interpolation.

Codex may implement only `M6.3-R2.3b` unless a later committed decision
explicitly authorizes R2.3c. R2.3c, R2.3d, R2.3e, R2.3f, M6.4, new
quantization families, native group16/group8 kernels, and cache-policy changes
remain out of scope.
