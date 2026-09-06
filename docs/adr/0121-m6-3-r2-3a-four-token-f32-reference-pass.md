# ADR 0121: M6.3-R2.3a Four-Token F32 Reference PASS

## Status

Accepted. R2.3a closes `PASS` and authorizes R2.3b per-layer candidate characterization only.

Result SHA-256:
`8a5c0f254e0d9b36bc8556ed9e4330dc4896f12a3767f2525c9fd04a63790203`.

Reference SHA-256:
`9f8de6841ff883c062c2ed8387dba93631c9de0565943f1de2bb1a27f0fada1e`.

## Context

ADR 0120 requires an extended canonical-F32 bilingual reference to freeze before any of the 45 unmeasured layers may execute a packed candidate. The reference extends the accepted R1.2/D5 semantic anchor from two generated tokens to four while preserving the existing prompt guard identities.

## Decision

The one-shot canonical-F32 execution completed successfully: test runtime `438.91s`, durable runtime `439.892212s`, exit code `0`, and empty stderr.

The frozen generations are:

- English: `[0, 358, 2776, 264]`
- Thai: `[7360, 91, 16, 15]`

The test directly anchors the first two generated IDs, prompt argmax/top-20, Layers 0/24/47 router guards, prompt-logit SHA-256, and final-norm SHA-256 against the previously frozen R1.2 reference. Every anchor is exact.

The new TSV also freezes prompt top-20 logit values, fixed-logit indices/values, router guards, and full prompt-logit/final-norm hashes for both fixtures.

## Consequences

R2.3b may begin the contract-defined per-layer characterization of Layers `1-23,25-46`. No layer may skip the 21-candidate direct-evidence grid, no cumulative all-layer candidate is authorized yet, and R2.3c-R2.3f plus M6.4 remain blocked behind their ordered gates.
