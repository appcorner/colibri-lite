# ADR 0114: M6.3-R2.2-D5b Production-Path Quality GO

## Status

Accepted. D5b closes `GO` and authorizes D5c paired performance/resource measurement only.

Result SHA-256:
`0c742147f967583a735a5a0d8a9dcc2482878204fcf257507532839d40a3872e`.

## Context

ADR 0113 closed D5a with a production-like Layer24 artifact containing F32 gate/up plus packed group8 down and a direct reader with no F32 down buffer. D5b reruns the frozen D4D4 bilingual quality gates through that production path.

The one-shot durable execution completed with exit code `0`, test runtime `745.56s`, zero stderr, and evidence SHA-256 `a330462130baa7b06b52122ce5851ffe412ff6496ade909b418c370d04767a6b`.

## Decision

The D5b evidence is byte-identical to the accepted D4D4 evidence. English remains `[0, 358]`; Thai remains `[7360, 91]`. Prompt top-20 ordering, argmax, router guards, finite logits, Layer0/Layer24 local budgets, native/scalar budgets, compact F32 logit envelope, and exact mixed-run repeatability all pass unchanged.

D5c paired performance/resource measurement is authorized under the frozen D5 contract. Historical R2.2c/R2.2d, threshold retuning, new precision, native group8 work, all-layer rollout, and M6.4 remain unauthorized.
