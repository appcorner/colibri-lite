# ADR 0071: M6.3-R1.2 Quality Gate Result

## Status

Accepted. M6.3-R1.2 closes with `GO` for the admitted Layer-0 group-32 candidate only.

## Evidence

ADR 0070 was accepted before held-out candidate execution. Its pre-registered contract hash is `571c6b745d1bb809eef270e87647e699e911df1905a2ddd1251b08356cc842d1`.

The unchanged F32 path first froze two-token greedy references before candidate execution:

- `short_english`: `[0, 358]`
- `short_thai`: `[7360, 91]`

The admitted artifact was freshly reconverted from the canonical Layer-0 F32 shard and independently matched the admitted identity: 679,477,248 bytes and SHA-256 `777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2`.

The held-out candidate test passed. Both fixtures preserved exact prompt top-20 IDs, exact prompt greedy ID, exact safe-margin router IDs at Layers 0/24/47, the frozen two-token sequence, finite Layer-0 candidate-vs-F32 error, and exact repeatability across two executions per fixture.

Observed maximum prompt logit errors were `3.3779144287109375e-3` for `short_english` and `7.321834564208984e-3` for `short_thai`, both below the locked effective cap `5.000000074505806e-2`. Layer-0 MoE maximum absolute errors were `8.989572525024414e-4` and `8.705407381057739e-4`, respectively.

## Decision

R1.2 is `GO`. R1.3 is authorized to perform paired F32/candidate release-process resource and I/O measurements. Candidate reselection remains prohibited. M6.4 remains blocked by the existing all-layer review and full-runtime-benefit gates.

## Scope and limitation

This result validates only the Layer-0 group-32 vertical slice. It is not evidence of all-layer quality, end-to-end speedup, working-set improvement, or production readiness. The two repeat executions use fresh inference/KV/F32-store state; the validated candidate reader file handle is reused only as a seek-based byte source with cumulative telemetry and contains no cache or numerical state that can affect outputs.
