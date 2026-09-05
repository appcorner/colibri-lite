# ADR 0104: M6.3-R2.2-D4D1 Layer24 Down-Projection Precision Characterization

## Status

Pre-registered before D4D1 implementation or execution.

## Context

ADR 0103 closed D4 `NO-GO` because `short_english` Layer24 scalar hybrid versus same-input/router F32 local max-abs was `0.0010073595`, narrowly above the frozen `0.001` limit. D3 had selected Layer24 `down`-only group32 using only `short_thai`, so cross-fixture generalization was never established.

D2 already froze independent Layer24 group16 and group8 artifacts. D4D1 reuses those exact artifacts; it does not create a new representation family.

## Decision

Characterize Layer24 `down` only at group32, group16, and group8. Gate/up remain canonical F32. Use scalar packed arithmetic only.

Run both held-out fixtures under two frozen prefixes: canonical F32 and Layer0 group32. At Layer24, compute canonical F32 routed output and all three down-only hybrid candidates from the same post-norm input/router; do not propagate any target candidate beyond Layer24.

The local max-abs limit remains `0.001`. Selection is deterministic in `32 -> 16 -> 8` order: select the first group that passes all four fixture/prefix contexts; if none passes, select canonical F32 fallback.

No generation, performance timing, native-kernel change, new artifact, new group size, higher bit width, retry, or post-result threshold/policy change is authorized.

A passing D4D1 selection authorizes only a separately pre-registered held-out quality re-entry. It does not authorize D5, historical R2.2c, all-layer rollout, or M6.4.
