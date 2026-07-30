# Planner Cost Model

The planner evaluates only backends that have a matching measured profile.
For every candidate it estimates prefill/decode time from measured compute,
transfer, and storage terms; it also reports RAM, VRAM, expert-cache capacity,
disk bytes/token, startup cost, context capacity, and quality risk.

Candidates violating an explicit budget are rejected with a machine-readable
reason. The ranking is deterministic: feasible candidates sort by estimated
decode tokens/s, then lower quality risk, then lower startup cost, then stable
plan identifier.

The initial analytical model must report prediction error against recorded
benchmarks. It must never disguise an unmeasured component as a measured fact.
