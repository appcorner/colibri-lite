# ADR 0007: Streaming Qwen Expert Payload

- Status: Accepted
- Date: 2026-07-14
- Milestone: M2.2

## Context

The resident M1 model is the numerical oracle and must remain stable. The
storage-aware path needs dense/router weights resident while loading only
routed experts under the cache budget. One logical expert must remain
independent of physical file placement so later artifacts may use shards.

## Decision

`StreamingQwen3MoeModel` is a separate public path; the resident
`Qwen3MoeModel` API remains unchanged.

Each expert is one logical F32 little-endian payload ordered:

1. gate matrix `[moe_intermediate, hidden]`;
2. up matrix `[moe_intermediate, hidden]`;
3. down matrix `[hidden, moe_intermediate]`.

`PackedExpertLayout` records every offset and length, dtype, byte order, and
total byte length. `TensorMetadata` independently records the logical payload's
shard-relative path, offset, length, dtype, endianness at manifest level, and
hash. No contract requires one expert per file; tests store all experts in one
sharded container with distinct ranges.

The streaming model retains embeddings, normalization, attention, router, and
LM-head weights. After routing it loads only selected `ExpertKey` payloads from
`ExpertStore`. The `ExpertLease` remains in lexical scope through decode and
all token computations for that expert. Payload slices are private borrows from
`lease.bytes()` and cannot outlive the lease.

Resident and streaming paths share:

- RMSNorm, attention, RoPE, routing, and residual functions;
- the expert-order/routing-weight combination loop;
- the scalar gate/up/`SiLU`/down `expert_mlp` function.

Only expert weight acquisition differs.

## Invariants

- Resident M1 API and oracle tests remain unchanged.
- Packed payload uses exact little-endian F32 round-trip.
- Payload ranges are config-derived and total length is validated before decode.
- Only routed experts are requested.
- A lease remains alive for the complete expert computation.
- Streaming uses the same numerical tolerance as M1.
- Resident bytes never exceed the configured cache budget.
- Malformed, truncated, hash-invalid, or oversized payloads fail before expert
  computation returns output.

## Evidence

- Every packed expert payload round-trips exact F32 bytes.
- Resident and streaming paths match every block stage, exact router expert IDs,
  expert outputs, block outputs, and final logits.
- A two-expert budget forces 8 misses, 8 loads, 6 deterministic evictions, zero
  hits, exact bytes-read, and peak/current resident bytes of two payloads.
- Existing cache tests prove a live lease prevents eviction while active.
- Payload larger than budget returns `ExpertExceedsBudget`.
- Corrupted and truncated shard payloads return structured hash/truncation
  errors before computation.
- All 59 workspace tests and standard verification commands pass without
  loosening M1 tolerances.
