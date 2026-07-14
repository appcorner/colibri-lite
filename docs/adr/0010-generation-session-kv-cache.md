# ADR 0010: Generation Session and KV Cache Contract

- Status: Accepted
- Date: 2026-07-14
- Milestone: M3

## Context

Stateless resident and streaming forwards are correctness oracles. Cached
generation needs persistent sequence, sampling, and per-layer attention state
without changing those APIs or forking resident/streaming block math.

## Decision

`GenerationSession` will own sequence IDs, sampling state, current cache length,
and one shared `KvCache`. Separate resident and streaming constructors select a
backend; streaming borrows `ExpertStore` for the session while expert leases
remain scoped to one expert computation.

`KvCache` allocates once with separate contiguous F32 key/value buffers per
layer, each logically `[capacity, kv_heads, head_dim]`. `capacity` is immutable;
`len` counts initialized token positions. Byte size is calculated with checked
arithmetic across layers, key/value, capacity, heads, dimension, and F32 width.

One token append supplies K/V vectors for every layer. Capacity, layer count,
and vector lengths are validated before any copy. K/V bytes are copied for all
layers and only then is `len` incremented. Cache never resizes.

Prefill starts with an empty cache and processes token IDs in order. Incremental
attention uses the current cache length as position and attends to cached K/V
plus the current token's local K/V. The complete token succeeds before its K/V
is committed. Decode appends exactly one position.

## Public API surface

- `RuntimeError::ContextLengthExceeded` reports requested and fixed capacity.
- `KvCache::{new, capacity, len, is_empty, byte_size}` exposes fixed layout and
  accounting; append and per-layer views remain crate-private.
- `GenerationError` preserves resident/runtime versus streaming failure
  categories.
- `PrefillOutput` exposes per-token logits and exact per-layer expert IDs.
- `GenerationSession::{resident, streaming, prefill, decode_greedy,
  decode_temperature, sequence, cache}` defines session construction,
  execution, and observable state.
- `frozen_tiny_model` and `frozen_tiny_prompt` expose only the versioned tiny
  fixture required by the M3 CLI; they do not introduce a general model loader.

Existing `Qwen3MoeModel::forward` and `StreamingQwen3MoeModel::forward`
signatures and behavior remain unchanged.

## Invariants

- Stateless full-forward APIs remain unchanged.
- Cache layout and allocation are identical for resident and streaming sessions.
- `len <= capacity` and allocation capacity never grows.
- Overflow/capacity/shape failure does not mutate cache or sequence state.
- Successful prefill ends with `len == input_ids.len()`.
- Decode position equals cache length before append.
- Resident and streaming use shared attention, append, block, and sampling code.
- No paged cache, sliding window, KV quantization, dynamic growth, batching, or
  GPU cache is included.

## Required evidence

- Checked byte/layout and transactional overflow tests.
- Prefill logits/expert IDs equal stateless full forward.
- Cached multi-step decode equals recomputing generation.
- Resident and streaming cached paths agree.
- Same seed/temperature sequence is reproducible.
- Fixed allocations do not grow and repeated session create/drop is clean.
