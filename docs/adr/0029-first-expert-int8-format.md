# ADR 0029: First Candidate Expert INT8 Format

- Status: Accepted for M4.3-02 format definition
- Date: 2026-07-17
- Milestone: M4.3
- Task: M4.3-02
- Model revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

M4.3-01 froze the ordered Rust F32 path as the authoritative baseline. The
first quantization task must select a format from evidence without mutating the
122 GB F32 artifact or adding quantized runtime behavior.

Representative evidence covers gate, up, and down matrices for eight validated
expert cases across early, middle, and late layers. The comparison includes
per-tensor INT8, per-output-channel INT8, and input-group-128 INT8.

## Decision

Select `symmetric_int8_per_output_channel_f32_scale` as the first candidate for
future implementation:

- source and accumulation remain F32;
- weights are signed INT8 in row-major output-by-input orientation;
- each output row has an F32 scale `max(abs(row))/127`;
- quantization uses IEEE nearest-even rounding and clamps to `[-127,127]`;
- NaN and infinity inputs are rejected; zero rows reconstruct to zero;
- payloads and scales use little-endian deterministic serialization with
  64-byte alignment.

The first implementation will dequantize a complete selected projection into
F32 and call the existing scalar F32 operation. This is a correctness-first
contract, not an optimized kernel contract.

The additive artifact draft is
`colibri-qwen3-moe-expert-int8-v1`; it cannot overwrite or modify the canonical
F32 artifact. Its schema, hashes, ordering, and offsets are versioned separately.

## Evidence

Per-output-channel INT8 reached maximum representative weighted expert error
`0.1043548584`, compared with `0.4204292595` per-tensor and `0.0820999146`
group-128. Its modeled expert size is `4,733,280` bytes and full 6,144-expert
size is `29,081,272,448` bytes excluding the external manifest, a `4.200218x`
reduction from the F32 artifact. It retains 226 experts under a 1 GiB cache
and 7,259 under 32 GiB.

Group-128 is recorded as promising but requires more evidence because its lower
error costs 2.81% more bytes and adds group indexing complexity. Per-tensor is
rejected for its materially larger projection and expert errors.

## Provisional Contract

The selected candidate has provisional representative gates for reconstruction,
gate/up/activation/product/down/weighted outputs, structural identity, finite
values, F32-safe router IDs, vocabulary top-k, and generated IDs. These values
remain outside the F32 registry and must be recharacterized across Tier A/B
before implementation acceptance.

## Consequences

- No runtime, ExpertStore, cache, artifact, or numerical F32 contract changes
  in M4.3-02.
- Future implementation can validate dequantized matrices against this compact
  evidence before considering INT8 kernels or SIMD.
- A future group-wise format remains possible without changing this schema or
  the canonical F32 artifact.

## Supporting Records

- `docs/reports/m4.3-02-candidate-quantization-format.md`
- `models/qwen3-30b-a3b/m4.3-02-quantization-evidence-v1.json`
- `models/qwen3-30b-a3b/m4.3-02-format-spec-v1.json`
- `models/qwen3-30b-a3b/m4.3-02-quantized-expert-artifact-schema-v1.json`
- `models/qwen3-30b-a3b/m4.3-02-runtime-kernel-contract-v1.json`
- `models/qwen3-30b-a3b/m4.3-02-provisional-correctness-gates-v1.json`
