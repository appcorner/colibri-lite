# ADR 0015: Dense BF16-to-F32 Artifact Conversion

- Status: Accepted
- Date: 2026-07-14
- Milestone: M4.1
- Task: M4.1-05
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

The pinned Qwen3-30B-A3B source stores all tensors as BF16 across 16
Safetensors shards. The current correctness runtime computes in F32 and must
make all 435 non-expert tensors available for resident access without holding
an entire source tensor in conversion memory. Expert conversion remains
M4.1-06.

M2 already accepted artifact format v1: little-endian logical tensor ranges,
each with name, F32 shape, byte offset/length, and payload SHA-256. M4.1-05 can
reuse that contract with many logical tensors in one physical file. No new or
incompatible runtime artifact format is required.

## Decision

Convert each canonical non-expert source tensor from contiguous little-endian
BF16 to little-endian F32 in ascending tensor-name order. Each BF16 value maps
deterministically by shifting its 16 bits into the high half of an F32 bit
pattern. This preserves normal values, positive/negative zero, subnormals,
positive/negative infinity, and NaN sign/payload bits without arithmetic or
canonicalization.

The converter uses a 65,536-byte source chunk, a 131,072-byte F32 output chunk,
and during exact verification another 131,072-byte artifact chunk. Maximum
explicit conversion buffers are therefore 327,680 bytes. Tensor size does not
change this bound.

Before any BF16 payload range is decoded, the converter:

1. validates chunk size, source dtype, shape-derived element/byte count, shard
   index, offset/length arithmetic, and file range;
2. checks caller-observed free space against converted payload bytes plus a
   conservative deterministic-manifest reserve;
3. verifies the complete byte length and SHA-256 of every referenced source
   shard against pinned provenance.

Conversion writes `dense-f32.bin` and `dense-manifest-v1.json` through fixed
incomplete paths. It syncs both temporary files, atomically renames the payload,
and renames the manifest last as the commit marker. A transaction guard removes
temporary files and any already-renamed final payload if conversion or manifest
commit fails.

Every logical tensor remains independently described by artifact format v1;
the complete dense artifact happens to pack 435 ranges into one physical
payload. One tensor is one logical payload, not one required file.

## Source plan and manifest

`models/qwen3-30b-a3b/dense-source-plan-v1.tsv` is a deterministic,
dependency-free conversion plan generated from the pinned Safetensors headers
and cross-checked against the pinned index. It records model ID/revision, all
16 shard names/lengths/hashes, and 435 tensor names, shard indices, absolute
source offsets, lengths, and shapes. Its SHA-256 is
`23c165cfddb8e1b516ed6cf181fef573f8025e22e857457a2d1c81aeef645d0d`.

The emitted deterministic JSON manifest records:

- format version, model ID, and immutable revision;
- BF16 source and F32 artifact dtypes;
- little-endian byte order;
- physical artifact path, byte length, and SHA-256;
- every logical tensor name, shape, F32 offset/length/hash, and source
  shard/range.

This JSON is a deterministic conversion serialization around the already
accepted in-memory `ArtifactManifest`; this task does not add a runtime JSON
parser or change artifact format v1.

## Qwen validation

The Qwen wrapper validates every selected tensor through the M4.1-04 pinned
name/shape mapping before storage conversion. Expert gate/up/down roles fail
before I/O. Complete mode additionally requires all 435 expected dense names;
vertical-slice mode permits a non-empty reviewed subset.

No source tensor required transpose, repacking, or layout transformation.

## Public API surface

`clr-storage` adds generic safe conversion contracts:

- `DenseSourceShard`, `DenseSourceTensor`, and `DenseConversionSpec`;
- `DenseConversionSummary` and `DenseConversionError`;
- `decode_bf16`, `dense_conversion_preflight_bytes`, and
  `convert_dense_bf16_to_f32`;
- `DEFAULT_CONVERSION_CHUNK_BYTES`.

`clr-qwen3-moe` adds pinned role validation and orchestration:

- `Qwen3MoeDenseSourceTensor`, `Qwen3MoeDenseConversionSpec`, and
  `Qwen3MoeDenseConversionScope`;
- `Qwen3MoeDenseConversionError` and
  `convert_pinned_qwen3_moe_dense_tensors`;
- pinned model ID and revision constants.

`clr-core` and the accepted `ArtifactManifest` contract do not change. No
dependency or unsafe code is added.

## Evidence

The initial real vertical slice converted three pinned norm tensors from
verified shard 16. Both independent runs produced payload SHA-256
`d265721759f19bd8a62c1dd7746f841592e0e7a46827e884f9df58ccad649f4d`
and manifest SHA-256
`7112a40c15bd68122f32fe1fe38392cb2f115a524ab4808f3040d1442d1caf8c`.

The complete conversion verified 61,066,575,648 source-shard bytes before
decoding, converted all 435 dense tensors, and wrote 6,164,373,504 F32 bytes.
Two independent complete runs produced payload SHA-256
`bfad7fdc0f8611537ca5751f8e7140c35cf72bb4e67934b02d99711c89640893`
and manifest SHA-256
`aff0f4c59a71b98590bd2a1efc4e3fd6a95d58794063ce4b1092acce1597bc54`.

Streaming verification compared every written F32 element byte-for-byte with
direct BF16 decoding. Compact evidence is recorded in
`models/qwen3-30b-a3b/dense-conversion-evidence-v1.json`; large source and
converted artifacts are not committed.

## Excluded work

- No expert conversion, quantization, mmap, SIMD, FFI, native kernel, or
  tokenizer parsing.
- No full tensor is resident during conversion.
- No performance claim is made from the release-profile evidence run.
