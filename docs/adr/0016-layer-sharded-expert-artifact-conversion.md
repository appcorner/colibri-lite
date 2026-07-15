# ADR 0016: Layer-Sharded Expert Artifact Conversion

- Status: Accepted
- Date: 2026-07-15
- Milestone: M4.1
- Task: M4.1-06
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

Qwen3-30B-A3B contains 6,144 logical experts across 48 decoder layers. Each
expert has gate, up, and down BF16 source tensors, for 18,432 expert tensors in
the pinned Safetensors inventory. The existing runtime contract identifies an
expert by layer and expert index and consumes one packed F32 payload in gate,
up, then down order through `ExpertStore`, `ExpertLease`,
`ExpertWeightsView`, and the shared scalar numerical implementation.

M4.1-06 must preserve that execution path while avoiding one physical file per
source tensor or logical expert. It must also permit one expert to be read
without reading unrelated expert payloads.

## Decision

Artifact format version 1 continues to describe each logical expert as an
independent tensor range. The complete conversion uses 48 physical containers,
one per layer, with 128 consecutive logical expert ranges per container. Each
expert range is 18,874,368 bytes and packs these little-endian F32 projections:

1. gate: shape `[768, 2048]`, relative offset 0, length 6,291,456 bytes;
2. up: shape `[768, 2048]`, relative offset 6,291,456, length 6,291,456 bytes;
3. down: shape `[2048, 768]`, relative offset 12,582,912, length 6,291,456 bytes.

The source shapes match the proven M1 matrix orientation. No transpose,
alignment padding, or layout transformation is applied. Conversion order is
layer ascending, expert ascending, then gate/up/down. Each layer container is
therefore exactly 2,415,919,104 bytes.

The detailed expert manifest is versioned independently and maps every logical
layer/expert key to an arbitrary shard ID, absolute payload offset/length,
relative projection ranges and shapes, BF16/F32 dtype metadata, and logical
payload SHA-256. It also records every physical shard path, length, and hash.
The selected one-container-per-layer policy does not constrain future manifests
to derive shard IDs or offsets from layer numbers.

The existing `ArtifactManifest` and `ExpertRegistration` values are constructed
from these records. `ArtifactReader` and `ExpertStore` continue to seek and read
only the logical expert range. No existing expert storage or numerical contract
is redesigned.

## Conversion and transaction policy

Before decoding any selected payload, the converter validates canonical tensor
roles, layer/expert identity, gate/up/down order, BF16 dtype, exact shapes,
shape-derived lengths, source shard indices and ranges, ordering, duplicates,
and required inventory completeness. It then verifies the complete length and
SHA-256 of every referenced pinned source shard.

BF16 values are decoded deterministically to F32 with the accepted M4.1-05
bit-preserving conversion. A 65,536-byte source chunk, 131,072-byte output or
expected chunk, and 131,072-byte artifact verification chunk bound maximum
explicit buffers to 327,680 bytes. A full layer and multiple experts are never
resident in conversion memory.

Preflight accounts for every converted expert byte plus the exact deterministic
manifest length. Renaming a completed temporary layer container does not copy
or duplicate its bytes, so the transaction requires no second layer-sized
allocation. Each shard is written to a fixed incomplete path, synced, and
atomically renamed. The manifest is synced and renamed last as the commit
marker. A transaction guard removes temporary and finalized files created by a
failed conversion.

## Source plan and public surface

`models/qwen3-30b-a3b/expert-source-plan-v1.tsv` records the pinned model ID and
revision, all 16 shard lengths and hashes, and all 18,432 projection identities,
source ranges, and shapes. It contains 1,819,195 bytes and has SHA-256
`619deb861e51d65cbb5dbbff1b7b64bc068515884cbed863ef9eb81f3153757b`.
The Apache-2.0 license and immutable upstream provenance remain anchored by
`source-manifest-v1.json` and ADR 0011. The compact conversion evidence records
the generation date, command, and exact Rust/Cargo tool versions.

`clr-qwen3-moe` adds explicit conversion specifications, records, errors, and
the orchestration entry point. `clr-storage` exposes its existing incremental
dependency-free SHA-256 implementation as `Sha256Hasher` so the Qwen converter
and evidence example can hash logical and physical streams without a new
dependency or full-payload residency. Existing expert APIs and artifact format
version 1 are unchanged.

## Evidence

The real vertical slice converted experts `0:0`, `0:127`, and `47:127`, covering
the first and last payload boundaries and two source/layer containers. It
verified 5,087,346,088 complete source-shard bytes, read 56,623,104 source
payload bytes across conversion and exact verification, and wrote 56,623,104
artifact bytes. Independent runs produced manifest SHA-256
`a78dad0cb0d625f0a50e8b65623745312729d2cab6404742f7e78df15f48e3eb`
and ordered shard-set SHA-256
`e46cbddf1e28a203e26fbdb86bf2909026c703c86fa339b226c2fe9973a612a6`.

The first complete run accounted for all 6,144 logical experts and 18,432
source tensors. It wrote 115,964,116,992 bytes across 48 layer containers plus
a 3,130,926-byte manifest. The manifest SHA-256 is
`9c581c6c46ecf830e7d0dd0e380d26b17803784f009b37ef2657ae34d06b2939`;
the SHA-256 over the 48 ordered binary shard hashes is
`b90d537f5c0c202b2bf5db0e74b8bf8b9ba9ea5378d788733d2bf9e11d36bf91`.
Every expert was compared byte-for-byte with direct BF16 decoding during the
conversion. A second complete run produced a byte-identical manifest and the
same 48 individual shard hashes.

Loading expert `23:64` through the unchanged `ExpertStore` read exactly
18,874,368 bytes while its layer container remained 2,415,919,104 bytes.

## Excluded work

- No quantization, BF16 compute kernel, mmap, async prefetch, SIMD, FFI, native
  kernel, tokenizer parsing, or full-model generation.
- No new dependency or unsafe code.
- Large source and converted files remain local evidence and are not committed.
