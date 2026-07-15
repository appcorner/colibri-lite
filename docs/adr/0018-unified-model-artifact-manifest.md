# ADR 0018: Unified Model Artifact Manifest

- Status: Accepted
- Date: 2026-07-15
- Milestone: M4.1
- Task: M4.1-08
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

M4.1 produced three reviewed but independently rooted artifact components:

- one dense F32 payload with a 435-tensor manifest;
- 48 expert F32 shards with 6,144 logical expert records;
- four canonical tokenizer assets with tokenizer/chat metadata.

The first full-model correctness checkpoint needs one deterministic entry point
that binds these components to the approved source identity and validated
runtime dimensions. It must remain relocatable, offer a fast metadata check and
a complete byte-integrity check, and reject incompatible or incomplete roots.

The existing dense, expert, tokenizer, and storage contracts already contain
the required logical ranges and payload hashes. Replacing them would add risk
without adding information.

## Decision

Add root artifact format `colibri-lite-model` version 1. Its canonical file is
`model-manifest-v1.json`, serialized as sorted, compact UTF-8 JSON with one
trailing newline. Canonical content contains no generation timestamp, absolute
path, host name, user name, temporary directory, or build commit.

The root manifest records:

- immutable model ID, revision, architecture, model type, and Apache-2.0
  license;
- BF16 source, F32 storage/compute, and little-endian metadata;
- runtime compatibility requirements and all validated Qwen3-MoE dimensions;
- separate model, tokenizer, and runtime-session limit concepts;
- source-contract path, size, and hash;
- dense component manifest/payload path, size, and hash;
- expert component manifest plus all 48 shard IDs, paths, sizes, and hashes;
- ordered expert shard-set SHA-256;
- tokenizer component manifest, four asset hashes, vocabulary counts, and
  preserved chat-template metadata;
- exact file/component counts and byte totals.

Every path uses canonical POSIX-style relative syntax from the artifact root.
The root manifest references component format version 1 rather than changing
any accepted dense, expert, tokenizer, `ArtifactManifest`, or `ExpertStore`
contract.

## Canonical directory layout

```text
model-manifest-v1.json
provenance/source-manifest-v1.json
dense/dense-manifest-v1.json
dense/dense-f32.bin
experts/expert-manifest-v1.json
experts/experts-layer-00000-of-00048.bin
...
experts/experts-layer-00047-of-00048.bin
tokenizer/tokenizer-artifact-manifest-v1.json
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
tokenizer/vocab.json
tokenizer/merges.txt
```

The local closure directory uses hard links to the already-reviewed dense and
expert files. Hard links are an evidence assembly detail, not an artifact
contract requirement. The generator reads existing component metadata and does
not convert, copy, transpose, or decode model payloads.

## Validation modes

`validate-metadata` verifies canonical root serialization, strict root fields,
versions, architecture/model identity, relative and unique paths, component
cross-references, counts, all file sizes, source/component-manifest hashes, and
all tokenizer hashes. It deliberately does not hash the 49 large F32 payload
files. For this artifact it hashes 19,187,816 bytes across 9 files including
the root manifest.

`validate-full` performs every metadata check and hashes all 57 referenced
files: the source contract, dense manifest/payload, expert manifest and 48
shards, tokenizer manifest, and four tokenizer assets. Including the root
manifest, it hashes 122,147,678,312 bytes across 58 files.

Both modes reject missing/renamed/truncated files, size mismatches, duplicate or
missing expert shards, unsupported root/component versions, incompatible
architecture/model type, unknown root critical fields, unsafe/non-canonical
paths, and incomplete temporary output. Full mode additionally detects
same-size dense or expert payload corruption.

The validator is a Python standard-library evidence tool. It adds no runtime
dependency, native library, unsafe boundary, or public API.

## Determinism and relocation

Two independent generations from the same component directory produced the
same 11,395 canonical bytes and SHA-256
`f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`.

After moving the complete directory to another root, metadata validation
passed and regeneration produced the same byte length and hash. Synthetic
tests also run both metadata and full validation after relocating an entire
fixture root.

## Artifact footprint

The 57 referenced component files occupy 122,147,666,917 logical bytes. Adding
the 11,395-byte root manifest gives an exact complete footprint of
122,147,678,312 bytes across 58 files.

## Excluded work

- No new conversion, tensor decoding, or inference.
- No quantization, mmap, SIMD, GPU work, or performance optimization.
- No tokenizer algorithm or chat-template rendering.
- No dense/expert artifact redesign, dependency, native library, or unsafe
  code.
