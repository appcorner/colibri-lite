# ADR 0057: M6.3 Layer-0 Group-128 Safe CPU Candidate

## Status

Accepted for M6.3-01 proposal only. This authorizes investigation in M6.3-02;
it does not authorize all-layer rollout, a production precision policy, or a
throughput claim.

## Decision

Evaluate exactly one candidate:

- scope: Qwen3-30B-A3B MoE Layer 0 experts only;
- weight format: symmetric signed INT8, input groups of 128, with one F32
  scale per output-row/input-group;
- backend: native safe-Rust scalar CPU loop that consumes INT8 values and F32
  group scales directly, with F32 activations and F32 accumulation;
- resident non-expert path: existing `reference-f32-v1` router, norms,
  routing weights, attention, residuals, softmax, and cache remain unchanged.

The Layer-0 candidate must never reconstruct a complete projection or expert
as F32 on its normal execution path. Its direct-consumption operation is
equivalent to accumulating `activation[column] * (INT8[column] * scale[group])`
in the existing deterministic order. SIMD, FFI, threading, mmap, prefetch,
GPU backends, and a custom allocator are not part of this candidate.

## Rationale

The tracked M6.1 profile has no usable GPU backend or VRAM budget, while its
CPU/RAM and SSD measurements are available. Group-128 had the lowest recorded
representative weighted-expert error among the evaluated INT8 formats
(`0.0820999` versus `0.1043549` for per-output-channel INT8) at `4.085321x`
modeled F32-expert compression. The earlier per-output-channel format is
explicitly `quality_risk` and cannot be reused as this candidate.

Layer 0 gives the first slice an F32 input and localizes any first divergence
before error can accumulate through earlier quantized layers. The candidate is
still experimental: its group-128 evidence was representative only and does
not establish bilingual quality or end-to-end performance.

## Dependency, unsafe, and licensing review

No Cargo dependency, native library, generated binary, or external source code
is added. `clr-core` remains dependency-free and the implementation stays in a
Qwen-specific private module. The workspace `unsafe_code = "forbid"` remains
in force; no unsafe boundary is proposed.

The model-derived output remains governed by the pinned
`Qwen/Qwen3-30B-A3B` revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
under Apache-2.0, as recorded in ADR 0011. The quantizer, artifact reader, and
scalar operation must be independently implemented; no code may be copied from
Colibri, llama.cpp, ik_llama.cpp, or another runtime. There is therefore no
third-party code attribution or dependency license to add for this proposal.

## Provenance and admission requirements

M6.3-02 must create an additive, versioned group-128 artifact that cites the
canonical F32 root manifest
`f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2`, the
pinned model revision, Apache-2.0 license, conversion command, tool versions,
source and output hashes, tensor names/shapes, group axis/size, scale layout,
rounding, saturation, byte order, and generation date. It must use the temp
artifact policy's disk preflight and one flat run directory.

The first slice is admitted only if it reads the new INT8/scales directly,
keeps a bounded working set, and preserves the executable F32 comparison path.
M6.3-04 and M6.3-05 must supply the correctness and repeated cold/warm evidence
before M6.3-06 may make a stop/go decision.
