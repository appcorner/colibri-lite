# M6.3-R2.3 Current State

## Project objective

`colibri-lite-rs` is a Rust-first MoE inference runtime. The active objective is
to maximize measured tokens/s from available RAM/VRAM/SSD while preserving
model quality, numerical correctness, determinism, and explicit resource
budgets. The current model target is Qwen3-30B-A3B on Windows x64, CPU-first.

## Current repository state

- Current closed reference point: commit `82327280a6fb8ac6b15684c1e695e0bdd6f46238`.
- `M6.3-R2.2-D5c` is closed `GO`.
- `M6.3-R2.3` all-layer plan review is closed `GO`.
- `M6.3-R2.3-IC` implementation contract is frozen.
- `M6.3-R2.3a` canonical-F32 four-token reference is closed `PASS`.
- Exact next task: `M6.3-R2.3b` per-layer candidate characterization.

## D5 production-like hybrid result

The validated D5 policy is:

- Layer 0: group32 gate/up/down using the existing AVX2+FMA native backend with
  scalar fallback.
- Layer 24: F32 gate/up plus packed group8 down, scalar packed down only.
- Layer 47: canonical F32.
- All other layers: canonical F32.

Official D5c performance used 40 fresh processes and closed `GO`:

- English median wall speedup vs F32: `3.4380408493%`.
- Thai median wall speedup vs F32: `4.0295565832%`.
- Hybrid wall wins by fixture/cache cell: `5/5`, `5/5`, `4/5`, `5/5`.
- Worst median memory regression: `0.8145196990%`, within the frozen 1% cap.
- Logical expert bytes decreased in every pair.
- Samples SHA-256: `47ac917cb7b3886e4ff786639c832e28936f1d64fd6c4335e5819d3f924c8415`.
- Result SHA-256: `b6387a2975074a9f2fc164c04a1741fc6324d64c54fd1947a27fd4e023a730e3`.

Physical SSD bytes are not part of this D5 decision. The reused ETW exact
physical-I/O join failed on the current Windows trace and was explicitly
removed from the D5c critical path under ADR 0116. Do not claim physical SSD
reduction from D5 evidence.

## R2.3 all-layer planning result

The R2.3 review deliberately forbids depth interpolation. Every layer has an
explicit status:

- Layer 0: admitted non-F32 policy from D5.
- Layer 24: admitted non-F32 policy from D5.
- Layer 47: F32-locked by prior characterization.
- Layers `1-23,25-46`: unmeasured and F32 by default.

Plan-review contract SHA-256:
`4c4f37356e01d57b247664d0c2537b2f928ad329863778f98a9de0b54624ba2c`.
Plan-review result SHA-256:
`c525cfc8ed536fb7d0033104f41e5e2f64a0d9a2842c4fcddba5b32dfac7e6e2`.

The implementation contract is
`models/qwen3-30b-a3b/m6.3-r2-3-implementation-contract-v1.json`, SHA-256
`97dd33b5cdb576d1ebcf9b3b7661f5462c912be40c7485c7e6bd7f8773d7771b`.
Its phase order is R2.3a through R2.3f and must not be reordered based on
results.

## R2.3a frozen F32 reference

Canonical F32 four-token generation is now frozen before any new candidate
layer executes:

- English: `[0, 358, 2776, 264]`.
- Thai: `[7360, 91, 16, 15]`.
- The first two tokens and prompt semantics anchor to the accepted R1.2/D5
  reference.
- Reference SHA-256:
  `9f8de6841ff883c062c2ed8387dba93631c9de0565943f1de2bb1a27f0fada1e`.
- R2.3a result SHA-256:
  `8a5c0f254e0d9b36bc8556ed9e4330dc4896f12a3767f2525c9fd04a63790203`.
- One-shot test runtime: `438.91s`, exit 0, stderr empty.

## Exact next task: R2.3b

Characterize layers `1-23,25-46` independently. Each layer has 21 candidate
configurations:

- projection subsets: `gate_up_down`, `gate_up`, `gate_down`, `up_down`,
  `gate`, `up`, `down`;
- group sizes: `32`, `16`, `8`;
- total: `7 x 3 = 21` candidates per layer, `45 x 21 = 945` layer-candidates.

Every candidate must pass both:

- same-input local max-abs `<= 0.001`;
- sequence-aware bilingual one-generated-token max-abs `<= 0.001`.

Selection is deterministic: choose the candidate with maximum logical bytes
saved among candidates passing all local and sequence gates. Tie 1 prefers the
larger group size. Tie 2 uses the frozen projection-subset order listed above.
If no candidate passes, the layer remains canonical F32.

## Global prohibitions

Do not:

- reuse historical R2.2c/R2.2d results;
- infer a layer policy from depth or a neighboring layer;
- retune thresholds after seeing results;
- automatically retry an unfavorable candidate/sample;
- introduce a new precision family;
- implement native group16 or native group8;
- change cache policy;
- claim physical SSD I/O reduction without a new telemetry contract;
- start R2.3c or later before R2.3b is formally closed;
- start M6.4 implementation.

## Useful prior implementation seams

Prefer reusing proven code rather than creating parallel machinery:

- grouped-layer artifact builder: `scripts/convert_m6_3_r2_2_d2_grouped_layer.py`;
- projection sensitivity harness: `crates/clr-qwen3-moe/src/r2_2_d3_tests.rs`;
- sequence-aware precision harness: `crates/clr-qwen3-moe/src/r2_2_d4d3_tests.rs`;
- D5 production reader/path: `crates/clr-qwen3-moe/src/r2_2_d5_hybrid.rs` and
  `r2_2_d5b_tests.rs`;
- R2.3a F32 reference harness: `crates/clr-qwen3-moe/src/r2_3a_reference_tests.rs`.

Preserve frozen historical harnesses. Add new R2.3b code instead of rewriting
accepted evidence paths unless a narrowly scoped reusable helper change is
necessary and separately justified.
