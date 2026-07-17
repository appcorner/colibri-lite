# colibri-lite-rs

A Rust-first, storage-aware inference runtime for low-memory
Mixture-of-Experts models.

## Initial target

- Windows x64
- CPU-first
- Tiny Qwen3-MoE correctness
- Qwen3-30B-A3B
- On-demand expert loading
- RAM-budgeted expert cache

## Current milestone

M0, M1, M2, M3, and M4 are complete. M4 closed with the validated ordered F32
baseline and documented numerical variance. Quantized runtime work is not
accepted; the next task is the simulation-only M5.1 memory-hierarchy study.

The frozen tiny model accepts token IDs directly:

```powershell
cargo run -p clr-cli -- generate --tokens 1,7,3,12 --max-new-tokens 4
```


## Project Status and Achievement Assessment

`colibri-lite-rs` has progressed beyond a feasibility experiment. The project now has a correctness-validated, deterministic, storage-aware Rust runtime capable of executing the full Qwen3-30B-A3B model and generating tokens with a bounded-memory expert-streaming path.

### Original Goals

The project was created to determine whether a Rust-first MoE inference runtime could:

1. Run Qwen3-30B-A3B on Windows x64 with a CPU-first execution path.
2. Avoid loading the complete model into RAM.
3. Load only router-selected experts from storage.
4. Preserve numerical correctness and deterministic behavior.
5. Operate under an explicit memory budget.
6. Reach practical inference performance.
7. Offer a useful alternative to RAM-first runtimes.
8. Provide a foundation for later memory, I/O, kernel, quantization, and accelerator optimization.

The core architectural goal is not merely to minimize RAM usage. It is to maximize inference performance within a user-selected memory budget by coordinating disk, RAM, and future accelerator memory.


### M4 Release Identity

M4 is formally closed and released as a reproducible correctness and performance baseline.

| Release field | Value |
|---|---|
| Release ID | `colibri-lite-rs-m4-qwen3-30b-a3b-f32-v1` |
| Git tag | `m4-full-qwen3-baseline-v1` |
| Final M4 closure commit | `d353d3d9324b6c47b58d5e4415a1d71ccee6abd3` |
| Frozen runtime source commit | `a230074959fc3b55ff73e8f4eb24e377a0a6b79f` |
| Model revision | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| Canonical artifact root SHA-256 | `f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2` |
| Performance baseline ID | `qwen3-30b-a3b-colibri-f32-windows-x64-v1` |
| Performance baseline schema | `colibri-qwen3-moe-m4.4-performance-baseline-v1` |
| Performance baseline SHA-256 | `29b2d95fa9eb74c1085cb31d2f63adbaa711fe8739d3051fa04f7b2b1c27ce9d` |
| Release provenance SHA-256 | `cc515a805b6b21b5cb8f59740ecb5759d5bdd9d809e36fe51ca3aa4adb7dab74` |

The authoritative release provenance is recorded in:

- [`models/qwen3-30b-a3b/m4-release-provenance-v1.json`](models/qwen3-30b-a3b/m4-release-provenance-v1.json)
- [`docs/reports/m4-release-closure.md`](docs/reports/m4-release-closure.md)

The release tag points directly to the final clean M4 closure commit. No M5 runtime optimization was included in this release.

### Current Milestone Status

| Area | Status | Assessment |
|---|---|---:|
| Runtime foundations, streaming, and generation (M0-M3) | Complete | 100% |
| Full Qwen3-30B-A3B artifact and integrity (M4.1) | Complete | 100% |
| Full-model correctness validation (M4.2) | Complete with documented numerical variance | 95% |
| Frozen F32 optimization baseline (M4.3-01) | Complete | 100% |
| Quantization candidate evaluation and decision (M4.3-02 to M4.3-06) | Complete | 100% evaluation complete |
| Same-hardware optimized-runtime comparison (M4.3-05) | Complete | 100% |
| Canonical performance baseline and release provenance (M4.4) | Complete | 100% |
| M4 milestone | Officially closed and tagged | 100% |
| Production performance readiness | Not ready | 5-10% |
| Overall product readiness | Research runtime | 25-35% |

### What Has Been Proven

The validated inference path covers:

```text
Token IDs
  -> Embedding
  -> Transformer Layers 0-47
  -> MoE Router
  -> Selected Expert Loading and Execution
  -> Final RMSNorm
  -> LM Head
  -> Vocabulary Logits
  -> Deterministic Token Selection
  -> KV-Cache Incremental Decode
```

The authoritative deterministic fixture is:

```text
Input IDs:     [9707, 11, 1879, 0]
Generated IDs: [1096, 374]
First mismatch: none
```

Validated properties include:

- Exact canonical artifact identity and reproducible manifests.
- Exact BF16-to-F32 tensor conversion for selected source values.
- Correct tensor shapes, offsets, orientation, and packed expert layout.
- Router agreement at Layers 0, 1, 24, and 47 under safe-margin rules.
- Selected expert gate, up, activation, product, down, weighted, and aggregated outputs.
- Full Layer 0-47 numerical propagation within documented contracts.
- Final normalization, LM-head logits, and deterministic token selection.
- Fixed-capacity KV cache with no growth or previous-position overwrite.
- Byte-identical evidence generation across deterministic reruns.

No unresolved Rust implementation defect remains in the validated F32 path. Observed differences were classified and documented as cross-runtime floating-point variance, BF16 amplification, reference-exporter behavior, or deterministic runtime refinements.

### Low-Memory Result

The current F32 runtime demonstrates that a 30B-class MoE model can execute without keeping the complete weight set resident in RAM.

| Resource | Observed baseline |
|---|---:|
| Canonical model artifact | approximately 122.15 GB |
| Peak process working set | approximately 145-148 MB |
| Modeled explicit runtime memory | approximately 127.8 MB |
| Peak resident expert payload | approximately 18.87 MB |
| KV-cache allocation | 1,179,648 bytes |

This validates the storage-aware, budgeted expert-streaming architecture. It does not yet validate practical latency.

### Current Performance Limitation

The authoritative unoptimized F32 baseline is:

| Metric | Result |
|---|---:|
| Prompt throughput | approximately 0.014-0.016 tokens/s |
| Decode throughput | approximately 0.013-0.016 tokens/s |
| Decode latency | approximately 62-78 seconds/token |
| Total logical reads for the frozen run | approximately 73.00 GB |
| Logical reads per cached decode token | approximately 12.17 GB |
| Expert-cache hits | 0 |
| Expert loads | 2,304 |
| Expert evictions | 2,303 |

The current runtime is therefore **correctness-valid but not performance-ready**.

A same-hardware CPU-only comparison with `ik_llama.cpp` and a Qwen3-30B-A3B Q4_K_M model produced approximately:

| Runtime | Prompt | Decode |
|---|---:|---:|
| `colibri-lite-rs` frozen F32 baseline | approximately 0.015 tok/s | approximately 0.015 tok/s |
| `ik_llama.cpp` Q4_K_M | approximately 3.97 tok/s | approximately 2.82 tok/s |

The comparison is directional rather than format-controlled. The gap reflects quantization, mmap and page-cache behavior, SIMD, threading, fused MoE, graph reuse, optimized kernels, and memory layout together. It cannot be attributed to quantization alone.

### Quantization Findings

The first full-model expert INT8 candidate used symmetric INT8 per output channel, F32 scales, F32 activations, and F32 accumulation.

It reduced the estimated expert artifact from approximately 115.96 GB to approximately 29.08 GB, but was classified as `quality_risk` because:

- Material propagated drift appeared from Layer 1.
- The largest Layer-47 final-block drift reached approximately 152.13.
- A Thai fixture changed argmax from token 7360 to 16222.
- Some router decisions changed or became numerically ambiguous.
- Minimum Tier-B Top-20 overlap fell to 0.65.
- The frozen Tier-A generated IDs still matched, but that alone was insufficient for acceptance.

Current precision policy:

- **Must remain F32:** router, RMSNorm, Q/K norms, final norm, routing weights, residuals, softmax, and accumulations.
- **Rejected:** router INT8 and full-model expert INT8 per tensor/per output channel for production.
- **Promising but insufficient evidence:** expert INT8 group-128 and selective mixed precision.
- **Future candidates:** BF16 storage or kernels for embedding, attention projections, and LM head, subject to Tier A/B/C validation.

### Achievement Assessment

| Dimension | Assessment |
|---|---:|
| Architecture feasibility | 100% |
| Full-model execution | 100% |
| Artifact integrity | 100% |
| Numerical correctness | approximately 95% |
| Determinism and reproducibility | approximately 95-100% |
| Low-memory objective | 100% feasibility proven |
| Expert-streaming correctness | approximately 90% |
| Fixture and platform coverage | approximately 60-70% |
| Quantization research | approximately 70% |
| Production quantized runtime | 0% |
| Performance maturity | approximately 5-10% |
| Production readiness | approximately 25-35% |

If the goal is to prove that a Rust storage-aware MoE runtime can correctly run Qwen3-30B-A3B under an extremely small RAM footprint, the project is approximately **90-95% successful**.

If the goal is a practical interactive production runtime, the project is approximately **30-40% complete**, with performance recovery as the dominant remaining challenge.

### M4 Final Verdict

```text
Research and feasibility success:  9/10
Correctness maturity:              9/10
Architecture validation:          9/10
Performance maturity:             1/10
Production readiness:             3/10
Strategic value of continuing:     8/10
```

M4 is officially complete. The project should continue, but the optimization strategy now pivots from "minimum RAM at all costs" to:

> Maximize tokens per second within a configurable user-selected memory budget.

### Next Phase: M5 Performance Recovery

The exact next task is `M5.1-01 Trace-driven memory hierarchy simulation`. The recommended order is:

1. Trace-driven cache and RAM-budget simulation.
2. Resident dense weights.
3. Configurable F32 expert cache.
4. I/O layout, mmap, coalesced reads, and prefetch.
5. Threaded and SIMD matrix kernels.
6. Revised mixed-precision or calibrated quantization candidates.
7. Optional CUDA acceleration for LM head, selected experts, and hot-expert VRAM caching.

Suggested decode-performance gates:

| Gate | Decode target | Interpretation |
|---|---:|---|
| M5-A | at least 0.05 tok/s | no more than 20 seconds/token |
| M5-B | at least 0.2 tok/s | usable for batch/background workloads |
| M5-C | at least 0.5 tok/s | beginning of practical interactive use |
| M5-D | at least 1.0 tok/s | practical CPU-first inference |
| Stretch | at least 2.0 tok/s | approaching the optimized external reference |

Every optimization must preserve the frozen F32 correctness invariants, deterministic fixture results, router guard decisions, selected intermediate evidence, KV-cache guarantees, finite outputs, bounded memory, and evidence compatibility.

### Current Project Position

`colibri-lite-rs` is no longer a proof that asks whether the model can run. M4 has established a tagged, reproducible, correctness-valid F32 baseline for Qwen3-30B-A3B on Windows x64.

The project is now entering M5 as a gated performance-recovery effort. M5 begins with trace-driven RAM and expert-cache simulation before any runtime modification, followed by resident dense weights, configurable expert caching, I/O improvements, optimized CPU kernels, and later mixed-precision or accelerator work where evidence supports it.

In short:

```text
M4: Can the full model run correctly with bounded RAM?  YES
M5: Can it be made fast enough for practical use?       NEXT
```

## Project documents

- [Implementation plan](docs/implementation-plan.md)
- [Task tracker](docs/tasks.md)
- [Deferred ideas and backlog](docs/backlog.md)
- [Work log](docs/work-log.md)
- [M0 milestone report](docs/reports/m0.md)
- [M1 correctness report](docs/reports/m1.md)
- [M2 storage and residency report](docs/reports/m2.md)
- [M3 autoregressive generation report](docs/reports/m3.md)
- [M4 release closure](docs/reports/m4-release-closure.md)
