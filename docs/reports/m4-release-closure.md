# M4 Release Closure

Release ID: `colibri-lite-rs-m4-qwen3-30b-a3b-f32-v1`

The authoritative machine-readable provenance record is
`models/qwen3-30b-a3b/m4-release-provenance-v1.json`. It pins the runtime
source commit, Qwen3-30B-A3B revision, tokenizer and canonical artifact hashes,
the accepted F32 correctness contracts, the M4.4 resource baseline, and the
directional ik_llama reference without copying model payloads.

## What was built and proven

M4 converted the pinned Qwen3-MoE artifact into independently addressable dense
and expert components, validated the storage-aware Rust F32 path through all 48
layers, final normalization, LM head, deterministic generation, and cached
decode, and recorded bounded expert residency and KV-cache behavior. The F32
baseline generated `[1096, 374]` for the frozen fixture with documented
cross-runtime numerical variance.

The canonical artifact is format version 1 with 57 payload files and
`122147666917` bytes. The release uses BF16 source weights, F32 storage and
ordered scalar F32 computation. The M4.4 performance baseline records
`73004834816` logical bytes read, a one-expert `18874368`-byte cache, and
`127823000` modeled explicit bytes. These are application-level logical reads;
physical device I/O was not measured and filesystem cache state was
uncontrolled.

## Decisions and limitations

The first all-layer expert INT8 per-output-channel candidate was rejected for
full-model production because propagation and Tier-B semantic stability were
not sufficient. It remains diagnostic/selective-use research only. The
ik_llama Q4_K_M comparison validates useful hardware capability but is
directional and does not establish quality equivalence to the F32 baseline.
Performance readiness is therefore not claimed: the current path is scalar and
disk-streaming, with optimization intentionally deferred.

M4 verdict: full-model feasibility, artifact integrity, F32 correctness,
deterministic generation, and low-memory feasibility passed; performance
readiness is not ready. The investment decision is to continue with the F32
memory-hierarchy and performance-recovery pivot while preserving all frozen
correctness invariants.

## Closure

The approved release tag is `m4-full-qwen3-baseline-v1`. It must point to the
final clean M4 closure commit. The exact next task is
`M5.1-01 Trace-driven memory hierarchy simulation`; no M5 implementation or
RAM/cache simulation is included in this release.

Reproduce and validate the provenance payload with:

```powershell
python python/reference/build_m4_release_provenance.py
python -m unittest python.reference.test_m4_release_provenance
python python/reference/build_m4_release_provenance.py --verify-tag --expected-commit <closure-commit>
```
