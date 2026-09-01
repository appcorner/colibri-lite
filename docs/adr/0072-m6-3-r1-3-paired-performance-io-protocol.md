# ADR 0072: M6.3-R1.3 Paired Performance and I/O Protocol

## Status

Accepted before any R1.3 candidate performance sample is executed.

## Context

ADR 0071 closes R1.2 with `GO` for the single admitted Layer-0 candidate
`cpu-safe-rust-int8-group32-layer0-r1-1a`. R1.3 is measurement-only. It must
not change the quantization layout, numerical path, cache policy, artifact
format, or production defaults, and it does not authorize M6.4.

## Measurement identity

Use the frozen `short_english` fixture `[9707, 1879]`, the same release test
binary for reference and candidate samples, the canonical F32 artifact root,
and the admitted group-32 artifact identity
`777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2`
with 679,477,248 bytes. The F32 ExpertStore byte budget remains 18,874,368
bytes in both paths. The candidate continues to use the direct packed reader
from R1.2 and receives no new retained candidate cache.

The declared usable-RAM ceiling is 16 GiB (17,179,869,184 bytes). This is below
the previously measured 16.55 GiB advisory usable-RAM budget and is a fixed
review gate rather than a live-memory estimate.

## Conditions and ordering

Measure two runtime-cache conditions independently:

- `runtime_cache_cold`: the existing F32 ExpertStore is empty at the timed
  phase boundary;
- `runtime_cache_warm`: the same fixture is executed once before the timed
  phase using the same F32 ExpertStore, then the timed phase starts with that
  retained cache state.

`warm` does not mean device-cold/device-warm. OS filesystem cache state remains
`uncontrolled` and is reported, not inferred.

Run exactly five process pairs per condition. Pair order is frozen as:

1. candidate then reference;
2. reference then candidate;
3. candidate then reference;
4. reference then candidate;
5. candidate then reference.

Each sample is a fresh process. Reference and candidate in a pair use the same
binary, fixture, context shape, F32 cache budget, host, and collector version.

## Timed-phase and ETW boundary

Each process performs immutable identity/setup work before the timed phase. A
warm process also performs its single prewarm pass before the timed phase. The
process then creates a readiness marker and waits for a launch marker.

The external collector starts process-correlated Kernel File/Disk ETW only
after readiness, creates the launch marker, and measures the inference phase.
This prevents warm-up I/O from being counted as timed physical reads. Process
wall/setup time is still recorded separately so candidate verification and
prewarm overhead cannot disappear from the evidence.

## Required per-sample evidence

Record:

- process wall seconds, setup seconds, timed wall seconds, TTFT, prefill tok/s,
  and decode tok/s;
- sampled process working-set peak and private-byte peak at 100 ms cadence;
- dense, F32-expert, candidate, and total logical artifact bytes;
- logical bytes/token;
- exact process/file-correlated ETW physical read bytes and physical
  reads/token;
- F32 ExpertStore hit/miss/load/eviction counters, configured byte budget, and
  peak resident bytes;
- candidate direct-reader payload bytes and peak packed-expert bytes;
- KV-cache bytes and zero VRAM bytes;
- executable SHA-256, candidate SHA-256, ETW provider/parser identity, PowerShell
  version, and OS filesystem-cache classification.

A missing or uncorrelated ETW result makes the sample invalid. Physical bytes
must never be replaced with logical bytes or zero.

## Resource and reporting gates

Every valid sample must stay at or below 17,179,869,184 process working-set
bytes. Candidate memory must remain explainable by the paired reference plus
the direct packed-expert peak, the one-MiB verification buffer used during
setup, and a 10% accounting allowance. No mapped candidate bytes are allowed.

R1.3 itself records measurements; it does not promote the candidate. For each
condition report all five paired percentage deltas plus median/min/max. A
throughput or I/O win may be claimed only when the median and all five paired
deltas point in the claimed direction. Otherwise classify that metric as mixed
or no-win and leave the decision to R1.4.

## Stop conditions

Stop R1.3 evidence collection as invalid if a sample has a process failure,
missing metrics, missing memory samples, non-correlated ETW, mismatched binary
or artifact identity, wrong fixture/result, or violated RAM/accounting gate.
Do not rerun only an unfavorable sample; replacement runs require a reviewed
invalid-run reason.
