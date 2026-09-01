# ADR 0074: M6.3-R1.3 Candidate FileObject Correlation Amendment

## Status

Accepted after the first official-attempt sample was rejected and before any valid R1.3 sample was admitted.

## Context

The v2 official runner correctly verified the group-32 artifact before READY and started ETW before GO. However, the verified `R1_1CandidateReader` retained the same file handle into the timed phase. Kernel File Create/path registration for that handle therefore occurred before ETW started. The timed trace contained candidate reads but could not resolve their old FileObject to the exact candidate path, so the parser reported zero candidate File Read/FileKey evidence and rejected the sample.

The rejected attempt is retained under `D:/tmp/colibri-lite-runs/m6.3-r1-3-official-20260901/runtime_cache_cold-pair-01-order-1-candidate`; `samples.partial.json` was never created, so zero official samples were admitted.

## Decision

Keep full candidate byte-length and SHA-256 verification before READY. After GO, reopen the already-verified candidate path through a crate-private measurement-only helper that preserves the verified identity but performs no second hash. Drop the old handle and use the new post-GO handle for timed direct packed reads.

This makes Kernel File Create/path registration visible inside the ETW window without adding full-artifact verification I/O to the timed phase. It changes no tensor bytes, layout, quantization rule, router, numerical path, cache policy, threshold, pair order, or reporting rule.

The release binary identity must be re-frozen before the next candidate measurement. The rejected v2 attempt remains diagnostic evidence and may not be retried or counted.
