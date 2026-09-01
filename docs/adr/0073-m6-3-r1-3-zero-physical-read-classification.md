# ADR 0073: M6.3-R1.3 Zero Physical-Read Classification Amendment

## Status

Accepted after reference-only collector preflight and before any R1.3 candidate performance sample.

## Context

ADR 0072 and measurement-contract-v1 required exact process/file-correlated ETW and prohibited replacing missing physical I/O with zero. The first reference-only preflight used the frozen runner/collector boundary and produced a zero-loss Kernel File/Disk trace. All 48 expert shards had matching physical Disk events, but `dense-f32.bin` had no matching Disk event, causing the v1 parser to classify the whole trace `not_measured`.

Raw trace inspection established that this was not missing file correlation. For `dense-f32.bin`, the target PID emitted 2,055 Kernel File Read events, one exact FileKey, and 9,839,435,776 logical event bytes, exactly matching runtime dense logical-byte accounting. No target-PID Kernel Disk read matched that FileKey, and tracerpt reported zero lost events/buffers.

## Decision

For CSV/tracerpt captures only, classify an artifact as observed when the zero-loss trace contains at least one exact target-PID Kernel File Read resolved through the exact artifact path/FileObject and at least one FileKey. Physical bytes are the sum of target-PID Kernel Disk Read bytes matching those FileKeys.

A physically cached artifact may therefore report measured physical bytes of zero when its exact File Read/FileKey evidence is present but no matching Kernel Disk Read exists. This is measured zero, not zero substitution. Missing File Read/FileKey evidence or any trace loss remains `not_measured`.

The legacy XML parser semantics are unchanged. The preflight sample is diagnostic only and is excluded from the five official pairs per condition. No candidate performance result was observed before this amendment.
