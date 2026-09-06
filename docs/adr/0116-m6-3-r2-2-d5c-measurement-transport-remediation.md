# ADR 0116: M6.3-R2.2-D5c Measurement Transport Remediation

## Status

Accepted after the first D5c execution attempt failed before any sample was admitted.

## Context

The D5c execution frozen at commit `b47791a` reused the R1.3 ETW physical-I/O collector even though the D5 contract does not include physical-read bytes in its measured metrics or GO rule. The first official process completed inference successfully with exact READY/GO handoff and memory sampling, but the ETW parser returned `not_measured`, so the runner correctly stopped without retrying and admitted zero samples.

The captured trace contains 3,593 logical artifact File Read events and zero exact Kernel File/Kernel Disk correlations. Removing the disk-PID restriction does not restore the join. Exact intersections between FileKey/FileObject/IRP and Disk FileObject/IORequestPacket are all empty on the current Windows ETW rendering. Therefore no heuristic physical-I/O attribution is scientifically defensible.

Failure evidence is frozen in `models/qwen3-30b-a3b/m6.3-r2-2-d5c-transport-failure-v1.json`. Performance values from that failed sample are not admitted or used for any decision.

## Decision

D5c transport is remediated with a dedicated memory/process collector that preserves the same process launch, artifact identity checks, READY/GO boundary, stdout/stderr capture, and 100 ms working-set/private-byte sampling, but does not start ETW, tracerpt, or the physical-I/O parser.

This is a measurement-scope correction, not a performance-threshold or candidate change. The frozen D5 metrics remain wall time, prefill/decode throughput, peak working set, peak private bytes, logical artifact bytes, expert payload bytes, and candidate artifact bytes. All D5 GO thresholds and balanced pair order remain unchanged.

The remediated matrix must use a new output root and contain 40 fresh processes. No v1 sample may be imported. Automatic retry remains prohibited. Exact-prefix resume remains transport recovery only and requires explicit review.

A non-official transport smoke may validate collector mechanics but its performance metrics must not be read or admitted.

D5c `GO`, if achieved, still authorizes only a separate all-layer plan review.