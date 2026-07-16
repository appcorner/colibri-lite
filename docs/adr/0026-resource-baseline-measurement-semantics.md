# ADR 0026: Resource Baseline Measurement Semantics

- Status: Accepted provisionally for M4.2-05
- Date: 2026-07-16
- Milestone: M4.2
- Task: M4.2-05
- Fixture: prompt `[9707, 11, 1879, 0]`, generated `[1096, 374]`

## Context

M4.2-04 proved the complete storage-aware full-model path, including two
cached decode steps. M4.2-05 measures that unchanged correctness path. The
baseline must distinguish application-level resource accounting from operating
system observations and must not imply that the Windows filesystem cache was
controlled.

The feature-gated validation test also holds compact Transformers checkpoints
and temporary full-logit rows. Those validation buffers are not part of a
production inference session, but they are real allocations in the measured
test process and must remain visible in the modeled-memory breakdown.

## Decision

Use `std::time::Instant` for Rust phase timing. Prepare reference files before
the Rust runtime timing boundary. For every processed position, stop its
inference timer immediately after streamed LM-head evaluation, deterministic
greedy selection, and transactional KV append. Perform checkpoint comparisons
after that boundary. Report the separately sampled whole-process wall and CPU
times, which include the test harness and comparisons.

Define prefill as the sum of the four sequential cached prompt-position
forwards used by the proven path. Define decode 1 and decode 2 as the forwards
for generated input tokens 1096 and 374 respectively. Do not reinterpret this
scalar correctness path as a batched prefill benchmark.

Count artifact I/O at the application range-read boundary:

- requested bytes equal bytes successfully returned by the range readers;
- one dense payload file remains open for the run;
- every expert miss opens one expert shard, reads one selected payload, and
  closes it;
- embedded plans and manifests contribute zero artifact metadata bytes;
- logical bytes are not physical device bytes and do not establish cold-cache
  performance.

Measure process working set and private bytes from a separate PowerShell
process at 100 ms intervals. Retain the larger of the Windows process peak
working-set property and the sampled peak. Private-byte peak and final resident
values are sampled observations. System available memory is a before/after
host observation and may change because of unrelated activity.

Run three new processes without dropping or otherwise controlling the OS
cache:

1. fresh process, uncontrolled OS cache;
2. fresh process after run A, uncontrolled OS cache;
3. immediate fresh process, potentially filesystem-cache-warm.

## Reconciliation Contract

Every accepted run must satisfy:

```text
requests = hits + misses
misses = loads
evictions = loads - peak resident expert count
expert bytes read = loads * one expert payload
total artifact bytes = dense bytes + expert bytes
repeated bytes = total bytes - useful unique bytes
resident expert bytes <= configured cache budget
```

KV evidence must retain 48 layers, capacity and final length 6, per-layer K/V
shape `[6, 4, 128]`, 96 fixed payload allocations, `1,179,648` allocated bytes,
no decode allocation growth, and no previous-position overwrite.

## Consequences

The baseline is trustworthy for logical application I/O, explicit-buffer
modeling, cache behavior, and observed process memory on the recorded host. It
is not a physical-storage benchmark and makes no cold-cache claim. The zero-hit
one-expert cache is an accepted correctness-baseline result and does not
authorize cache optimization during M4.2-05.
