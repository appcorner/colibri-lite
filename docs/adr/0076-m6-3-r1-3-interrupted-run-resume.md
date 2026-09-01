# ADR 0076: M6.3-R1.3 Interrupted Run Resume

## Status

Accepted after an external orchestration interruption and before any replacement sample is executed.

## Context

The R1.3 v5 official set admitted four samples. The fifth expected sample,
`runtime_cache_cold-pair-03-order-1-candidate`, completed model execution and ETW
correlation, but the orchestration process ended before `capture.json` and the
partial sample record were committed. Therefore memory/process-wall evidence is
incomplete and the attempt is invalid despite its correlated ETW trace.

## Decision

Preserve the interrupted directory as diagnostic evidence and never promote it.
Resume only from the exact `samples.partial.json` prefix. The runner must verify
contract identity, binary/candidate/parser/collector/provider identities, exact
sample order, and the persisted prefix before executing the next sample.

The interrupted attempt does not count toward the 20 valid samples and cannot be
selected based on its numerical result. A replacement is permitted only because
the failure is an external orchestration/evidence-commit failure, not an
unfavorable performance outcome.

No threshold, pair order, cache policy, quantization path, runtime binary,
collector, parser, provider set, or measurement condition changes. The four
already admitted v5 samples remain valid. The replacement sample restarts the
same expected identity from a fresh process and fresh ETW session.

R1.3 remains blocked if the persisted prefix is inconsistent or if any resumed
sample fails the existing v5 gates. M6.4 remains blocked pending R1.4 review.
