# ADR 0051: Analytical planner cost model

## Status

Accepted for M6.2-02.

## Decision

`clr-core` owns a dependency-free, safe-Rust analytical cost model. Its inputs
are model-derived per-token compute/RAM/storage demand and explicitly cited
measured rates: GFLOP/s, GiB/s, and MiB/s. It returns individual serial cost
terms plus prefill, decode, startup, and decode-throughput predictions.

The model does not parse files, inspect the current machine, choose a backend,
or include a hardware-performance constant. A caller must provide a non-empty
measurement ID with every rate. Compute work remains an explicit model-derived
input because the frozen M6.1 model profile does not yet record its FLOP count.

## Consequences

M6.2-03 can populate this generic model while enumerating placements. M6.2-05
will map cited JSON profiles into its inputs at the CLI boundary. The estimate
is an analytical baseline and must be compared against recorded end-to-end
measurements in M6.2-06; it is not a measured tokens/s claim.
