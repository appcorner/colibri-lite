# ADR 0066: M6.3-R1 ETW Working-Directory Contract

## Status

Accepted before the next R1.1a candidate execution. No candidate numerical
result existed when this contract was frozen.

## Context

The direct Rust test executable resolves the frozen R1.1b checkpoint payload
relative to the `clr-qwen3-moe` crate directory. The ETW collector previously
forced every child process to the repository root. A group-64 run therefore
completed conversion and started the full-model workload, but failed before
checkpoint comparison when `../../models/...` resolved outside the repository.
This is an infrastructure failure, not a candidate numerical result.

## Decision

Every ETW configuration must provide an explicit `working_directory`. The
collector resolves it before starting ETW, requires it to be the repository
root or a descendant, requires the path to be a directory, and records the
resolved path in `capture.json`. R1.1a direct test executables must use:

```text
D:\sandboxs\colibri-lite\crates\clr-qwen3-moe
```

Candidate configurations must also list
`COLIBRI_ARTIFACT_ROOT`, `COLIBRI_R1_1A_CANDIDATE_ID`, and
`COLIBRI_R1_1A_CANDIDATE_PATH` in `required_environment`. The new
`-ValidateConfigOnly` mode verifies these bindings, artifact paths, output
scope, and working directory without requiring Administrator privilege or
starting ETW.

## Consequences

Missing, nonexistent, or out-of-repository working directories fail before
ETW. The failed group-64 run cannot be reused. A fresh conversion and fresh
three-run characterization remain required; group-32 and R1.2 stay blocked.
