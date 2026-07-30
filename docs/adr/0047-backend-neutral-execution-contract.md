# ADR 0047: Backend-Neutral Execution Contract

## Status

Accepted for M6.0-02. No compute backend is selected or implemented by this
decision.

## Context

`reference-f32-v1` is frozen, while M6 will compare candidate CPU or approved
accelerator paths against it. Existing core tensors own F32 payloads; storage
and Qwen execution must not determine generic backend contracts.

## Decision

`clr-core` exposes validated metadata-only types:

- `TensorDescriptor` for shape and dtype;
- `OperationDescriptor` and `OperationKind` for the existing generic scalar
  operation signatures;
- `ExecutionBudget` for per-operation host/device byte limits;
- `ExecutionRequest`, `ExecutionMetrics`, and `ExecutionResult` for admitted
  work and backend-reported result metadata.

The result validates exact output descriptor agreement and resource-budget
compliance. Invalid signatures and reports use the matchable
`RuntimeError::BackendContractViolation` category.

The contract intentionally excludes payload ownership, file paths, artifact
formats, cache policy, Qwen operation names, device handles, backend discovery,
threading, and quantization layouts. These remain out of scope until later M6
tasks establish their requirements.

## Consequences

Future backends can describe and report a comparable operation without making
`clr-core` depend on storage, Qwen, or a hardware SDK. M6.0-03 can use these
descriptors to identify a candidate's first divergent stage. A backend trait
is deliberately deferred: no implementation currently needs a common dispatch
surface, and adding one now would broaden the public API prematurely.
