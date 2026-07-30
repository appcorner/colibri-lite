# Backend Contract

The M6.0 contract is intentionally small. `clr-core` exposes
`TensorDescriptor`, `OperationDescriptor`, `ExecutionBudget`,
`ExecutionRequest`, `ExecutionMetrics`, and `ExecutionResult`. It describes
metadata, not tensor ownership: a backend receives a validated generic
operation and execution budget, then reports output metadata and measured
host/device memory. It never changes routing policy or silently exceeds the
admitted plan.

Required result fields are output shape/dtype, elapsed time, and host/device
memory. Layer identity, physical/logical reads, storage metrics, and backend
selection are deliberately deferred to later contracts. Unsupported operations
must return a structured error, not fall back silently.

The comparator runs the same deterministic inputs through the F32 path and the
candidate, checks exact IDs/shapes/names first, then numerical checkpoints. Its
report identifies the first divergence and tolerance used.
