# Backend Contract

The M6.0 contract is intentionally small. A backend receives validated tensor
views, operation descriptors, precision metadata, and an execution budget. It
returns outputs plus measured counters; it never changes routing policy or
silently allocates beyond the admitted plan.

Required result fields are operation/layer identity, output shape/dtype,
elapsed time, allocated/resident bytes, device bytes, and physical/logical read
counters where applicable. Unsupported operations must return a structured
error, not fall back silently.

The comparator runs the same deterministic inputs through the F32 path and the
candidate, checks exact IDs/shapes/names first, then numerical checkpoints. Its
report identifies the first divergence and tolerance used.
