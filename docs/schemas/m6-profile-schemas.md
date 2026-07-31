# M6 Profile Schemas

`hardware-profile-v1` is produced by `doctor`; `model-profile-v1` is produced
by `profile-model`. Both are immutable observations with a schema/version and
runtime identity. Planner inputs must cite their profile IDs and hashes.

`measured` records require a distribution. `unavailable` and `not_run` records
require a reason; neither represents zero throughput, zero latency, or an
unsupported model. A planner may only score a backend/path with the required
`measured` data, and must surface other candidates as unsupported or
low-confidence.

Hardware profile byte recommendations are advisory safe budgets; runtime plan
admission remains responsible for enforcing user-selected RAM/VRAM budgets.
Model profiles describe pinned artifact facts and possible precision inventory;
they do not authorize a quantized format or backend implementation.

M6.2 planner requests and results are defined separately in
`planner-contract-v1.schema.json`. They cite these immutable profile IDs and
document hashes, then carry only explicit workload and budget inputs.
