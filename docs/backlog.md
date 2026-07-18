# colibri-lite-rs Backlog

This file captures ideas that are interesting but are not required to close
the current milestone. Recording an idea here does not approve implementation
or change milestone scope.

Classifications:

- `NEXT`: required by the immediately following milestone, but not the current
  one.
- `BACKLOG`: potentially useful after M4.
- `OUT-OF-SCOPE`: not aligned with the current product mission.

Before promoting an item, update `implementation-plan.md` and `tasks.md`, state
the evidence or requirement that changed, and obtain review when required by
`AGENTS.md`.

## Deferred ideas

| Idea | Class | Why it is deferred | Revisit when |
| --- | --- | --- | --- |
| CUDA backend | BACKLOG | M0-M4 target CPU correctness and storage-aware execution on Windows x64; a GPU backend would add kernels, placement policy, and a new correctness surface. | M4 is complete and profiling shows GPU acceleration serves a defined product requirement. |
| Other GPU backends (Vulkan, Metal, DirectML) | BACKLOG | Multiple backends would broaden platform scope before the first architecture is complete. | A post-M4 backend strategy is approved. |
| Reusable expert read buffer | BACKLOG | M5.3-02 reduced allocations in an isolated benchmark but was slower in 9/12 matched full-runtime rows. It is diagnostic/microbenchmark-only and must not replace the reference reader. | A fresh, repeatable full-runtime matrix shows an end-to-end improvement against the reference reader at identical F32 fixtures and budgets, with exact traces/correctness, no memory-budget regression, and an explicit working-set measurement. |
| Read-only memory mapping | BACKLOG | M5.3-04 was technically correct but regressed all eight paired full-runtime comparisons (median +5.92%) and raised measured peak working set to 29.46--39.00 GiB. It is rejected for runtime adoption and is retained only as isolated diagnostic evidence. | A materially different, reviewed design first shows in simulation why mapping semantics remain bounded, then demonstrates repeatable end-to-end improvement versus the reference reader, exact F32 correctness, Windows cleanup/lifetime safety, no configured-budget breach, and no unacceptable working-set increase. |
| Resident dense weights plus strict global LRU | NEXT | M5.4-01 simulation is complete for review and shows modeled logical-read value, but the resident-dense runtime path has not been executed and no throughput claim is established. | A separately reviewed M5.4-02 experiment preserves frozen F32 evidence, explicit total-RAM accounting, the configured limits, and the reference reader while showing repeatable full-runtime improvement. |
| Ornith architecture support | BACKLOG | Qwen3-MoE is the only architecture through M4; adding Ornith would weaken the single-model correctness path and expand generic contracts prematurely. | Qwen3-30B-A3B passes M4 and a concrete Ornith compatibility goal is approved. |
| Broad GGUF compatibility | BACKLOG | M4 requires a focused, versioned artifact that supports independent expert access; general GGUF compatibility may conflict with that storage contract. | The M4 artifact contract is stable and an interoperability use case justifies GGUF support. |
| Speculative decoding | BACKLOG | It adds draft-model coordination and verification behavior before basic generation and bounded memory are proven. | M3 and M4 are complete and generation profiling identifies decoding latency as the target bottleneck. |
| OpenAI-compatible HTTP server | BACKLOG | Serving, concurrency, request lifecycle, and API compatibility do not help close inference correctness or residency milestones. | The M4 runtime is reproducible and a deployment interface is explicitly planned. |
| Web UI | BACKLOG | A UI does not advance numerical correctness, storage behavior, or the target runtime path. | A post-M4 user workflow requires it. |
| Continuous batching | BACKLOG | It introduces scheduling and shared-cache complexity before single-request generation is correct. | Single-request M4 behavior is stable and measured workload evidence requires batching. |
| Multimodal support | BACKLOG | It requires additional model architectures, preprocessing, and artifact contracts outside the Qwen3-MoE target. | A post-M4 product requirement and architecture plan are approved. |
| Distributed inference or RPC | BACKLOG | Network partitioning and remote ownership would obscure local memory-budget correctness. | Local M4 execution is complete and model/hardware evidence requires distribution. |
| Agent frameworks and tool calling | BACKLOG | These are application-layer features outside the inference runtime mission. | A separate post-M4 application scope is approved. |
| katgpt-rs reasoning features | BACKLOG | Reasoning policies are not required for Qwen3-MoE numerical or storage correctness. | Core M4 runtime goals are complete and evaluation criteria are defined. |
| Performance parity with llama.cpp or ik_llama.cpp | BACKLOG | Established runtimes are baselines, while current milestones prioritize correctness and predictable memory rather than feature or speed parity. | M4 has a reproducible baseline and profiling identifies specific gaps worth addressing. |

## Idea entry template

Add new ideas using one table row with:

- A concrete idea name.
- One classification.
- The current reason not to implement it.
- An evidence-based condition for reconsideration.

Do not add implementation tasks for a backlog item until it is promoted into
the implementation plan through review.
