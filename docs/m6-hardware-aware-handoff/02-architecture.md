# M6 Architecture

```text
Hardware profile + Model profile
              ↓
           Planner ──> Reproducible execution plan
              ↓                         ↓
Reference F32 comparator <── Backend-neutral executor ──> telemetry
                                      ↓
                        CPU / approved GPU / tiered storage
```

Rust remains the control plane: profiles, planning, budget admission,
placement, telemetry, and correctness comparison. Compute backends are
replaceable implementations behind a narrow execution contract.

The existing crate direction is retained. `clr-core` owns generic contracts;
`clr-storage` owns artifact and residency mechanics; `clr-qwen3-moe` maps
Qwen operations; `clr-cli` presents commands. A backend must not make
`clr-core` aware of files, Qwen fields, or accelerator APIs.
