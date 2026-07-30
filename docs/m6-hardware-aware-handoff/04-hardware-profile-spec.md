# Hardware Profile Specification

`doctor` emits versioned JSON containing OS/Rust/runtime identity, CPU model
and logical cores, measured CPU-kernel throughput, RAM capacity and bandwidth,
storage device/path and sequential plus expert-sized random-read results,
available backend/driver identities, usable VRAM, and host/device transfer.

Every benchmark records payload size, repetitions, warm/cold cache semantics,
median and dispersion, clock source, and timestamp. “Unknown” is valid; a
device must not be inferred usable merely because it is detected.

RAM and VRAM recommendations include a safety reserve. They are advisory
inputs to the planner; runtime admission still enforces the user’s explicit
budget.
