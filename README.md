# colibri-lite-rs

A Rust-first, storage-aware inference runtime for low-memory
Mixture-of-Experts models.

## Initial target

- Windows x64
- CPU-first
- Tiny Qwen3-MoE correctness
- Qwen3-30B-A3B
- On-demand expert loading
- RAM-budgeted expert cache

## Current milestone

M0, M1, and M2 are complete. M3 autoregressive generation is in progress.

The frozen tiny model accepts token IDs directly:

```powershell
cargo run -p clr-cli -- generate --tokens 1,7,3,12 --max-new-tokens 4
```

## Project documents

- [Implementation plan](docs/implementation-plan.md)
- [Task tracker](docs/tasks.md)
- [Deferred ideas and backlog](docs/backlog.md)
- [Work log](docs/work-log.md)
- [M0 milestone report](docs/reports/m0.md)
- [M1 correctness report](docs/reports/m1.md)
- [M2 storage and residency report](docs/reports/m2.md)
