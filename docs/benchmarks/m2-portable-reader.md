# M2 Portable Reader Baseline

- Date: 2026-07-14
- Branch: `milestone/m2-expert-residency`
- Runtime commit before benchmark: `2fb3664`
- Rust: `rustc 1.96.1 (31fca3adb 2026-06-26)`
- OS: Microsoft Windows 11 Enterprise 10.0.26200
- CPU: 11th Gen Intel Core i7-1165G7 @ 2.80 GHz
- RAM: 51,249,209,344 bytes
- Storage: HS-SSD-FUTURE 2048G, fixed disk
- Build: Cargo release profile

## Workload

The benchmark measures the current complete verified access path, not raw disk
copy speed:

```text
open file -> seek to tensor offset -> read_exact 1 MiB -> SHA-256 verify -> close
```

- Payload: 1,048,576 bytes
- Iterations: 200
- Total verified bytes: 209,715,200
- Artifact location: Windows temporary directory
- Warm-up: compilation only; no discarded measured read
- Command:

```powershell
cargo run --release -p clr-storage --example portable_reader_bench
```

## Result

```json
{"method":"portable_open_seek_read_exact_sha256","payload_bytes":1048576,"iterations":200,"total_bytes":209715200,"elapsed_seconds":1.545990,"mib_per_second":129.367}
```

This is the baseline that any mapping candidate must rerun unchanged. Mapping
is retained only if it preserves all correctness/hash/file-lifetime behavior
and demonstrates a material benefit or simpler residency ownership.

## Decision gate

No mapping implementation exists yet, so this result alone does not prove that
mapping is beneficial. Adding mapping requires review of a new unsafe or
externally audited mapping boundary under Stop Condition 5 in `AGENTS.md`.
