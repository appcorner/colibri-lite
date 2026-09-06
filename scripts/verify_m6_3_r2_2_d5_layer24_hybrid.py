#!/usr/bin/env python3
"""Verify every D5 Layer24 hybrid expert byte-for-byte against frozen sources."""

from pathlib import Path
import argparse

EXPERTS = 128
HIDDEN = 2048
INTERMEDIATE = 768
GROUP_SIZE = 8
F32_PROJECTION_BYTES = HIDDEN * INTERMEDIATE * 4
F32_EXPERT_BYTES = F32_PROJECTION_BYTES * 3
PACKED_VALUES_BYTES = HIDDEN * INTERMEDIATE
PACKED_SCALES_BYTES = (HIDDEN * INTERMEDIATE // GROUP_SIZE) * 4
PACKED_PROJECTION_BYTES = PACKED_VALUES_BYTES + PACKED_SCALES_BYTES
PACKED_EXPERT_BYTES = PACKED_PROJECTION_BYTES * 3
HYBRID_EXPERT_BYTES = 2 * F32_PROJECTION_BYTES + PACKED_PROJECTION_BYTES


def verify(f32_path: Path, group8_path: Path, hybrid_path: Path) -> int:
    checked = 0
    with f32_path.open("rb") as f32, group8_path.open("rb") as group8, hybrid_path.open("rb") as hybrid:
        for expert in range(EXPERTS):
            f32.seek(expert * F32_EXPERT_BYTES)
            expected_gate_up = f32.read(2 * F32_PROJECTION_BYTES)
            group8.seek(expert * PACKED_EXPERT_BYTES + 2 * PACKED_PROJECTION_BYTES)
            expected_down = group8.read(PACKED_PROJECTION_BYTES)
            hybrid.seek(expert * HYBRID_EXPERT_BYTES)
            actual = hybrid.read(HYBRID_EXPERT_BYTES)
            if actual[: 2 * F32_PROJECTION_BYTES] != expected_gate_up:
                raise RuntimeError(f"F32 gate/up mismatch at expert {expert}")
            if actual[2 * F32_PROJECTION_BYTES :] != expected_down:
                raise RuntimeError(f"group8 down mismatch at expert {expert}")
            checked += len(actual)
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f32-source", type=Path, required=True)
    parser.add_argument("--group8-source", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    args = parser.parse_args()
    checked = verify(args.f32_source, args.group8_source, args.hybrid)
    print(f"experts_checked={EXPERTS}")
    print(f"hybrid_bytes_checked={checked}")
    print("byte_exact=true")


if __name__ == "__main__":
    main()
