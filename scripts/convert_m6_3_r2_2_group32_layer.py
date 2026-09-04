#!/usr/bin/env python3
"""Convert one canonical F32 expert layer into the frozen R2.2 group32 format."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

EXPERTS = 128
HIDDEN = 2048
INTERMEDIATE = 768
GROUP = 32
SOURCE_EXPERT_BYTES = 18_874_368
SOURCE_LAYER_BYTES = 2_415_919_104
PACKED_EXPERT_BYTES = 5_308_416
PACKED_LAYER_BYTES = 679_477_248
PROJECTIONS = ((INTERMEDIATE, HIDDEN), (INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize_projection(values: np.ndarray) -> tuple[bytes, bytes]:
    rows, columns = values.shape
    assert columns % GROUP == 0
    assert values.dtype == np.dtype("<f4") or values.dtype == np.dtype("float32")
    assert np.isfinite(values).all()
    grouped = values.reshape(rows, columns // GROUP, GROUP)
    maxima = np.max(np.abs(grouped), axis=2).astype(np.float32, copy=False)
    scales = np.divide(maxima, np.float32(127.0), dtype=np.float32)
    denom = np.where(scales == 0.0, np.float32(1.0), scales).astype(np.float32, copy=False)
    normalized = np.divide(grouped, denom[..., None], dtype=np.float32)
    rounded = np.copysign(
        np.floor(np.abs(normalized) + np.float32(0.5)), normalized
    ).astype(np.float32, copy=False)
    np.clip(rounded, np.float32(-127.0), np.float32(127.0), out=rounded)
    quantized = rounded.astype(np.int8, copy=False)
    return quantized.reshape(rows, columns).tobytes(order="C"), scales.astype("<f4").tobytes(order="C")


def convert(source: Path, output: Path, expected_source_sha: str) -> tuple[str, str]:
    if source.stat().st_size != SOURCE_LAYER_BYTES:
        raise RuntimeError("canonical source layer byte length mismatch")
    actual_source_sha = sha256_file(source)
    if actual_source_sha != expected_source_sha:
        raise RuntimeError("canonical source layer SHA-256 mismatch")
    if output.exists():
        raise RuntimeError("output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        raise RuntimeError("partial output already exists")

    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            for expert in range(EXPERTS):
                for rows, columns in PROJECTIONS:
                    count = rows * columns
                    values = np.fromfile(reader, dtype="<f4", count=count)
                    if values.size != count:
                        raise RuntimeError(f"short source read at expert {expert}")
                    values = values.reshape(rows, columns)
                    qbytes, sbytes = quantize_projection(values)
                    writer.write(qbytes)
                    writer.write(sbytes)
                    digest.update(qbytes)
                    digest.update(sbytes)
            if reader.tell() != SOURCE_LAYER_BYTES:
                raise RuntimeError("source layer consumption mismatch")
            writer.flush()
        if temporary.stat().st_size != PACKED_LAYER_BYTES:
            raise RuntimeError("packed layer byte length mismatch")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return actual_source_sha, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, choices=range(48))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_source_sha256) != 64:
        raise RuntimeError("expected source SHA-256 must be 64 hex characters")
    if any(ch not in "0123456789abcdef" for ch in args.expected_source_sha256):
        raise RuntimeError("expected source SHA-256 must be lowercase hex")
    if SOURCE_EXPERT_BYTES * EXPERTS != SOURCE_LAYER_BYTES:
        raise RuntimeError("source geometry constants are inconsistent")
    if PACKED_EXPERT_BYTES * EXPERTS != PACKED_LAYER_BYTES:
        raise RuntimeError("packed geometry constants are inconsistent")
    source_sha, output_sha = convert(
        args.source, args.output, args.expected_source_sha256
    )
    print(f"layer={args.layer}")
    print(f"source_sha256={source_sha}")
    print(f"output_sha256={output_sha}")
    print(f"output_bytes={args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
