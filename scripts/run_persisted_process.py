#!/usr/bin/env python3
"""Run an argv-only process and atomically persist its evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> dict[str, int | str]:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    cwd = Path(request["cwd"])
    argv = request["command"]
    evidence = Path(request["evidence_directory"])
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("command must be nonempty argv strings")
    overrides = request.get("environment", {})
    if not isinstance(overrides, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in overrides.items()
    ):
        raise ValueError("environment must contain string keys and values")

    child_environment = os.environ.copy()
    child_environment.update(overrides)
    use_existing_evidence = request.get("use_existing_evidence_directory", False)
    if not isinstance(use_existing_evidence, bool):
        raise ValueError("use_existing_evidence_directory must be boolean")
    if use_existing_evidence:
        if not evidence.is_dir():
            raise ValueError("existing evidence directory is not a directory")
    else:
        evidence.mkdir(parents=True, exist_ok=False)
    stdout_path = evidence / "stdout.log"
    stderr_path = evidence / "stderr.log"
    stdout_path.touch(exist_ok=False)
    stderr_path.touch(exist_ok=False)
    started = time.monotonic()
    record: dict[str, object] = {
        "argv": argv,
        "cwd": str(cwd),
        "environment": overrides,
        "start_utc": utc_now(),
    }
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            child = subprocess.Popen(
                argv,
                cwd=cwd,
                env=child_environment,
                stdout=stdout,
                stderr=stderr,
                shell=False,
            )
            record["child_pid"] = child.pid
            exit_code = child.wait()
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
        record.update(status="completed", exit_code=exit_code)
    except OSError as error:
        exit_code = 127
        record.update(status="launch_failed", exit_code=exit_code, error=str(error))

    record.update(
        end_utc=utc_now(),
        duration_seconds=time.monotonic() - started,
        stdout=digest(stdout_path),
        stderr=digest(stderr_path),
    )
    incomplete = evidence / "exit.json.incomplete"
    incomplete.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incomplete, evidence / "exit.json")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
