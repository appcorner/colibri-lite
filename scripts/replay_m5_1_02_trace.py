"""Replay the authoritative M5.1-00 order through the Rust ExpertCache.

The temporary TSV is only an input adapter. The cache implementation and LRU
decisions are production clr-storage code; the trace is never reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

TRACE_SHA = "f3f87f05d15424030c9261cdf3e93bd72e9c006a55303bc0c28a92a4fb3ff2d0"
RESULTS = Path("models/qwen3-30b-a3b/m5.1-01-memory-hierarchy-results-v1.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("models/qwen3-30b-a3b/m5.1-00-ordered-expert-trace-v1.json"))
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", action="store_true", help="use release build")
    args = parser.parse_args()
    trace_bytes = args.trace.read_bytes()
    actual_sha = hashlib.sha256(trace_bytes).hexdigest()
    if actual_sha != TRACE_SHA:
        raise SystemExit(f"trace hash mismatch: {actual_sha}")
    trace = json.loads(trace_bytes)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    budgets = {}
    for scenario in results["scenarios"]:
        if scenario.get("configuration") == "streamed_dense" and scenario.get("policy") == "lru" and scenario["budget_gib"] in (8, 16):
            budgets[scenario["budget_gib"]] = scenario["usable_expert_cache_bytes"]
    if set(budgets) != {8, 16}:
        raise SystemExit("missing streamed-dense LRU reference budgets")
    with tempfile.TemporaryDirectory(prefix="colibri-m5.1-02-") as directory:
        trace_tsv = Path(directory) / "ordered-expert-trace.tsv"
        with trace_tsv.open("w", encoding="utf-8", newline="\n") as handle:
            for record in trace["records"]:
                handle.write(f"{record['layer_index']}\t{record['expert_id']}\t{record['payload_bytes']}\n")
        rows = []
        for gib in (8, 16):
            command = ["cargo", "run", "--quiet", "-p", "clr-storage", "--example", "replay_expert_trace"]
            if args.release:
                command.append("--release")
            command += ["--", "--trace", str(trace_tsv), "--budget-bytes", str(budgets[gib])]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            metrics = json.loads(completed.stdout.strip())
            reference = next(s for s in results["scenarios"] if s.get("configuration") == "streamed_dense" and s.get("policy") == "lru" and s["budget_gib"] == gib)
            counter_agreement = all(
                metrics[key] == reference[value]
                for key, value in (
                    ("hits", "hits"),
                    ("misses", "misses"),
                    ("loads", "loads"),
                    ("evictions", "evictions"),
                )
            )
            row = {"budget_gib": gib, "configured_payload_budget_bytes": budgets[gib], "runtime": metrics, "simulation": {"hits": reference["hits"], "misses": reference["misses"], "loads": reference["loads"], "evictions": reference["evictions"], "peak_resident_bytes_including_metadata": reference["peak_cache_bytes"], "bytes_avoided": reference["expert_bytes_avoided"]}, "counter_agreement": counter_agreement, "peak_payload_within_budget": metrics["peak_resident_bytes"] <= budgets[gib], "peak_metadata_accounting_delta_bytes": reference["peak_cache_bytes"] - metrics["peak_resident_bytes"]}
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "colibri-qwen3-moe-m5.1-02-cache-replay-v1", "trace_sha256": actual_sha, "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(), "operating_points": rows}, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
