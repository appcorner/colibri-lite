"""Validate the authoritative M5.1-00 ordered expert trace.

The validator consumes the captured execution order. It never reconstructs
order from router summaries or aggregate counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_BASELINE_ID = "qwen3-30b-a3b-colibri-f32-windows-x64-v1"
EXPECTED_TRACE_SCHEMA = "colibri-qwen3-moe-m5.1-00-ordered-expert-trace-v1"
EXPECTED_ARTIFACT_ROOT = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
EXPECTED_INPUT = [9707, 11, 1879, 0]
EXPECTED_GENERATED = [1096, 374]
EXPECTED_LAYER_COUNT = 48
EXPECTED_EXPERT_COUNT = 128
EXPECTED_PAYLOAD_BYTES = 18_874_368


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reuse_distances(records: list[dict[str, Any]]) -> list[int]:
    last: dict[str, int] = {}
    distances: list[int] = []
    for index, record in enumerate(records):
        key = record["layer_expert_key"]
        if key in last:
            distances.append(index - last[key])
        last[key] = index
    return distances


def validate(trace_path: Path, baseline_path: Path, expert_manifest_path: Path) -> dict[str, Any]:
    trace = load_json(trace_path)
    baseline = load_json(baseline_path)
    expert_manifest = load_json(expert_manifest_path)
    assert trace["schema"] == EXPECTED_TRACE_SCHEMA
    assert trace["baseline_id"] == EXPECTED_BASELINE_ID
    assert trace["canonical_artifact_root_sha256"] == EXPECTED_ARTIFACT_ROOT
    assert trace["input_token_ids"] == EXPECTED_INPUT
    assert trace["expected_generated_token_ids"] == EXPECTED_GENERATED
    assert trace["classification"].startswith("M5 measurement supplement")

    baseline_cache = baseline["expert_cache"]
    assert baseline["fixture"]["generated_token_ids"] == EXPECTED_GENERATED
    assert baseline_cache["occurrences"] == 2304
    assert baseline_cache["unique_layer_expert_requests"] == 1332
    assert baseline_cache["hits"] == 0
    assert baseline_cache["loads"] == 2304
    assert baseline_cache["evictions"] == 2303
    assert baseline["phases"][-1]["expert_bytes"] == 43_486_543_872

    payload_sizes = {
        (item["layer"], item["expert"]): item["payload_length"]
        for item in expert_manifest["experts"]
    }
    records = trace["records"]
    assert len(records) == trace["requested_trace_count"] == 2304
    keys: list[str] = []
    loaded_bytes = 0
    evictions = 0
    hits = 0
    loads = 0
    per_layer = [0] * EXPECTED_LAYER_COUNT
    for ordinal, record in enumerate(records):
        assert record["global_ordinal"] == ordinal
        assert 0 <= record["layer_index"] < EXPECTED_LAYER_COUNT
        assert 0 <= record["expert_id"] < EXPECTED_EXPERT_COUNT
        expected_key = f"layer.{record['layer_index']}.expert.{record['expert_id']}"
        assert record["layer_expert_key"] == expected_key
        assert record["payload_bytes"] == payload_sizes[(record["layer_index"], record["expert_id"])]
        assert record["payload_bytes"] == EXPECTED_PAYLOAD_BYTES
        assert record["cache_hit"] is False
        assert record["loaded"] is True
        assert record["evictions_caused"] in (0, 1)
        assert record["phase"] in ("prefill", "decode")
        assert record["input_token_id"] == EXPECTED_INPUT[record["absolute_position"]] if record["absolute_position"] < 4 else record["input_token_id"] in (1096, 374)
        keys.append(expected_key)
        loaded_bytes += record["payload_bytes"]
        evictions += record["evictions_caused"]
        hits += int(record["cache_hit"])
        loads += int(record["loaded"])
        per_layer[record["layer_index"]] += 1

    assert len(set(keys)) == 1332
    assert hits == baseline_cache["hits"]
    assert loads == baseline_cache["loads"]
    assert evictions == baseline_cache["evictions"]
    assert loaded_bytes == baseline["phases"][-1]["expert_bytes"]
    distances = reuse_distances(records)
    frozen_distance = baseline_cache["reuse_distance"]
    assert len(distances) == baseline_cache["repeated_across_tokens"]
    assert min(distances) == frozen_distance["minimum"]
    assert sorted(distances)[(len(distances) - 1) // 2] == frozen_distance["median"]
    assert max(distances) == frozen_distance["maximum"]
    assert sum(distance <= 384 for distance in distances) == frozen_distance["at_most_384"]
    assert sum(385 <= distance <= 768 for distance in distances) == frozen_distance["from_385_through_768"]
    assert sum(distance > 768 for distance in distances) == frozen_distance["above_768"]
    aggregate = {
        "schema": "colibri-qwen3-moe-m5.1-00-trace-derived-aggregate-v1",
        "schema_version": 1,
        "trace_sha256": sha256(trace_path),
        "baseline_id": trace["baseline_id"],
        "record_count": len(records),
        "unique_layer_expert_keys": len(set(keys)),
        "cache_hits": hits,
        "cache_misses": len(records) - hits,
        "loads": loads,
        "evictions": evictions,
        "expert_logical_read_bytes": loaded_bytes,
        "reuse_distance": {
            "minimum": min(distances),
            "median": sorted(distances)[(len(distances) - 1) // 2],
            "maximum": max(distances),
            "at_most_384": sum(distance <= 384 for distance in distances),
            "from_385_through_768": sum(385 <= distance <= 768 for distance in distances),
            "above_768": sum(distance > 768 for distance in distances),
        },
        "per_layer_request_counts": per_layer,
        "accounting": {
            "requests_equal_hits_plus_misses": len(records) == hits + (len(records) - hits),
            "misses_equal_loads": len(records) - hits == loads,
            "expert_bytes_equal_loaded_payload_sum": loaded_bytes == 43_486_543_872,
            "capacity_one_evictions_equal_loads_minus_first": evictions == loads - 1,
        },
    }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    aggregate = validate(args.trace, args.baseline, args.expert_manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(aggregate, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
