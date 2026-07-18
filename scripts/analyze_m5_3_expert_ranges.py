#!/usr/bin/env python3
"""Analyze Qwen3 expert ranges and simulate bounded-gap read grouping.

This tool is deliberately independent of the Rust cache implementation.  It
validates the committed M5.2 trace identities, replays strict global LRU for
miss sequences, and computes deterministic range-grouping statistics.  It
does not read expert payloads and it does not change the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT_SHA256 = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
PAYLOAD_BYTES = 18_874_368
ONE_GIB = 1 << 30
TRACE_ROOT = Path("models/qwen3-30b-a3b")
CORPUS_MANIFEST = TRACE_ROOT / "m5.2-01-trace-corpus-manifest-v1.json"
SELECTED = [
    "tier_a_control",
    "tier_b_short_thai",
    "tier_b_code_newline",
    "long_english_context",
    "long_decode_english",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def write_layer47_sequence(path: Path, selected: dict[str, dict[str, Any]], ranges: dict[str, Range]) -> None:
    lines = ["fixture_id\tcache_case\tmiss_ordinal\tlayer_index\texpert_id"]
    for fixture_id, trace in selected.items():
        trace["_ranges"] = ranges
        for cache_name, budget in (("one_expert", PAYLOAD_BYTES), ("8_gib", 8 * ONE_GIB), ("16_gib", 16 * ONE_GIB)):
            misses, _ = replay_lru(trace, budget)
            for ordinal, item in enumerate(misses):
                if item.layer == 47:
                    lines.append(f"{fixture_id}\t{cache_name}\t{ordinal}\t47\t{item.expert}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


@dataclass(frozen=True)
class Range:
    key: str
    layer: int
    expert: int
    shard: int
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class ReadGroup:
    shard: int
    start: int
    end: int
    useful_bytes: int

    @property
    def bytes_read(self) -> int:
        return self.end - self.start


def trace_path(fixture: dict[str, Any]) -> Path:
    return Path(fixture["trace_path"])


def load_selected_traces(repo_root: Path) -> dict[str, dict[str, Any]]:
    corpus = load_json(repo_root / CORPUS_MANIFEST)
    if corpus["corpus_id"] != "qwen3-30b-a3b-m5.2-01-representative-expert-traces-v1":
        raise ValueError("unexpected M5.2 corpus identity")
    if corpus["model"]["canonical_artifact_root_sha256"] != ROOT_SHA256:
        raise ValueError("corpus artifact identity mismatch")
    selected: dict[str, dict[str, Any]] = {}
    for fixture in corpus["fixtures"]:
        fixture_id = fixture["fixture_id"]
        if fixture_id not in SELECTED:
            continue
        path = repo_root / trace_path(fixture)
        validate_hash(path, fixture["trace_sha256"], f"trace {fixture_id}")
        trace = load_json(path)
        if trace.get("fixture_id", fixture_id) != fixture_id:
            raise ValueError(f"trace fixture identity mismatch: {fixture_id}")
        records = trace["records"]
        if len(records) != fixture["record_count"]:
            raise ValueError(f"trace record count mismatch: {fixture_id}")
        for ordinal, record in enumerate(records):
            if record["global_ordinal"] != ordinal:
                raise ValueError(f"trace ordinal mismatch: {fixture_id} at {ordinal}")
            if record["payload_bytes"] != PAYLOAD_BYTES:
                raise ValueError(f"trace payload size mismatch: {fixture_id} at {ordinal}")
            if not 0 <= record["layer_index"] < 48 or not 0 <= record["expert_id"] < 128:
                raise ValueError(f"trace key range mismatch: {fixture_id} at {ordinal}")
        selected[fixture_id] = trace
    if list(selected) != [fixture_id for fixture_id in SELECTED if fixture_id in selected]:
        raise ValueError("selected fixture ordering is not deterministic")
    if set(selected) != set(SELECTED):
        raise ValueError(f"selected fixture set mismatch: {sorted(selected)}")
    return selected


def load_layout(artifact_root: Path) -> tuple[dict[str, Any], dict[str, Range], dict[str, Any]]:
    root_manifest = artifact_root / "model-manifest-v1.json"
    validate_hash(root_manifest, ROOT_SHA256, "canonical root manifest")
    expert_manifest_path = artifact_root / "experts" / "expert-manifest-v1.json"
    expert_manifest = load_json(expert_manifest_path)
    expert_manifest_hash = sha256(expert_manifest_path)
    ranges: dict[str, Range] = {}
    for item in expert_manifest["experts"]:
        layer = int(item["layer"])
        expert = int(item["expert"])
        key = f"layer.{layer}.expert.{expert}"
        payload_offset = int(item["payload_offset"])
        payload_length = int(item["payload_length"])
        if payload_length != PAYLOAD_BYTES:
            raise ValueError(f"unexpected payload length for {key}")
        if item["shard_id"] != layer:
            raise ValueError(f"unexpected shard mapping for {key}")
        gate = item["gate"]
        up = item["up"]
        down = item["down"]
        if gate["offset"] != 0:
            raise ValueError(f"gate offset mismatch for {key}")
        if up["offset"] != gate["offset"] + gate["length"]:
            raise ValueError(f"up is not contiguous for {key}")
        if down["offset"] != up["offset"] + up["length"]:
            raise ValueError(f"down is not contiguous for {key}")
        if down["offset"] + down["length"] != payload_length:
            raise ValueError(f"payload end mismatch for {key}")
        ranges[key] = Range(key, layer, expert, int(item["shard_id"]), payload_offset, payload_length)
    if len(ranges) != 48 * 128:
        raise ValueError(f"expert range count mismatch: {len(ranges)}")
    shard_records = expert_manifest["shards"]
    if len(shard_records) != 48:
        raise ValueError("expert shard count mismatch")
    for shard in shard_records:
        if int(shard["byte_length"]) != 128 * PAYLOAD_BYTES:
            raise ValueError(f"unexpected shard size for {shard['shard_id']}")
        path = artifact_root / "experts" / shard["path"]
        if not path.is_file() or path.stat().st_size != int(shard["byte_length"]):
            raise ValueError(f"missing or short shard {path}")
    return ranges, expert_manifest, {
        "root_manifest_sha256": ROOT_SHA256,
        "expert_manifest_sha256": expert_manifest_hash,
    }


def replay_lru(trace: dict[str, Any], budget: int) -> tuple[list[Range], dict[str, int]]:
    entries: dict[str, int] = {}
    last_used: dict[str, int] = {}
    misses: list[Range] = []
    hits = 0
    evictions = 0
    ranges = trace["_ranges"]
    for ordinal, record in enumerate(trace["records"]):
        key = record["layer_expert_key"]
        if key in entries:
            hits += 1
            last_used[key] = ordinal
            continue
        while entries and (len(entries) + 1) * PAYLOAD_BYTES > budget:
            candidate = min(entries, key=lambda item: (last_used[item], item))
            del entries[candidate]
            del last_used[candidate]
            evictions += 1
        entries[key] = PAYLOAD_BYTES
        last_used[key] = ordinal
        misses.append(ranges[key])
    requests = len(trace["records"])
    return misses, {"requests": requests, "hits": hits, "misses": requests - hits, "loads": requests - hits, "evictions": evictions}


def group_ranges(ranges: list[Range], gap: int, layer_batch: bool = False) -> list[ReadGroup]:
    if not ranges:
        return []
    groups: list[ReadGroup] = []
    if layer_batch:
        start = 0
        while start < len(ranges):
            end = start + 1
            while end < len(ranges) and ranges[end].layer == ranges[start].layer:
                end += 1
            batch = ranges[start:end]
            groups.append(ReadGroup(batch[0].shard, min(r.start for r in batch), max(r.end for r in batch), sum(r.length for r in batch)))
            start = end
        return groups
    current = ranges[0]
    useful = current.length
    for item in ranges[1:]:
        if item.shard == current.shard and item.start <= current.end + gap:
            current = Range(current.key, current.layer, current.expert, current.shard, current.start, max(current.end, item.end) - current.start)
            useful += item.length
        else:
            groups.append(ReadGroup(current.shard, current.start, current.end, useful))
            current = item
            useful = item.length
    groups.append(ReadGroup(current.shard, current.start, current.end, useful))
    return groups


def range_stats(misses: list[Range], gap: int, layer_batch: bool = False) -> dict[str, Any]:
    groups = group_ranges(misses, gap, layer_batch)
    useful = sum(item.length for item in misses)
    total = sum(group.bytes_read for group in groups)
    return {
        "read_operations": len(groups),
        "useful_payload_bytes": useful,
        "total_bytes_read": total,
        "over_read_bytes": total - useful,
        "operation_reduction": 0.0 if not misses else 1.0 - len(groups) / len(misses),
        "byte_amplification": 0.0 if useful == 0 else total / useful,
        "maximum_temporary_buffer_bytes": max((group.bytes_read for group in groups), default=0),
        "group_count_by_shard": dict(sorted(Counter(group.shard for group in groups).items())),
    }


def build_layout_summary(expert_manifest: dict[str, Any], ranges: dict[str, Range]) -> dict[str, Any]:
    lengths = [item["payload_length"] for item in expert_manifest["experts"]]
    projection_lengths = {
        name: sorted({int(item[name]["length"]) for item in expert_manifest["experts"]})
        for name in ("gate", "up", "down")
    }
    expert_offsets = defaultdict(list)
    for item in ranges.values():
        expert_offsets[item.shard].append(item.start)
    consecutive = all(
        sorted(offsets) == list(range(min(offsets), max(offsets) + PAYLOAD_BYTES, PAYLOAD_BYTES))
        for offsets in expert_offsets.values()
    )
    return {
        "shard_count": len(expert_manifest["shards"]),
        "expert_count": len(ranges),
        "shard_sizes": sorted({int(item["byte_length"]) for item in expert_manifest["shards"]}),
        "expert_payload_lengths": sorted(set(lengths)),
        "projection_lengths": projection_lengths,
        "expert_payload_contiguous": True,
        "gate_up_down_contiguous": True,
        "payload_start_mod_4096_values": sorted({item.start % 4096 for item in ranges.values()}),
        "payload_start_gcd_bytes": math.gcd(*[item.start for item in ranges.values()]),
        "expert_stride_bytes": PAYLOAD_BYTES,
        "experts_ordered_by_id_within_shard": consecutive,
        "inter_expert_gap_bytes": 0,
        "shard_policy": expert_manifest["shard_policy"],
        "artifact_dtype": expert_manifest["artifact_dtype"],
        "expert_manifest_format_version": expert_manifest["format_version"],
    }


def analyze_fixture(trace: dict[str, Any], ranges: dict[str, Range]) -> dict[str, Any]:
    trace["_ranges"] = ranges
    requests = len(trace["records"])
    keys = [record["layer_expert_key"] for record in trace["records"]]
    unique_keys = len(set(keys))
    cache_cases = {}
    for cache_name, budget in (("one_expert", PAYLOAD_BYTES), ("8_gib", 8 * ONE_GIB), ("16_gib", 16 * ONE_GIB)):
        misses, counters = replay_lru(trace, budget)
        strategies = {"no_coalescing": range_stats(misses, 0)}
        for gap in (0, 4096, 64 * 1024, 1 * 1024 * 1024):
            strategies[f"gap_{gap}_bytes"] = range_stats(misses, gap)
        strategies["one_layer_batch"] = range_stats(misses, 0, layer_batch=True)
        cache_cases[cache_name] = {
            "budget_bytes": budget,
            "miss_sequence_count": len(misses),
            "cache": counters,
            "strategies": strategies,
        }
    gaps = []
    prior: Range | None = None
    for record in trace["records"]:
        current = ranges[record["layer_expert_key"]]
        if prior is not None and prior.layer == current.layer:
            gaps.append(current.start - prior.end)
        prior = current
    return {
        "requests": requests,
        "unique_keys": unique_keys,
        "unique_key_ratio": unique_keys / requests,
        "payload_bytes_requested": requests * PAYLOAD_BYTES,
        "same_layer_adjacent_read_gaps": {
            "count": len(gaps),
            "minimum_bytes": min(gaps, default=0),
            "median_bytes": statistics.median(gaps) if gaps else 0,
            "maximum_bytes": max(gaps, default=0),
            "zero_gap_percentage": 0.0 if not gaps else 100.0 * sum(gap == 0 for gap in gaps) / len(gaps),
        },
        "cache_cases": cache_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, default=Path(r"D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1"))
    parser.add_argument("--output", type=Path, default=Path("models/qwen3-30b-a3b/m5.3-01-expert-access-results-v1.json"))
    parser.add_argument("--sequence-output", type=Path, default=Path("models/qwen3-30b-a3b/m5.3-01-layer47-miss-sequence-v1.tsv"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    artifact_root = args.artifact_root.resolve()
    selected = load_selected_traces(repo_root)
    ranges, expert_manifest, artifact_identity = load_layout(artifact_root)
    for trace in selected.values():
        trace["_ranges"] = ranges
    write_layer47_sequence(repo_root / args.sequence_output, selected, ranges)
    result = {
        "schema": "colibri-qwen3-moe-m5.3-01-expert-access-results-v1",
        "schema_version": 1,
        "task": "M5.3-01",
        "artifact": {
            "root_sha256": ROOT_SHA256,
            **artifact_identity,
        },
        "trace_contract": {
            "corpus_manifest": str(CORPUS_MANIFEST).replace("\\", "/"),
            "selected_fixture_ids": SELECTED,
            "payload_bytes": PAYLOAD_BYTES,
            "cache_policy": "strict_global_lru",
            "cache_scenarios": ["one_expert", "8_gib", "16_gib"],
            "layer47_miss_sequence": str(args.sequence_output).replace("\\", "/"),
            "serialization": "UTF-8 JSON, sorted keys, compact separators, trailing newline; no timings or machine-local paths in deterministic content",
        },
        "layout": build_layout_summary(expert_manifest, ranges),
        "fixtures": {
            fixture_id: analyze_fixture(trace, ranges)
            for fixture_id, trace in selected.items()
        },
    }
    dump_json(repo_root / args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": sha256(repo_root / args.output), "fixtures": SELECTED}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
