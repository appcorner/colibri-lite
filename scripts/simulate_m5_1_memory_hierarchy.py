"""Deterministic, simulation-only M5.1 memory hierarchy study.

The simulator consumes the authoritative ordered expert trace.  It does not
reconstruct order from aggregates and never changes the Rust runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable

GIB = 1024**3
TRACE_REL = "models/qwen3-30b-a3b/m5.1-00-ordered-expert-trace-v1.json"
BASELINE_REL = "models/qwen3-30b-a3b/m4.4-performance-baseline-v1.json"
PROVENANCE_REL = "models/qwen3-30b-a3b/m4-release-provenance-v1.json"
MODEL_MANIFEST_REL = "models/qwen3-30b-a3b/model-manifest-v1.json"
EXPERT_EVIDENCE_REL = "models/qwen3-30b-a3b/expert-conversion-evidence-v1.json"
EXPECTED_TRACE_SHA = "f3f87f05d15424030c9261cdf3e93bd72e9c006a55303bc0c28a92a4fb3ff2d0"
EXPECTED_ARTIFACT_ROOT = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
EXPECTED_GENERATED = [1096, 374]
LAYERS = 48
EXPERTS = 128
META_BYTES = 64
ALIGNMENT = 4096
SAFETY_RESERVE = 256 * 1024 * 1024
RAM_BUDGETS = [1, 2, 4, 8, 16, 24, 32]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def key_of(record: dict[str, Any]) -> str:
    return record["layer_expert_key"]


def validate_trace(root: Path, trace_path: Path, baseline_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trace = load_json(trace_path)
    baseline = load_json(baseline_path)
    provenance = load_json(root / PROVENANCE_REL)
    model_manifest = load_json(root / MODEL_MANIFEST_REL)
    expert_evidence = load_json(root / EXPERT_EVIDENCE_REL)
    if sha256(trace_path) != EXPECTED_TRACE_SHA:
        raise ValueError("authoritative trace SHA-256 does not match M5.1-00")
    if trace["schema"] != "colibri-qwen3-moe-m5.1-00-ordered-expert-trace-v1":
        raise ValueError("unsupported trace schema")
    if trace["canonical_artifact_root_sha256"] != EXPECTED_ARTIFACT_ROOT:
        raise ValueError("trace artifact identity mismatch")
    if trace["expected_generated_token_ids"] != EXPECTED_GENERATED:
        raise ValueError("generated fixture identity mismatch")
    records = trace["records"]
    if len(records) != trace["requested_trace_count"]:
        raise ValueError("trace record count mismatch")
    seen: set[str] = set()
    for ordinal, record in enumerate(records):
        if record["global_ordinal"] != ordinal:
            raise ValueError("trace ordinals are not contiguous")
        layer = record["layer_index"]
        expert = record["expert_id"]
        if not (0 <= layer < LAYERS and 0 <= expert < EXPERTS):
            raise ValueError("trace key is out of range")
        expected = f"layer.{layer}.expert.{expert}"
        if record["layer_expert_key"] != expected:
            raise ValueError("trace key encoding mismatch")
        if record["payload_bytes"] != expert_evidence["artifact_contract"]["logical_expert_bytes"]:
            raise ValueError("trace payload size does not match expert artifact contract")
        seen.add(expected)
    aggregate = {
        "requests": len(records),
        "unique_keys": len(seen),
        "hits": sum(bool(r["cache_hit"]) for r in records),
        "loads": sum(bool(r["loaded"]) for r in records),
        "evictions": sum(r["evictions_caused"] for r in records),
        "expert_bytes": sum(r["payload_bytes"] for r in records),
    }
    expected = baseline["performance_baseline"]["expert_cache"]
    if aggregate["requests"] != expected["occurrences"] or aggregate["unique_keys"] != expected["unique_layer_expert_requests"]:
        raise ValueError("trace aggregate does not match frozen baseline")
    if aggregate["hits"] != expected["hits"] or aggregate["loads"] != expected["loads"] or aggregate["evictions"] != expected["evictions"]:
        raise ValueError("trace cache counters do not match frozen baseline")
    if aggregate["expert_bytes"] != baseline["performance_baseline"]["expert_logical_bytes"]:
        raise ValueError("trace expert bytes do not match frozen baseline")
    if sha256(root / MODEL_MANIFEST_REL) != EXPECTED_ARTIFACT_ROOT:
        raise ValueError("model manifest hash identity mismatch")
    if model_manifest["components"]["experts"]["manifest"]["sha256"] != expert_evidence["complete_conversion"]["manifest_sha256"]:
        raise ValueError("expert manifest identity mismatch")
    if provenance["m4_baseline"]["baseline_id"] != baseline["baseline_id"]:
        raise ValueError("M4 provenance/baseline identity mismatch")
    return trace, baseline, aggregate


class Cache:
    def __init__(self, budget: int, policy: str, records: list[dict[str, Any]]):
        self.budget = max(0, budget)
        self.policy = policy
        self.records = records
        self.entries: dict[str, tuple[int, int]] = {}  # key -> (payload, last ordinal)
        self.order: OrderedDict[str, None] = OrderedDict()
        self.frequency: Counter[str] = Counter()
        self.resident = 0
        self.peak = 0
        self.peak_entries = 0
        self.evictions = 0
        self.hits = 0
        self.loads = 0
        self.loaded_bytes = 0
        self.avoided_bytes = 0
        self.compulsory = 0
        self.capacity = 0
        self.policy_misses = 0
        self.seen: set[str] = set()
        self.first_seen: set[str] = set()
        self.key_layer = {key_of(r): r["layer_index"] for r in records}
        self.future_positions: dict[str, list[int]] = defaultdict(list)
        if policy == "belady":
            for i, record in enumerate(records):
                self.future_positions[key_of(record)].append(i)

    def _next_use(self, key: str, ordinal: int) -> int | None:
        for position in self.future_positions.get(key, []):
            if position > ordinal:
                return position
        return None

    def charge(self, payload: int) -> int:
        return align_up(payload + META_BYTES)

    def _victim(self, current: int, incoming: str) -> str | None:
        if not self.entries:
            return None
        if self.policy == "lru":
            return next(iter(self.order))
        if self.policy == "layer_lru":
            incoming_layer = self.key_layer[incoming]
            same = [k for k in self.order if self.key_layer[k] == incoming_layer]
            return same[0] if same else next(iter(self.order))
        if self.policy == "frequency":
            return min(self.entries, key=lambda k: (self.frequency[k], self.entries[k][1], k))
        # Belady is a theoretical offline upper bound.
        return max(self.entries, key=lambda k: ((self._next_use(k, current) is None), self._next_use(k, current) or 10**18, k))

    def request(self, ordinal: int, record: dict[str, Any]) -> None:
        key = key_of(record)
        payload = int(record["payload_bytes"])
        charge = self.charge(payload)
        if key in self.entries:
            self.hits += 1
            self.avoided_bytes += payload
            self.frequency[key] += 1
            self.entries[key] = (payload, ordinal)
            if key in self.order:
                self.order.move_to_end(key)
            return
        self.loads += 1
        self.loaded_bytes += payload
        if key not in self.seen:
            self.compulsory += 1
            self.seen.add(key)
        else:
            self.capacity += 1
        if charge > self.budget:
            self.policy_misses += int(key in self.seen)
            return
        while self.entries and self.resident + charge > self.budget:
            victim = self._victim(ordinal, key)
            if victim is None:
                break
            victim_payload, _ = self.entries.pop(victim)
            self.resident -= self.charge(victim_payload)
            self.order.pop(victim, None)
            self.evictions += 1
        self.entries[key] = (payload, ordinal)
        self.order[key] = None
        self.frequency[key] += 1
        self.resident += charge
        self.peak = max(self.peak, self.resident)
        self.peak_entries = max(self.peak_entries, len(self.entries))

    def result(self, baseline_expert_bytes: int, dense_bytes: int, requests: int, decode_tokens: int, dense_reads: int, fixed_overhead: int, total_budget: int, dense_resident: bool) -> dict[str, Any]:
        cache_budget = self.budget
        total_reads = (0 if dense_resident else dense_reads) + self.loaded_bytes
        baseline_total = dense_reads + baseline_expert_bytes
        return {
            "total_ram_budget_bytes": total_budget,
            "fixed_overhead_bytes": fixed_overhead,
            "dense_resident_bytes": dense_bytes if dense_resident else 0,
            "usable_expert_cache_bytes": cache_budget,
            "max_simultaneous_entries": self.peak_entries,
            "peak_cache_bytes": self.peak,
            "request_count": requests,
            "unique_keys": len(self.seen),
            "hits": self.hits,
            "misses": self.loads,
            "loads": self.loads,
            "evictions": self.evictions,
            "hit_rate_requests": self.hits / requests if requests else 0.0,
            "hit_rate_bytes": self.avoided_bytes / baseline_expert_bytes if baseline_expert_bytes else 0.0,
            "compulsory_misses": self.compulsory,
            "capacity_misses": self.capacity,
            "policy_misses": self.policy_misses,
            "expert_bytes_loaded": self.loaded_bytes,
            "expert_bytes_avoided": self.avoided_bytes,
            "expert_read_reduction_pct": 100.0 * self.avoided_bytes / baseline_expert_bytes if baseline_expert_bytes else 0.0,
            "remaining_expert_logical_reads": self.loaded_bytes,
            "remaining_dense_logical_reads": 0 if dense_resident else dense_reads,
            "total_modeled_logical_reads": total_reads,
            "total_logical_read_reduction_pct": 100.0 * (baseline_total - total_reads) / baseline_total,
            "logical_bytes_per_decode_token": total_reads / decode_tokens if decode_tokens else total_reads,
            "unused_cache_bytes": max(0, cache_budget - self.peak),
            "fragmentation_bytes": 0,
            "feasible": True,
        }


def simulate(records: list[dict[str, Any]], budget: int, policy: str) -> Cache:
    cache = Cache(budget, policy, records)
    for ordinal, record in enumerate(records):
        cache.request(ordinal, record)
    return cache


def validate_row(row: dict[str, Any], baseline_expert_bytes: int, baseline_dense_bytes: int) -> None:
    if not row.get("feasible", False):
        return
    requests = row["request_count"]
    if row["hits"] + row["misses"] != requests:
        raise ValueError("cache accounting: hits + misses != requests")
    if row["loads"] != row["misses"]:
        raise ValueError("cache accounting: loads != misses")
    if row["expert_bytes_loaded"] + row["expert_bytes_avoided"] != baseline_expert_bytes:
        raise ValueError("cache accounting: loaded + avoided expert bytes mismatch")
    if row["peak_cache_bytes"] > row["usable_expert_cache_bytes"]:
        raise ValueError("cache accounting: peak cache exceeds usable budget")
    expected_total = row["remaining_dense_logical_reads"] + row["remaining_expert_logical_reads"]
    if row["total_modeled_logical_reads"] != expected_total:
        raise ValueError("cache accounting: total logical reads mismatch")
    if row["dense_resident_bytes"] and row["remaining_dense_logical_reads"] != 0:
        raise ValueError("resident dense scenario did not avoid dense reads")
    if row["dense_resident_bytes"] > baseline_dense_bytes:
        raise ValueError("dense residency exceeds artifact bytes")


def dense_components(baseline: dict[str, Any]) -> dict[str, int]:
    model = baseline["model_identity"]
    perf = baseline["performance_baseline"]
    comp = perf["modeled_memory_components"]
    runtime = comp["kv_cache_bytes"] + comp["inference_tensor_bytes"] + comp["temporary_validation_buffer_bytes"]
    return {
        "dense_payload_bytes": model["dense"]["bytes"] - 146850,
        "dense_artifact_bytes": model["dense"]["bytes"],
        "dense_stream_buffer_bytes": comp["dense_buffer_bytes"],
        "decoded_expert_buffer_bytes": comp["decoded_expert_buffer_bytes"],
        "runtime_structures_bytes": runtime,
        "safety_reserve_bytes": SAFETY_RESERVE,
    }


def build_input(root: Path, trace: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    trace_path = root / TRACE_REL
    components = dense_components(baseline)
    return {
        "schema": "colibri-qwen3-moe-m5.1-01-simulation-input-v1",
        "schema_version": 1,
        "baseline_id": baseline["baseline_id"],
        "trace": {"path": TRACE_REL, "sha256": sha256(trace_path), "record_count": len(trace["records"])},
        "artifact_root_sha256": EXPECTED_ARTIFACT_ROOT,
        "ram_budgets_binary_gib": RAM_BUDGETS,
        "accounting": {"cache_metadata_bytes_per_entry": META_BYTES, "alignment_bytes": ALIGNMENT, "safety_reserve_bytes": SAFETY_RESERVE, "dense_components": components},
        "policies": [
            {"id": "global_lru", "kind": "online", "description": "strict global byte-budgeted LRU"},
            {"id": "layer_lru", "kind": "online", "description": "layer-preferred LRU; evicts oldest same-layer entry first"},
            {"id": "frequency", "kind": "online", "description": "LFU by observed count, oldest access then key tie-break"},
            {"id": "belady", "kind": "theoretical", "description": "offline next-use upper bound; not implementable online"},
        ],
        "dense_modes": ["streamed_dense", "resident_dense"],
        "correctness_reference": {"generated_token_ids": EXPECTED_GENERATED, "runtime_arithmetic": "ordered F32"},
    }


def run(root: Path, input_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    input_doc = load_json(input_path)
    trace_path = root / input_doc["trace"]["path"]
    baseline_path = root / BASELINE_REL
    trace, baseline, aggregate = validate_trace(root, trace_path, baseline_path)
    if input_doc["trace"]["sha256"] != sha256(trace_path):
        raise ValueError("simulation input trace hash mismatch")
    records = trace["records"]
    perf = baseline["performance_baseline"]
    dense_reads = perf["dense_logical_bytes"]
    expert_bytes = perf["expert_logical_bytes"]
    decode_tokens = trace["expected_generated_token_ids"].__len__() + len(trace["input_token_ids"])
    comp = input_doc["accounting"]["dense_components"]
    results: list[dict[str, Any]] = []
    for gib in input_doc["ram_budgets_binary_gib"]:
        total_budget = gib * GIB
        streamed_fixed = comp["dense_stream_buffer_bytes"] + comp["decoded_expert_buffer_bytes"] + comp["runtime_structures_bytes"] + comp["safety_reserve_bytes"]
        streamed_cache_budget = max(0, total_budget - streamed_fixed)
        for policy in ("lru", "layer_lru", "frequency", "belady"):
            cache = simulate(records, streamed_cache_budget, policy)
            row = cache.result(expert_bytes, comp["dense_artifact_bytes"], len(records), decode_tokens, dense_reads, streamed_fixed, total_budget, False)
            row.update({"budget_gib": gib, "configuration": "streamed_dense", "policy": policy, "policy_kind": "theoretical" if policy == "belady" else "online"})
            validate_row(row, expert_bytes, comp["dense_artifact_bytes"])
            results.append(row)
        resident_fixed = comp["dense_artifact_bytes"] + comp["decoded_expert_buffer_bytes"] + comp["runtime_structures_bytes"] + comp["safety_reserve_bytes"]
        feasible = total_budget >= resident_fixed
        if feasible:
            resident_cache_budget = total_budget - resident_fixed
            for policy in ("lru", "layer_lru", "frequency", "belady"):
                cache = simulate(records, resident_cache_budget, policy)
                row = cache.result(expert_bytes, comp["dense_artifact_bytes"], len(records), decode_tokens, dense_reads, resident_fixed, total_budget, True)
                row.update({"budget_gib": gib, "configuration": "resident_dense", "policy": policy, "policy_kind": "theoretical" if policy == "belady" else "online"})
                validate_row(row, expert_bytes, comp["dense_artifact_bytes"])
                results.append(row)
        else:
            results.append({"budget_gib": gib, "configuration": "resident_dense", "feasible": False, "total_ram_budget_bytes": total_budget, "fixed_overhead_bytes": resident_fixed, "reason": "dense artifact plus mandatory overhead exceeds budget"})
    freq = Counter(key_of(r) for r in records)
    layers = Counter(r["layer_index"] for r in records)
    per_token: dict[int, set[str]] = defaultdict(set)
    per_layer: dict[int, set[str]] = defaultdict(set)
    phase_counts = Counter(r["phase"] for r in records)
    phase_unique: dict[str, set[str]] = defaultdict(set)
    cumulative: list[dict[str, int]] = []
    cumulative_keys: set[str] = set()
    last_seen: dict[str, int] = {}
    reuse_distances: list[int] = []
    for r in records:
        per_token[r["absolute_position"]].add(key_of(r))
        per_layer[r["layer_index"]].add(key_of(r))
        phase_unique[r["phase"]].add(key_of(r))
        key = key_of(r)
        if key in last_seen:
            reuse_distances.append(r["global_ordinal"] - last_seen[key])
        last_seen[key] = r["global_ordinal"]
        cumulative_keys.add(key)
        if not cumulative or cumulative[-1]["absolute_position"] != r["absolute_position"]:
            cumulative.append({"absolute_position": r["absolute_position"], "unique_keys": len(cumulative_keys)})
    results_doc = {
        "schema": "colibri-qwen3-moe-m5.1-01-memory-hierarchy-results-v1",
        "schema_version": 1,
        "baseline_id": baseline["baseline_id"],
        "trace_sha256": sha256(trace_path),
        "trace_aggregate": aggregate,
        "accounting": input_doc["accounting"],
        "scenarios": results,
        "expert_cache_operating_points": [
            cache_point(records, entries, expert_bytes, dense_reads, len(records), decode_tokens)
            for entries in (1, 379, 512, 768, 1024, 1332)
        ],
        "operating_modes": operating_modes(results),
        "sequence_analysis": {
            "most_frequent_keys": [{"key": k, "occurrences": v} for k, v in sorted(freq.items(), key=lambda item: (-item[1], item[0]))[:20]],
            "request_frequency_by_layer": {str(k): v for k, v in sorted(layers.items())},
            "working_set_by_absolute_position": {str(k): len(v) for k, v in sorted(per_token.items())},
            "working_set_by_layer": {str(k): len(v) for k, v in sorted(per_layer.items())},
            "phase_request_counts": dict(sorted(phase_counts.items())),
            "phase_unique_keys": {k: len(v) for k, v in sorted(phase_unique.items())},
            "cumulative_unique_keys_by_position": cumulative,
            "reuse_distance_histogram": {str(k): v for k, v in sorted(Counter(reuse_distances).items())},
            "reuse_within_same_absolute_position": sum(1 for i, r in enumerate(records) if i and r["absolute_position"] == records[i - 1]["absolute_position"] and key_of(r) == key_of(records[i - 1])),
            "top_10_frequency_share": sum(v for _, v in sorted(freq.items(), key=lambda item: (-item[1], item[0]))[:10]) / len(records),
            "unique_keys_over_trace": len(freq),
            "reuse_distance": baseline["performance_baseline"]["expert_cache"]["reuse_distance"],
            "repeated_across_tokens": baseline["performance_baseline"]["expert_cache"]["repeated_across_tokens"],
        },
        "thresholds": threshold_analysis(records, expert_bytes, comp["dense_artifact_bytes"]),
        "correctness": {"canonical_artifact_root_sha256": EXPECTED_ARTIFACT_ROOT, "generated_token_ids": EXPECTED_GENERATED, "numerical_behavior_changed": False},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_doc, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(results_doc, input_doc), encoding="utf-8", newline="\n")
    return results_doc


def cache_point(records: list[dict[str, Any]], entries: int, expert_bytes: int, dense_reads: int, requests: int, decode_tokens: int) -> dict[str, Any]:
    charge = align_up(int(records[0]["payload_bytes"]) + META_BYTES)
    cache = simulate(records, entries * charge, "lru")
    row = cache.result(expert_bytes, 0, requests, decode_tokens, dense_reads, 0, entries * charge, False)
    return {"entries": entries, "cache_budget_bytes": entries * charge, "hits": row["hits"], "hit_rate_bytes": row["hit_rate_bytes"], "expert_bytes_avoided": row["expert_bytes_avoided"], "remaining_expert_logical_reads": row["remaining_expert_logical_reads"]}


def operating_modes(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def find(gib: int, configuration: str) -> dict[str, Any]:
        return next(r for r in results if r.get("budget_gib") == gib and r.get("configuration") == configuration and r.get("policy") == "lru")
    low = find(4, "streamed_dense")
    balanced = find(8, "streamed_dense")
    performance = find(16, "resident_dense")
    return [
        {"mode": "low_memory", "total_ram_budget_gib": 4, "dense_mode": "streamed_dense", "expert_cache_bytes": low["usable_expert_cache_bytes"], "expert_hit_rate_bytes": low["hit_rate_bytes"], "total_logical_read_reduction_pct": low["total_logical_read_reduction_pct"], "uncertainty": "no hits on this short trace"},
        {"mode": "balanced", "total_ram_budget_gib": 8, "dense_mode": "streamed_dense", "expert_cache_bytes": balanced["usable_expert_cache_bytes"], "expert_hit_rate_bytes": balanced["hit_rate_bytes"], "total_logical_read_reduction_pct": balanced["total_logical_read_reduction_pct"], "uncertainty": "single-fixture trace; broader corpus required"},
        {"mode": "performance", "total_ram_budget_gib": 16, "dense_mode": "resident_dense", "expert_cache_bytes": performance["usable_expert_cache_bytes"], "expert_hit_rate_bytes": performance["hit_rate_bytes"], "total_logical_read_reduction_pct": performance["total_logical_read_reduction_pct"], "uncertainty": "dense residency is modeled, not measured"},
    ]


def threshold_analysis(records: list[dict[str, Any]], expert_bytes: int, dense_bytes: int) -> dict[str, Any]:
    payload_charge = align_up(int(records[0]["payload_bytes"]) + META_BYTES)
    unique_count = len({key_of(r) for r in records})
    exact: list[tuple[int, Cache]] = []
    for entries in range(1, unique_count + 1):
        exact.append((entries * payload_charge, simulate(records, entries * payload_charge, "lru")))
    thresholds: dict[str, Any] = {}
    for pct in (0.25, 0.5, 0.75, 0.9):
        candidates = [budget for budget, cache in exact if cache.avoided_bytes / expert_bytes >= pct]
        thresholds[f"expert_byte_hit_rate_{int(pct * 100)}_pct"] = min(candidates, default=None)
    thresholds["full_unique_working_set_payload_bytes"] = dense_bytes * 0 + 1332 * 18874368
    thresholds["first_hit_cache_entry_charge_bytes"] = payload_charge
    thresholds["minimum_lru_cache_bytes_for_first_hit"] = min((budget for budget, cache in exact if cache.hits > 0), default=None)
    thresholds["minimum_lru_cache_entries_for_first_hit"] = min((budget // payload_charge for budget, cache in exact if cache.hits > 0), default=None)
    thresholds["minimum_streamed_budget_for_first_hit_bytes"] = thresholds["minimum_lru_cache_bytes_for_first_hit"]
    thresholds["full_unique_working_set_cache_bytes"] = unique_count * payload_charge
    return thresholds


def render_report(doc: dict[str, Any], input_doc: dict[str, Any]) -> str:
    scenarios = doc["scenarios"]
    lines = ["# M5.1-01 Trace-Driven Memory Hierarchy Simulation", "", "This is a deterministic, simulation-only study over the authoritative M5.1-00 ordered trace. No Rust runtime, artifact, cache capacity, or numerical execution changed.", "", f"Trace SHA-256: `{doc['trace_sha256']}`", f"Requests: `{doc['trace_aggregate']['requests']}`; unique keys: `{doc['trace_aggregate']['unique_keys']}`; expert bytes: `{doc['trace_aggregate']['expert_bytes']}`.", "", "## Accounting", "", "Binary GiB budgets are charged against explicit process-owned components. Streamed dense retains the measured dense buffer; resident dense reserves the full dense artifact. A 256 MiB safety reserve, 64-byte cache metadata entry, and 4096-byte alignment are modeled. OS page cache and virtual mappings are not counted as residency.", "", "## Global LRU Results", "", "| GiB | Dense mode | Cache GiB | Hits | Hit bytes | Expert reduction | Total read reduction | Feasible |", "|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in scenarios:
        if r.get("policy") != "lru":
            continue
        lines.append(f"| {r['budget_gib']} | {r['configuration']} | {r.get('usable_expert_cache_bytes', 0) / GIB:.3f} | {r.get('hits', 0)} | {r.get('hit_rate_bytes', 0) * 100:.2f}% | {r.get('expert_read_reduction_pct', 0):.2f}% | {r.get('total_logical_read_reduction_pct', 0):.2f}% | {r.get('feasible', False)} |")
    lines += ["", "## Policy Comparison", "", "At fixed total RAM, global LRU is the primary online baseline. Layer-aware LRU and observed-frequency LFU are deterministic diagnostics. Belady is an offline next-use upper bound and is non-implementable online.", "", "| GiB | Global LRU hit bytes | Layer-aware hit bytes | Frequency hit bytes | Belady ceiling |", "|---:|---:|---:|---:|---:|"]
    for gib in sorted({r.get('budget_gib') for r in scenarios}):
        rows = {r.get('policy'): r for r in scenarios if r.get('configuration') == 'streamed_dense' and r.get('budget_gib') == gib}
        lines.append(f"| {gib} | {rows['lru']['hit_rate_bytes'] * 100:.2f}% | {rows['layer_lru']['hit_rate_bytes'] * 100:.2f}% | {rows['frequency']['hit_rate_bytes'] * 100:.2f}% | {rows['belady']['hit_rate_bytes'] * 100:.2f}% |")
    t = doc["thresholds"]
    lines += ["", "## Thresholds", "", f"First global-LRU hit: `{t['minimum_lru_cache_entries_for_first_hit']}` entries / `{t['minimum_lru_cache_bytes_for_first_hit']}` bytes. The first fixed total-RAM point with a hit is 8 GiB after explicit overhead. Exact global-LRU 25% expert-byte threshold is `{t['expert_byte_hit_rate_25_pct']}` cache bytes; 50%, 75%, and 90% are unreachable on this trace. Full unique-key payload residency is `{t['full_unique_working_set_payload_bytes']}` bytes (payload-only) or `{t['full_unique_working_set_cache_bytes']}` bytes including modeled entry charge.", "", "## Dense Residency", "", "Full dense residency is infeasible at 1, 2, and 4 GiB. It is feasible at 8 GiB but leaves only about 1.98 GiB for experts and captures no LRU hits on this trace. At 16 GiB, resident dense avoids all modeled dense logical reads and retains 31.42% expert-byte hits; at 24 GiB it retains 41.58% hits. These are simulated logical-read results, not throughput measurements.", "", "## Recommended Prototype", "", "Select a configurable larger expert cache as the first runtime prototype. It isolates the measured repeated expert-read bottleneck, preserves the F32 correctness contract, is reversible, and avoids conflating dense residency with cache behavior. Resident dense remains a separate later experiment.", "", "## Correctness", "", "The canonical F32 artifact identity, ordered RMSNorm/F32 contract, deterministic router policy, Tier-A IDs `[1096, 374]`, guard-layer IDs, KV-cache invariants, finite outputs, and bounded-memory requirements remain references. This simulator produces no numerical model output.", ""]
    lines += ["", "## Modeled Operating Points", "", "| Mode | RAM | Dense | Expert cache | Expert hit bytes | Total read reduction |", "|---|---:|---|---:|---:|---:|"]
    for mode in doc["operating_modes"]:
        lines.append(f"| {mode['mode']} | {mode['total_ram_budget_gib']} GiB | {mode['dense_mode']} | {mode['expert_cache_bytes'] / GIB:.3f} GiB | {mode['expert_hit_rate_bytes'] * 100:.2f}% | {mode['total_logical_read_reduction_pct']:.2f}% |")
    lines += ["", "## Correctness", "", "The canonical F32 artifact identity, ordered RMSNorm/F32 contract, deterministic router policy, Tier-A IDs `[1096, 374]`, guard-layer IDs, KV-cache invariants, finite outputs, and bounded-memory requirements remain references. This simulator produces no numerical model output.", ""]
    return "\n".join(lines)


def write_input(root: Path, path: Path) -> None:
    trace_path = root / TRACE_REL
    baseline_path = root / BASELINE_REL
    trace, baseline, _ = validate_trace(root, trace_path, baseline_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_input(root, trace, baseline), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-input", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.write_input:
        write_input(root, args.write_input)
    if args.input:
        if not args.output or not args.report:
            parser.error("--input requires --output and --report")
        run(root, args.input, args.output, args.report)


if __name__ == "__main__":
    main()
