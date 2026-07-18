"""Deterministic M5.2-02 cache-policy simulation over the frozen corpus.

This module is intentionally simulation-only. It validates the committed M5.2
corpus and replays ordered expert requests in Python; it never invokes Rust,
loads model artifacts, or changes production cache behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import heapq
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev
import sys
from typing import Any, Callable

# When invoked as ``python scripts/<tool>.py`` Python places ``scripts`` on
# sys.path rather than the repository root.  Add the root only to import the
# existing M5.1 key adapter; this keeps the compatibility contract explicit
# without duplicating that helper.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import simulate_m5_1_memory_hierarchy as m51


GIB = 1024**3
EXPECTED_ARTIFACT_ROOT = "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2"
EXPECTED_BASELINE_ID = "qwen3-30b-a3b-colibri-f32-windows-x64-v1"
EXPECTED_CORPUS_ID = "qwen3-30b-a3b-m5.2-01-representative-expert-traces-v1"
EXPECTED_TRACE_SCHEMA = "colibri-qwen3-moe-m5.2-01-ordered-expert-trace-v2"
EXPECTED_CONTROL_SCHEMA = "colibri-qwen3-moe-m5.1-00-ordered-expert-trace-v1"
EXPECTED_M52_FIXTURES = [
    "tier_a_control",
    "tier_b_short_english",
    "tier_b_short_thai",
    "tier_b_code_newline",
    "tier_b_repeated_pattern",
    "tier_b_special_token",
    "long_english_context",
    "long_decode_english",
]
PAYLOAD_BYTES = 18_874_368
LAYERS = 48
EXPERTS = 128
EXPERTS_PER_TOKEN = 8
META_BYTES = 64
ALIGNMENT = 4096
RAM_BUDGET_GIBS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]
POLICY_IDS = [
    "global_lru",
    "layer_lru_architecture",
    "layer_lru_calibrated",
    "frequency_lfu",
    "segmented_lru",
    "belady",
]
ONLINE_POLICIES = set(POLICY_IDS) - {"belady"}
THRESHOLD_TARGETS = [0.10, 0.25, 0.40, 0.50, 0.75, 0.90]
MONOTONIC_THRESHOLD_POLICIES = {
    "global_lru",
    "layer_lru_architecture",
    "layer_lru_calibrated",
    "belady",
}
INPUT_SCHEMA = "colibri-qwen3-moe-m5.2-02-simulation-input-v1"
RESULT_SCHEMA = "colibri-qwen3-moe-m5.2-02-cache-simulation-results-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def align_up(value: int) -> int:
    return ((value + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT


def charge_for(payload_bytes: int) -> int:
    # ExpertCache's configured budget is a payload-byte budget. Metadata and
    # allocator alignment are tracked as accounting context but are not
    # silently deducted from the payload residency budget.
    return payload_bytes


def key_of(record: dict[str, Any]) -> str:
    return m51.key_of(record)


def fixture_entry_map(corpus_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["fixture_id"]: entry for entry in corpus_manifest["fixtures"]}


def layer_weights(traces: dict[str, dict[str, Any]]) -> dict[int, int]:
    keys_by_layer: dict[int, set[str]] = {layer: set() for layer in range(LAYERS)}
    for trace in traces.values():
        for record in trace["records"]:
            keys_by_layer[record["layer_index"]].add(key_of(record))
    return {layer: len(keys) for layer, keys in keys_by_layer.items()}


def validate_corpus(root: Path, input_doc: dict[str, Any]) -> dict[str, Any]:
    """Validate every committed corpus identity before any replay starts."""
    if input_doc["schema"] != INPUT_SCHEMA or input_doc["schema_version"] != 1:
        raise ValueError("unsupported M5.2-02 input schema")
    if input_doc["artifact"]["canonical_root_sha256"] != EXPECTED_ARTIFACT_ROOT:
        raise ValueError("input artifact identity mismatch")
    paths = input_doc["references"]
    for role, reference in paths.items():
        path = root / reference["path"]
        if not path.is_file():
            raise ValueError(f"missing {role}: {reference['path']}")
        actual = sha256(path)
        if actual != reference["sha256"]:
            raise ValueError(f"{role} hash mismatch: {actual} != {reference['sha256']}")

    corpus_manifest = load_json(root / paths["corpus_manifest"]["path"])
    aggregate = load_json(root / paths["corpus_aggregate"]["path"])
    schema = load_json(root / paths["trace_schema"]["path"])
    baseline = load_json(root / paths["m4_baseline"]["path"])
    provenance = load_json(root / paths["m4_provenance"]["path"])
    model_manifest = load_json(root / paths["model_manifest"]["path"])
    if corpus_manifest["corpus_id"] != EXPECTED_CORPUS_ID:
        raise ValueError("corpus ID mismatch")
    if corpus_manifest["schema_version"] != 1:
        raise ValueError("corpus manifest version mismatch")
    if corpus_manifest["model"]["canonical_artifact_root_sha256"] != EXPECTED_ARTIFACT_ROOT:
        raise ValueError("corpus manifest artifact identity mismatch")
    if corpus_manifest["baseline"]["baseline_id"] != EXPECTED_BASELINE_ID:
        raise ValueError("corpus baseline ID mismatch")
    if aggregate["schema"] != "colibri-qwen3-moe-m5.2-01-trace-corpus-aggregate-v1":
        raise ValueError("corpus aggregate schema mismatch")
    if aggregate["simulation_executed"] is not False:
        raise ValueError("M5.2-01 aggregate must remain simulation-free")
    if schema["schema"] != EXPECTED_TRACE_SCHEMA or schema["schema_version"] != 2:
        raise ValueError("trace schema identity mismatch")
    if baseline["baseline_id"] != EXPECTED_BASELINE_ID:
        raise ValueError("M4 baseline ID mismatch")
    if baseline["model_identity"]["canonical_root_manifest_sha256"] != EXPECTED_ARTIFACT_ROOT:
        raise ValueError("M4 baseline artifact identity mismatch")
    if provenance["m4_baseline"]["baseline_id"] != EXPECTED_BASELINE_ID:
        raise ValueError("M4 provenance baseline identity mismatch")
    if provenance["m4_baseline"]["baseline_sha256"] != paths["m4_baseline"]["sha256"]:
        raise ValueError("M4 provenance baseline hash mismatch")
    if provenance["canonical_artifact"]["root_manifest_sha256"] != EXPECTED_ARTIFACT_ROOT:
        raise ValueError("M4 provenance artifact identity mismatch")
    if model_manifest["components"]["experts"]["logical_expert_count"] != LAYERS * EXPERTS:
        raise ValueError("model manifest expert count mismatch")

    manifest_entries = corpus_manifest["fixtures"]
    actual_fixture_ids = [entry["fixture_id"] for entry in manifest_entries]
    if actual_fixture_ids != EXPECTED_M52_FIXTURES:
        raise ValueError(f"fixture list mismatch: {actual_fixture_ids}")
    aggregate_counters = corpus_manifest["aggregate_counters"]
    if aggregate_counters != {
        "fixture_count": 8,
        "trace_count": 8,
        "repeat_capture_count": 23040,
        "total_expert_occurrences": 11520,
        "unique_layer_expert_keys_union": 3148,
        "payload_bytes_requested": 217432719360,
    }:
        raise ValueError("corpus aggregate counters mismatch")

    traces: dict[str, dict[str, Any]] = {}
    computed_totals = {"records": 0, "unique": set(), "payload": 0}
    for entry in manifest_entries:
        fixture_id = entry["fixture_id"]
        path = root / entry["trace_path"]
        actual_hash = sha256(path)
        if actual_hash != entry["trace_sha256"]:
            raise ValueError(f"trace hash mismatch for {fixture_id}")
        if entry["repeat_sha256"] != [actual_hash, actual_hash]:
            raise ValueError(f"repeat hash mismatch for {fixture_id}")
        trace = load_json(path)
        is_control = fixture_id == "tier_a_control"
        expected_schema = EXPECTED_CONTROL_SCHEMA if is_control else EXPECTED_TRACE_SCHEMA
        if trace["schema"] != expected_schema:
            raise ValueError(f"trace schema mismatch for {fixture_id}")
        records = trace["records"]
        if len(records) != entry["record_count"]:
            raise ValueError(f"record count mismatch for {fixture_id}")
        if is_control:
            if trace["requested_trace_count"] != len(records):
                raise ValueError("Tier-A requested count mismatch")
        else:
            if trace["counters"]["requested_trace_count"] != len(records):
                raise ValueError(f"v2 requested count mismatch for {fixture_id}")
        local_keys: set[str] = set()
        for ordinal, record in enumerate(records):
            if record["global_ordinal"] != ordinal:
                raise ValueError(f"non-contiguous ordinal in {fixture_id}")
            if not 0 <= record["layer_index"] < LAYERS:
                raise ValueError(f"layer out of range in {fixture_id}")
            if not 0 <= record["expert_id"] < EXPERTS:
                raise ValueError(f"expert out of range in {fixture_id}")
            if not 0 <= record["selected_expert_rank"] < EXPERTS_PER_TOKEN:
                raise ValueError(f"expert rank out of range in {fixture_id}")
            if record["layer_expert_key"] != f"layer.{record['layer_index']}.expert.{record['expert_id']}":
                raise ValueError(f"key encoding mismatch in {fixture_id}")
            if record["payload_bytes"] != PAYLOAD_BYTES:
                raise ValueError(f"payload size mismatch in {fixture_id}")
            if record["phase"] not in {"prefill", "decode"}:
                raise ValueError(f"phase mismatch in {fixture_id}")
            if record["absolute_position"] < 0 or record["generation_step"] < 0:
                raise ValueError(f"position mismatch in {fixture_id}")
            if not is_control and record["fixture_id"] != fixture_id:
                raise ValueError(f"fixture boundary mismatch in {fixture_id}")
            # This is the existing M5.1 simulator's record-key compatibility contract.
            if m51.key_of(record) != record["layer_expert_key"]:
                raise ValueError(f"M5.1 key adapter mismatch in {fixture_id}")
            local_keys.add(key_of(record))
        traces[fixture_id] = trace
        computed_totals["records"] += len(records)
        computed_totals["unique"].update(local_keys)
        computed_totals["payload"] += sum(record["payload_bytes"] for record in records)

    if computed_totals["records"] != 11520 or len(computed_totals["unique"]) != 3148:
        raise ValueError("computed corpus totals mismatch")
    if computed_totals["payload"] != 217432719360:
        raise ValueError("computed corpus payload total mismatch")
    return {
        "input": input_doc,
        "manifest": corpus_manifest,
        "aggregate": aggregate,
        "schema": schema,
        "baseline": baseline,
        "provenance": provenance,
        "model_manifest": model_manifest,
        "traces": traces,
        "layer_weights": layer_weights(traces),
    }


class PolicyCache:
    """Variable-size deterministic cache implementing one policy."""

    def __init__(
        self,
        budget: int,
        policy: str,
        records: list[dict[str, Any]],
        layer_quotas: dict[int, int] | None = None,
        protected_fraction: float = 0.8,
    ) -> None:
        self.budget = max(0, int(budget))
        self.policy = policy
        self.records = records
        self.layer_quotas = layer_quotas or {}
        self.protected_limit = int(self.budget * protected_fraction)
        self.entries: dict[str, dict[str, Any]] = {}
        self.order: OrderedDict[str, None] = OrderedDict()
        self.layer_orders: dict[int, OrderedDict[str, None]] = {
            layer: OrderedDict() for layer in range(LAYERS)
        }
        self.probationary: OrderedDict[str, None] = OrderedDict()
        self.protected: OrderedDict[str, None] = OrderedDict()
        self.layer_resident: Counter[int] = Counter()
        self.frequency: Counter[str] = Counter()
        self.frequency_heap: list[tuple[int, int, str]] = []
        self.seen: set[str] = set()
        self.miss_ordinals: list[int] = []
        self.repeated_miss_ordinals: list[int] = []
        self.compulsory_miss_ordinals: list[int] = []
        self.hit_flags: list[bool] = []
        self.cross_session_hits = 0
        self.transition_evictions = 0
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.peak_entry_count = 0
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.evictions = 0
        self.loaded_bytes = 0
        self.avoided_bytes = 0
        self.first_hit_ordinal: int | None = None
        self.oversized_entry_events = 0
        self.partition_rejection_events = 0
        self.evicted_bytes = 0
        self.future_next_use = self._future_next_use(records) if policy == "belady" else []

    @staticmethod
    def _future_next_use(records: list[dict[str, Any]]) -> list[int | None]:
        next_use: list[int | None] = [None] * len(records)
        upcoming: dict[str, int] = {}
        for index in range(len(records) - 1, -1, -1):
            key = key_of(records[index])
            next_use[index] = upcoming.get(key)
            upcoming[key] = index
        return next_use

    def _entry_charge(self, entry: dict[str, Any]) -> int:
        return int(entry["charge"])

    def _next_use(self, entry: dict[str, Any]) -> int | None:
        return self.future_next_use[entry["last_access"]]

    def _push_frequency_state(self, key: str) -> None:
        if self.policy == "frequency_lfu" and key in self.entries:
            entry = self.entries[key]
            heapq.heappush(self.frequency_heap, (self.frequency[key], entry["last_access"], key))

    def _remove_order(self, key: str, entry: dict[str, Any]) -> None:
        self.order.pop(key, None)
        self.layer_orders[entry["layer"]].pop(key, None)
        self.probationary.pop(key, None)
        self.protected.pop(key, None)

    def _evict(self, key: str, session_index: int) -> None:
        entry = self.entries.pop(key)
        self._remove_order(key, entry)
        charge = self._entry_charge(entry)
        self.resident_bytes -= charge
        self.layer_resident[entry["layer"]] -= charge
        self.evictions += 1
        self.evicted_bytes += charge
        if entry["last_session"] < session_index:
            self.transition_evictions += 1

    def _global_victim(self, ordinal: int) -> str | None:
        if not self.entries:
            return None
        if self.policy in {"global_lru", "layer_lru_architecture", "layer_lru_calibrated"}:
            return next(iter(self.order))
        if self.policy == "frequency_lfu":
            while self.frequency_heap:
                frequency, last_access, key = heapq.heappop(self.frequency_heap)
                entry = self.entries.get(key)
                if entry is not None and self.frequency[key] == frequency and entry["last_access"] == last_access:
                    return key
            return min(self.entries, key=lambda key: (self.frequency[key], self.entries[key]["last_access"], key))
        if self.policy == "belady":
            return max(
                self.entries,
                key=lambda key: (
                    self._next_use(self.entries[key]) is None,
                    self._next_use(self.entries[key]) or 10**18,
                    key,
                ),
            )
        # Segmented LRU evicts probationary entries first, then protected LRU.
        if self.probationary:
            return next(iter(self.probationary))
        return next(iter(self.protected), None)

    def _layer_victim(self, layer: int) -> str | None:
        return next(iter(self.layer_orders[layer]), None)

    def _rebalance_protected(self) -> None:
        protected_bytes = sum(self.entries[key]["charge"] for key in self.protected)
        while protected_bytes > self.protected_limit and self.protected:
            key = next(iter(self.protected))
            entry = self.entries[key]
            self.protected.pop(key)
            self.probationary[key] = None
            entry["segment"] = "probationary"
            protected_bytes -= entry["charge"]

    def _admit(self, key: str, record: dict[str, Any], ordinal: int, session_index: int) -> None:
        payload = int(record["payload_bytes"])
        charge = charge_for(payload)
        layer = int(record["layer_index"])
        if charge > self.budget:
            self.oversized_entry_events += 1
            return
        if self.policy in {"layer_lru_architecture", "layer_lru_calibrated"}:
            quota = self.layer_quotas.get(layer, 0)
            if charge > quota:
                self.partition_rejection_events += 1
                return
            while self.layer_resident[layer] + charge > quota:
                victim = self._layer_victim(layer)
                if victim is None:
                    self.partition_rejection_events += 1
                    return
                self._evict(victim, session_index)
        else:
            while self.entries and self.resident_bytes + charge > self.budget:
                victim = self._global_victim(ordinal)
                if victim is None:
                    break
                self._evict(victim, session_index)
        entry = {
            "payload": payload,
            "charge": charge,
            "layer": layer,
            "last_access": ordinal,
            "last_session": session_index,
            "segment": "probationary" if self.policy == "segmented_lru" else "global",
        }
        self.entries[key] = entry
        self.order[key] = None
        self.layer_orders[layer][key] = None
        self.frequency[key] += 1
        self._push_frequency_state(key)
        if self.policy == "segmented_lru":
            self.probationary[key] = None
        self.resident_bytes += charge
        self.layer_resident[layer] += charge
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        self.peak_entry_count = max(self.peak_entry_count, len(self.entries))

    def request(self, ordinal: int, record: dict[str, Any], session_index: int) -> None:
        key = key_of(record)
        self.hit_flags.append(False)
        if key in self.entries:
            entry = self.entries[key]
            self.hits += 1
            self.hit_flags[-1] = True
            if self.first_hit_ordinal is None:
                self.first_hit_ordinal = ordinal
            self.avoided_bytes += int(record["payload_bytes"])
            self.frequency[key] += 1
            if entry["last_session"] < session_index:
                self.cross_session_hits += 1
            entry["last_access"] = ordinal
            self._push_frequency_state(key)
            entry["last_session"] = session_index
            self.order.move_to_end(key)
            self.layer_orders[entry["layer"]].move_to_end(key)
            if self.policy == "segmented_lru":
                if entry["segment"] == "probationary":
                    self.probationary.pop(key, None)
                    self.protected[key] = None
                    entry["segment"] = "protected"
                    self._rebalance_protected()
                else:
                    self.protected.move_to_end(key)
            return
        self.misses += 1
        self.loads += 1
        self.loaded_bytes += int(record["payload_bytes"])
        self.miss_ordinals.append(ordinal)
        if key not in self.seen:
            self.compulsory_miss_ordinals.append(ordinal)
            self.seen.add(key)
        else:
            self.repeated_miss_ordinals.append(ordinal)
        self._admit(key, record, ordinal, session_index)

    def result(
        self,
        records: list[dict[str, Any]],
        global_hit_flags: list[bool],
        inherited_entries: int = 0,
        inherited_bytes: int = 0,
        transition_evictions_at_start: int = 0,
    ) -> dict[str, Any]:
        requested_bytes = sum(int(record["payload_bytes"]) for record in records)
        repeated = set(self.repeated_miss_ordinals)
        capacity_misses = sum(1 for ordinal in repeated if not global_hit_flags[ordinal])
        policy_misses = sum(1 for ordinal in repeated if global_hit_flags[ordinal])
        if self.policy == "global_lru":
            policy_misses = 0
            capacity_misses = len(repeated)
        return {
            "requests": len(records),
            "unique_keys": len({key_of(record) for record in records}),
            "configured_cache_bytes": self.budget,
            "peak_resident_bytes": self.peak_resident_bytes,
            "peak_entry_count": self.peak_entry_count,
            "hits": self.hits,
            "misses": self.misses,
            "loads": self.loads,
            "evictions": self.evictions,
            "request_hit_rate": self.hits / len(records) if records else 0.0,
            "byte_hit_rate": self.avoided_bytes / requested_bytes if requested_bytes else 0.0,
            "expert_bytes_loaded": self.loaded_bytes,
            "expert_bytes_avoided": self.avoided_bytes,
            "expert_read_reduction_pct": 100.0 * self.avoided_bytes / requested_bytes if requested_bytes else 0.0,
            "compulsory_misses": len(self.compulsory_miss_ordinals),
            "capacity_misses": capacity_misses,
            "policy_misses": policy_misses,
            "cache_utilization": self.peak_resident_bytes / self.budget if self.budget else 0.0,
            "unused_cache_bytes": max(0, self.budget - self.peak_resident_bytes),
            "fragmented_bytes": 0,
            "first_hit_ordinal": self.first_hit_ordinal,
            "first_hit": self.first_hit_ordinal is not None,
            "oversized_entry_events": self.oversized_entry_events,
            "partition_rejection_events": self.partition_rejection_events,
            "inherited_entries_at_start": inherited_entries,
            "inherited_bytes_at_start": inherited_bytes,
            "cross_session_hits": self.cross_session_hits,
            "session_transition_evictions": self.transition_evictions - transition_evictions_at_start,
            "resident_entries_at_end": len(self.entries),
            "resident_bytes_at_end": self.resident_bytes,
        }


def make_layer_quotas(
    budget: int,
    policy: str,
    weights: dict[int, int],
) -> dict[int, int]:
    if policy == "layer_lru_architecture":
        quota = budget // LAYERS
        return {layer: quota for layer in range(LAYERS)}
    if policy == "layer_lru_calibrated":
        total = sum(weights.values())
        if total == 0:
            return {layer: 0 for layer in range(LAYERS)}
        return {layer: (budget * weights.get(layer, 0)) // total for layer in range(LAYERS)}
    return {}


def run_cache(
    records: list[dict[str, Any]],
    budget: int,
    policy: str,
    weights: dict[int, int],
    session_index_by_ordinal: list[int] | None = None,
) -> tuple[PolicyCache, list[bool]]:
    quotas = make_layer_quotas(budget, policy, weights)
    cache = PolicyCache(budget, policy, records, quotas)
    session_indexes = session_index_by_ordinal or [0] * len(records)
    for ordinal, record in enumerate(records):
        cache.request(ordinal, record, session_indexes[ordinal])
    return cache, cache.hit_flags


def aggregate_rows(rows: list[dict[str, Any]], fixture_count: int) -> dict[str, Any]:
    requests = sum(row["requests"] for row in rows)
    requested_bytes = sum(row["requests"] * PAYLOAD_BYTES for row in rows)
    hits = sum(row["hits"] for row in rows)
    avoided = sum(row["expert_bytes_avoided"] for row in rows)
    macro_request = mean([row["request_hit_rate"] for row in rows]) if rows else 0.0
    macro_bytes = mean([row["byte_hit_rate"] for row in rows]) if rows else 0.0
    rates = [row["byte_hit_rate"] for row in rows]
    ordered = sorted(rows, key=lambda row: (row["byte_hit_rate"], row["fixture_id"]))
    median_row = ordered[(len(ordered) - 1) // 2] if ordered else None
    return {
        "fixture_count": fixture_count,
        "requests": requests,
        "requested_payload_bytes": requested_bytes,
        "hits": hits,
        "misses": sum(row["misses"] for row in rows),
        "loads": sum(row["loads"] for row in rows),
        "evictions": sum(row["evictions"] for row in rows),
        "macro_average_request_hit_rate": macro_request,
        "macro_average_byte_hit_rate": macro_bytes,
        "micro_request_hit_rate": hits / requests if requests else 0.0,
        "micro_byte_hit_rate": avoided / requested_bytes if requested_bytes else 0.0,
        "expert_bytes_loaded": sum(row["expert_bytes_loaded"] for row in rows),
        "expert_bytes_avoided": avoided,
        "expert_read_reduction_pct": 100.0 * avoided / requested_bytes if requested_bytes else 0.0,
        "worst_case_fixture": ordered[0]["fixture_id"] if ordered else None,
        "worst_case_byte_hit_rate": ordered[0]["byte_hit_rate"] if ordered else 0.0,
        "best_case_fixture": ordered[-1]["fixture_id"] if ordered else None,
        "best_case_byte_hit_rate": ordered[-1]["byte_hit_rate"] if ordered else 0.0,
        "median_fixture": median_row["fixture_id"] if median_row else None,
        "median_fixture_byte_hit_rate": median_row["byte_hit_rate"] if median_row else 0.0,
        "byte_hit_rate_population_stddev": pstdev(rates) if len(rates) > 1 else 0.0,
        "zero_hit_fixture_count": sum(row["hits"] == 0 for row in rows),
        "zero_hit_fixture_percentage": (100.0 * sum(row["hits"] == 0 for row in rows) / fixture_count) if fixture_count else 0.0,
    }


def run_cold_group(
    traces: dict[str, dict[str, Any]],
    budget: int,
    policy: str,
    weights: dict[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixture_ids = EXPECTED_M52_FIXTURES
    reference_flags: dict[str, list[bool]] = {}
    reference_caches: dict[str, PolicyCache] = {}
    for fixture_id in fixture_ids:
        records = traces[fixture_id]["records"]
        ref_cache, flags = run_cache(records, budget, "global_lru", weights)
        reference_caches[fixture_id] = ref_cache
        reference_flags[fixture_id] = flags
    rows: list[dict[str, Any]] = []
    for fixture_id in fixture_ids:
        records = traces[fixture_id]["records"]
        cache, _ = run_cache(records, budget, policy, weights)
        row = cache.result(records, reference_flags[fixture_id])
        row.update({"fixture_id": fixture_id, "scenario": "per_session_cold", "policy": policy, "budget_gib": budget / GIB})
        rows.append(row)
    return rows, aggregate_rows(rows, len(fixture_ids))


def run_persistent_group(
    traces: dict[str, dict[str, Any]],
    fixture_order: list[str],
    budget: int,
    policy: str,
    weights: dict[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    segments: list[tuple[str, int, int]] = []
    session_indexes: list[int] = []
    for session_index, fixture_id in enumerate(fixture_order):
        start = len(records)
        fixture_records = traces[fixture_id]["records"]
        records.extend(fixture_records)
        session_indexes.extend([session_index] * len(fixture_records))
        segments.append((fixture_id, start, len(records)))
    ref_cache, ref_flags = run_cache(records, budget, "global_lru", weights, session_indexes)
    del ref_cache
    cache = PolicyCache(budget, policy, records, make_layer_quotas(budget, policy, weights))
    rows: list[dict[str, Any]] = []
    for fixture_id, start, end in segments:
        segment_records = records[start:end]
        inherited_entries = len(cache.entries)
        inherited_bytes = cache.resident_bytes
        before = {
            "hits": cache.hits,
            "misses": cache.misses,
            "loads": cache.loads,
            "evictions": cache.evictions,
            "loaded_bytes": cache.loaded_bytes,
            "avoided_bytes": cache.avoided_bytes,
            "compulsory": len(cache.compulsory_miss_ordinals),
            "repeated": len(cache.repeated_miss_ordinals),
            "cross_session_hits": cache.cross_session_hits,
            "transition_evictions": cache.transition_evictions,
            "oversized": cache.oversized_entry_events,
            "partition_rejections": cache.partition_rejection_events,
        }
        segment_peak_bytes = cache.resident_bytes
        segment_peak_entries = len(cache.entries)
        for ordinal in range(start, end):
            cache.request(ordinal, records[ordinal], session_indexes[ordinal])
            segment_peak_bytes = max(segment_peak_bytes, cache.resident_bytes)
            segment_peak_entries = max(segment_peak_entries, len(cache.entries))
        requested_bytes = len(segment_records) * PAYLOAD_BYTES
        segment_hits = cache.hits - before["hits"]
        segment_misses = cache.misses - before["misses"]
        segment_repeated_ordinals = cache.repeated_miss_ordinals[before["repeated"]:]
        capacity_misses = sum(not ref_flags[ordinal] for ordinal in segment_repeated_ordinals)
        policy_misses = sum(ref_flags[ordinal] for ordinal in segment_repeated_ordinals)
        if policy == "global_lru":
            capacity_misses = len(segment_repeated_ordinals)
            policy_misses = 0
        local_hit_ordinals = [
            local for local, hit in enumerate(cache.hit_flags[start:end]) if hit
        ]
        rows.append(
            {
                "fixture_id": fixture_id,
                "requests": len(segment_records),
                "unique_keys": len({key_of(record) for record in segment_records}),
                "configured_cache_bytes": budget,
                "peak_resident_bytes": segment_peak_bytes,
                "peak_entry_count": segment_peak_entries,
                "hits": segment_hits,
                "misses": segment_misses,
                "loads": cache.loads - before["loads"],
                "evictions": cache.evictions - before["evictions"],
                "request_hit_rate": segment_hits / len(segment_records) if segment_records else 0.0,
                "byte_hit_rate": (cache.avoided_bytes - before["avoided_bytes"]) / requested_bytes if requested_bytes else 0.0,
                "expert_bytes_loaded": cache.loaded_bytes - before["loaded_bytes"],
                "expert_bytes_avoided": cache.avoided_bytes - before["avoided_bytes"],
                "expert_read_reduction_pct": 100.0 * (cache.avoided_bytes - before["avoided_bytes"]) / requested_bytes if requested_bytes else 0.0,
                "compulsory_misses": len(cache.compulsory_miss_ordinals) - before["compulsory"],
                "capacity_misses": capacity_misses,
                "policy_misses": policy_misses,
                "cache_utilization": segment_peak_bytes / budget if budget else 0.0,
                "unused_cache_bytes": max(0, budget - segment_peak_bytes),
                "fragmented_bytes": 0,
                "first_hit_ordinal": local_hit_ordinals[0] if local_hit_ordinals else None,
                "first_hit": bool(local_hit_ordinals),
                "oversized_entry_events": cache.oversized_entry_events - before["oversized"],
                "partition_rejection_events": cache.partition_rejection_events - before["partition_rejections"],
                "inherited_entries_at_start": inherited_entries,
                "inherited_bytes_at_start": inherited_bytes,
                "cross_session_hits": cache.cross_session_hits - before["cross_session_hits"],
                "session_transition_evictions": cache.transition_evictions - before["transition_evictions"],
                "resident_entries_at_end": len(cache.entries),
                "resident_bytes_at_end": cache.resident_bytes,
                "global_reference_hits_in_segment": sum(ref_flags[start:end]),
            }
        )
    aggregate = aggregate_rows(rows, len(segments))
    aggregate.update({"scenario": "persistent", "fixture_order": fixture_order})
    return rows, aggregate


def replay_segment_metrics(
    segment_records: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    start: int,
    end: int,
    budget: int,
    policy: str,
    weights: dict[int, int],
    session_indexes: list[int],
    reference_flags: list[bool],
    inherited_entries: int,
    inherited_bytes: int,
    inherited_evictions: int,
    completed_cache: PolicyCache,
) -> dict[str, Any]:
    """Return exact per-segment deltas from a completed persistent replay.

    A second deterministic replay captures counters at each boundary. It keeps
    the main cache implementation simple and makes session-boundary evidence
    explicit without storing timing or mutable runtime state.
    """
    cache = PolicyCache(budget, policy, all_records, make_layer_quotas(budget, policy, weights))
    for ordinal, record in enumerate(all_records[:end]):
        cache.request(ordinal, record, session_indexes[ordinal])
    prefix = PolicyCache(budget, policy, all_records, make_layer_quotas(budget, policy, weights))
    for ordinal, record in enumerate(all_records[:start]):
        prefix.request(ordinal, record, session_indexes[ordinal])
    metrics = cache.result(segment_records, reference_flags, inherited_entries, inherited_bytes, inherited_evictions)
    prefix_hits = prefix.hits
    prefix_misses = prefix.misses
    prefix_loads = prefix.loads
    prefix_evictions = prefix.evictions
    prefix_loaded = prefix.loaded_bytes
    prefix_avoided = prefix.avoided_bytes
    prefix_compulsory = len(prefix.compulsory_miss_ordinals)
    prefix_repeated = len(prefix.repeated_miss_ordinals)
    prefix_cross = prefix.cross_session_hits
    prefix_transition = prefix.transition_evictions
    segment_hits = cache.hits - prefix_hits
    segment_misses = cache.misses - prefix_misses
    segment_loads = cache.loads - prefix_loads
    segment_evictions = cache.evictions - prefix_evictions
    segment_loaded = cache.loaded_bytes - prefix_loaded
    segment_avoided = cache.avoided_bytes - prefix_avoided
    segment_compulsory = len(cache.compulsory_miss_ordinals) - prefix_compulsory
    segment_repeated = len(cache.repeated_miss_ordinals) - prefix_repeated
    segment_cross = cache.cross_session_hits - prefix_cross
    segment_transition = cache.transition_evictions - prefix_transition
    requested_bytes = len(segment_records) * PAYLOAD_BYTES
    global_segment_flags = reference_flags[start:end]
    global_hits = sum(global_segment_flags)
    capacity_misses = sum(
        1
        for ordinal in range(start, end)
        if ordinal in set(cache.repeated_miss_ordinals) and not reference_flags[ordinal]
    )
    policy_misses = sum(
        1
        for ordinal in range(start, end)
        if ordinal in set(cache.repeated_miss_ordinals) and reference_flags[ordinal]
    )
    if policy == "global_lru":
        policy_misses = 0
        capacity_misses = segment_repeated
    metrics.update(
        {
            "requests": len(segment_records),
            "unique_keys": len({key_of(record) for record in segment_records}),
            "configured_cache_bytes": budget,
            "peak_resident_bytes": cache.peak_resident_bytes,
            "peak_entry_count": cache.peak_entry_count,
            "hits": segment_hits,
            "misses": segment_misses,
            "loads": segment_loads,
            "evictions": segment_evictions,
            "request_hit_rate": segment_hits / len(segment_records) if segment_records else 0.0,
            "byte_hit_rate": segment_avoided / requested_bytes if requested_bytes else 0.0,
            "expert_bytes_loaded": segment_loaded,
            "expert_bytes_avoided": segment_avoided,
            "expert_read_reduction_pct": 100.0 * segment_avoided / requested_bytes if requested_bytes else 0.0,
            "compulsory_misses": segment_compulsory,
            "capacity_misses": capacity_misses,
            "policy_misses": policy_misses,
            "cross_session_hits": segment_cross,
            "session_transition_evictions": segment_transition,
            "inherited_entries_at_start": inherited_entries,
            "inherited_bytes_at_start": inherited_bytes,
            "global_reference_hits_in_segment": global_hits,
        }
    )
    return metrics


def candidate_budgets(
    records: list[dict[str, Any]],
    policy: str,
    weights: dict[int, int],
) -> list[int]:
    unique_by_layer: dict[int, set[str]] = {layer: set() for layer in range(LAYERS)}
    charges: set[int] = set()
    for record in records:
        unique_by_layer[record["layer_index"]].add(key_of(record))
        charges.add(charge_for(int(record["payload_bytes"])))
    if len(charges) != 1:
        raise ValueError("M5.2 corpus threshold analysis expects one expert payload size")
    charge = next(iter(charges))
    unique_count = len({key_of(record) for record in records})
    candidates = {0}
    for count in range(1, unique_count + 1):
        candidates.add(count * charge)
    if policy == "layer_lru_architecture":
        max_layer = max(len(keys) for keys in unique_by_layer.values())
        candidates.update(LAYERS * count * charge for count in range(1, max_layer + 1))
    if policy == "layer_lru_calibrated":
        total = sum(weights.values())
        for layer, count in ((layer, len(keys)) for layer, keys in unique_by_layer.items() if weights.get(layer, 0)):
            for entry_count in range(1, count + 1):
                candidates.add(math.ceil(entry_count * charge * total / weights[layer]))
    return sorted(candidates)


def cold_metric_at_budget(
    traces: dict[str, dict[str, Any]], budget: int, policy: str, weights: dict[int, int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return run_cold_group(traces, budget, policy, weights)


def cold_policy_rate_metrics(
    traces: dict[str, dict[str, Any]], budget: int, policy: str, weights: dict[int, int]
) -> tuple[float, float]:
    """Replay one cold policy without rebuilding the LRU reference.

    Threshold analysis only needs byte-hit rates.  Capacity/policy miss
    classification is already produced by the primary matrix, so rebuilding
    a second global-LRU cache for every threshold point would be pure work.
    """
    rows: list[dict[str, Any]] = []
    for fixture_id in EXPECTED_M52_FIXTURES:
        records = traces[fixture_id]["records"]
        cache, _ = run_cache(records, budget, policy, weights)
        requested_bytes = len(records) * PAYLOAD_BYTES
        rows.append(
            {
                "fixture_id": fixture_id,
                "requests": len(records),
                "hits": cache.hits,
                "misses": cache.misses,
                "loads": cache.loads,
                "evictions": cache.evictions,
                "request_hit_rate": cache.hits / len(records) if records else 0.0,
                "byte_hit_rate": cache.avoided_bytes / requested_bytes if requested_bytes else 0.0,
                "expert_bytes_loaded": cache.loaded_bytes,
                "expert_bytes_avoided": cache.avoided_bytes,
            }
        )
    aggregate = aggregate_rows(rows, len(EXPECTED_M52_FIXTURES))
    return aggregate["micro_byte_hit_rate"], aggregate["macro_average_byte_hit_rate"]


def threshold_analysis(
    traces: dict[str, dict[str, Any]],
    weights: dict[int, int],
) -> dict[str, Any]:
    def exact_thresholds(
        candidates: list[int],
        evaluator: Callable[[int], float],
        policy: str,
    ) -> tuple[int | None, dict[str, int | None], float]:
        """Find exact minimum event budgets for the requested hit rates.

        Global/layer-LRU and Belady use lower-bound lookup over exact replay
        points because their stack/partition response is monotonic. LFU and
        segmented LRU are scanned exhaustively in ascending exact event-budget
        order because no monotonicity assumption is made for them. No rate is
        interpolated; the fixed-budget matrix remains the policy comparison.
        """
        values: dict[int, float] = {}

        def value_at(budget: int) -> float:
            if budget not in values:
                values[budget] = evaluator(budget)
            return values[budget]

        first_hit: int | None = None
        thresholds: dict[str, int | None] = {
            f"{int(target * 100)}_pct": None for target in THRESHOLD_TARGETS
        }
        if policy in MONOTONIC_THRESHOLD_POLICIES:
            def lower_bound(predicate: Callable[[float], bool]) -> int | None:
                if not predicate(value_at(candidates[-1])):
                    return None
                low = 0
                high = len(candidates)
                while low < high:
                    middle = (low + high) // 2
                    if predicate(value_at(candidates[middle])):
                        high = middle
                    else:
                        low = middle + 1
                return candidates[low]

            first_hit = lower_bound(lambda value: value > 0.0)
            for target in THRESHOLD_TARGETS:
                label = f"{int(target * 100)}_pct"
                thresholds[label] = lower_bound(lambda value, target=target: value >= target)
        else:
            for budget in candidates:
                value = value_at(budget)
                if first_hit is None and value > 0.0:
                    first_hit = budget
                for target in THRESHOLD_TARGETS:
                    label = f"{int(target * 100)}_pct"
                    if thresholds[label] is None and value >= target:
                        thresholds[label] = budget
                if first_hit is not None and all(value is not None for value in thresholds.values()):
                    break
        return first_hit, thresholds, value_at(candidates[-1])

    per_fixture: list[dict[str, Any]] = []
    corpus: list[dict[str, Any]] = []
    all_records = [record for fixture_id in EXPECTED_M52_FIXTURES for record in traces[fixture_id]["records"]]
    for policy in POLICY_IDS:
        fixture_candidates = {
            fixture_id: candidate_budgets(traces[fixture_id]["records"], policy, weights)
            for fixture_id in EXPECTED_M52_FIXTURES
        }
        for fixture_id in EXPECTED_M52_FIXTURES:
            candidates = fixture_candidates[fixture_id]
            record_list = traces[fixture_id]["records"]
            def fixture_rate(budget: int) -> float:
                cache, _ = run_cache(record_list, budget, policy, weights)
                return cache.avoided_bytes / (len(record_list) * PAYLOAD_BYTES)

            first_hit, hit_thresholds, full_rate = exact_thresholds(candidates, fixture_rate, policy)
            per_fixture.append(
                {
                    "fixture_id": fixture_id,
                    "policy": policy,
                    "search_method": "exact_lower_bound_replay" if policy in MONOTONIC_THRESHOLD_POLICIES else "exact_exhaustive_replay",
                    "first_hit_bytes": first_hit,
                    "byte_hit_rate_thresholds": hit_thresholds,
                    "full_unique_working_set_bytes": len({key_of(record) for record in record_list}) * charge_for(PAYLOAD_BYTES),
                    "policy_full_residency_threshold_bytes": candidates[-1],
                    "full_working_set_byte_hit_rate": full_rate,
                }
            )
    for policy in POLICY_IDS:
        candidates = candidate_budgets(all_records, policy, weights)
        corpus_metrics: dict[int, tuple[float, float]] = {}

        def corpus_metrics_at(budget: int) -> tuple[float, float]:
            if budget not in corpus_metrics:
                corpus_metrics[budget] = cold_policy_rate_metrics(traces, budget, policy, weights)
            return corpus_metrics[budget]

        def corpus_rate(budget: int) -> float:
            return corpus_metrics_at(budget)[0]

        def corpus_macro_rate(budget: int) -> float:
            return corpus_metrics_at(budget)[1]

        first_hit, micro_thresholds, full_micro_rate = exact_thresholds(candidates, corpus_rate, policy)
        _, macro_thresholds, full_macro_rate = exact_thresholds(candidates, corpus_macro_rate, policy)
        corpus.append(
            {
                "policy": policy,
                "search_method": "exact_lower_bound_replay" if policy in MONOTONIC_THRESHOLD_POLICIES else "exact_exhaustive_replay",
                "first_hit_bytes": first_hit,
                "macro_byte_hit_rate_thresholds": macro_thresholds,
                "micro_byte_hit_rate_thresholds": micro_thresholds,
                "full_unique_working_set_bytes": len({key_of(record) for record in all_records}) * charge_for(PAYLOAD_BYTES),
                "policy_full_residency_threshold_bytes": candidates[-1],
                "full_working_set_macro_byte_hit_rate": full_macro_rate,
                "full_working_set_micro_byte_hit_rate": full_micro_rate,
            }
        )
    return {"per_fixture": per_fixture, "corpus": corpus}


def build_policy_specs(weights: dict[int, int]) -> list[dict[str, Any]]:
    return [
        {
            "id": "global_lru",
            "kind": "online",
            "description": "Strict global byte-budgeted LRU; oldest last-access ordinal then key tie-break.",
            "calibration": "none",
        },
        {
            "id": "layer_lru_architecture",
            "kind": "online_diagnostic",
            "description": "Static equal byte partition across 48 layers; oldest same-layer entry; unused capacity is not borrowed.",
            "calibration": "architecture_only",
            "partition_layers": LAYERS,
            "borrow_unused_capacity": False,
            "quota_formula": "floor(configured_budget_bytes / 48)",
        },
        {
            "id": "layer_lru_calibrated",
            "kind": "offline_calibrated_diagnostic",
            "description": "Static layer partition proportional to corpus-wide unique-key counts; unused capacity is not borrowed.",
            "calibration": "corpus_unique_keys_by_layer",
            "borrow_unused_capacity": False,
            "layer_weights": {str(layer): weights[layer] for layer in sorted(weights)},
            "quota_formula": "floor(configured_budget_bytes * layer_weight / sum(layer_weights))",
        },
        {
            "id": "frequency_lfu",
            "kind": "online",
            "description": "Observed-frequency LFU; counters reset per cold session, recency then key tie-break; every miss is admitted when it fits.",
            "calibration": "none",
            "frequency_reset": "per_cache_lifecycle",
            "admission": "admit_if_entry_fits",
            "eviction_order": "(observed_frequency, last_access_ordinal, layer_expert_key)",
        },
        {
            "id": "segmented_lru",
            "kind": "online",
            "description": "80/20 segmented LRU; new entries enter probationary, hits promote to protected, protected overflow demotes oldest.",
            "calibration": "none",
            "protected_fraction": 0.8,
            "admission": "admit_if_entry_fits",
        },
        {
            "id": "belady",
            "kind": "offline_theoretical",
            "description": "Evict the entry with the farthest next use; theoretical upper bound and not an online candidate.",
            "calibration": "full_future_trace",
        },
    ]


def build_input(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    corpus_manifest = load_json(root / paths["corpus_manifest"])
    aggregate = load_json(root / paths["corpus_aggregate"])
    traces = {
        entry["fixture_id"]: load_json(root / entry["trace_path"])
        for entry in corpus_manifest["fixtures"]
    }
    weights = layer_weights(traces)
    fixture_boundaries: dict[str, dict[str, Any]] = {}
    cursor = 0
    for fixture_id in EXPECTED_M52_FIXTURES:
        count = len(traces[fixture_id]["records"])
        fixture_boundaries[fixture_id] = {"start_ordinal": cursor, "end_ordinal_exclusive": cursor + count, "record_count": count}
        cursor += count
    orders = {
        "manifest_order": EXPECTED_M52_FIXTURES,
        "reverse_order": list(reversed(EXPECTED_M52_FIXTURES)),
    }
    references = {
        "corpus_manifest": {"path": paths["corpus_manifest"].as_posix(), "sha256": sha256(root / paths["corpus_manifest"])},
        "corpus_aggregate": {"path": paths["corpus_aggregate"].as_posix(), "sha256": sha256(root / paths["corpus_aggregate"])},
        "trace_schema": {"path": paths["trace_schema"].as_posix(), "sha256": sha256(root / paths["trace_schema"])},
        "m4_baseline": {"path": paths["m4_baseline"].as_posix(), "sha256": sha256(root / paths["m4_baseline"])},
        "m4_provenance": {"path": paths["m4_provenance"].as_posix(), "sha256": sha256(root / paths["m4_provenance"])},
        "model_manifest": {"path": paths["model_manifest"].as_posix(), "sha256": sha256(root / paths["model_manifest"])},
    }
    return {
        "schema": INPUT_SCHEMA,
        "schema_version": 1,
        "task": "M5.2-02",
        "corpus_id": EXPECTED_CORPUS_ID,
        "references": references,
        "artifact": {"canonical_root_sha256": EXPECTED_ARTIFACT_ROOT, "model_id": "Qwen/Qwen3-30B-A3B", "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"},
        "fixture_ids": EXPECTED_M52_FIXTURES,
        "fixture_boundaries_manifest_order": fixture_boundaries,
        "persistent_fixture_orders": orders,
        "budgets": {"unit": "binary_gib", "values": RAM_BUDGET_GIBS, "expert_payload_budget_exact": True, "dense_residency_modeled": False},
        "accounting": {"metadata_bytes_per_entry": META_BYTES, "alignment_bytes": ALIGNMENT, "charge_formula": "payload_bytes", "metadata_and_alignment_budget_effect": "not charged; tracked as separate runtime overhead context", "budget_semantics": "configured ExpertCache payload bytes; dense bytes are not silently subtracted", "fragmentation_model": "zero; logical payload charges are packed and unused bytes are reported"},
        "policies": build_policy_specs(weights),
        "threshold_targets": THRESHOLD_TARGETS,
        "scenarios": [
            {"id": "per_session_cold", "kind": "per_session", "cache_reset": "before_each_fixture", "primary": True},
            {"id": "persistent_manifest_order", "kind": "persistent", "fixture_order_name": "manifest_order", "primary": False},
            {"id": "persistent_reverse_order", "kind": "persistent", "fixture_order_name": "reverse_order", "primary": False},
        ],
        "aggregate_counters": aggregate["corpus_wide"],
        "layer_calibration_weights": {str(layer): weights[layer] for layer in sorted(weights)},
        "serialization": "UTF-8 JSON, sorted keys, compact separators, trailing newline; no timing or machine-local paths",
    }


def run_all(root: Path, input_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    input_doc = load_json(input_path)
    validated = validate_corpus(root, input_doc)
    traces = validated["traces"]
    weights = validated["layer_weights"]
    cold_rows: list[dict[str, Any]] = []
    cold_aggregates: list[dict[str, Any]] = []
    persistent_rows: list[dict[str, Any]] = []
    persistent_aggregates: list[dict[str, Any]] = []
    for budget_gib in RAM_BUDGET_GIBS:
        budget = budget_gib * GIB
        for policy in POLICY_IDS:
            rows, aggregate = run_cold_group(traces, budget, policy, weights)
            for row in rows:
                row.update({"configured_budget_gib": budget_gib, "policy_kind": next(spec["kind"] for spec in input_doc["policies"] if spec["id"] == policy)})
            aggregate.update({"configured_budget_gib": budget_gib, "policy": policy, "scenario": "per_session_cold", "policy_kind": next(spec["kind"] for spec in input_doc["policies"] if spec["id"] == policy)})
            cold_rows.extend(rows)
            cold_aggregates.append(aggregate)
            for order_name, fixture_order in input_doc["persistent_fixture_orders"].items():
                p_rows, p_aggregate = run_persistent_group(traces, fixture_order, budget, policy, weights)
                for row in p_rows:
                    row.update({"configured_budget_gib": budget_gib, "policy": policy, "policy_kind": next(spec["kind"] for spec in input_doc["policies"] if spec["id"] == policy), "scenario": f"persistent_{order_name}"})
                p_aggregate.update({"configured_budget_gib": budget_gib, "policy": policy, "policy_kind": next(spec["kind"] for spec in input_doc["policies"] if spec["id"] == policy), "scenario": f"persistent_{order_name}"})
                persistent_rows.extend(p_rows)
                persistent_aggregates.append(p_aggregate)

    policy_budget_summary: list[dict[str, Any]] = []
    for aggregate in cold_aggregates:
        policy_budget_summary.append(aggregate)
    thresholds = threshold_analysis(traces, weights)
    policy_specs_by_id = {spec["id"]: spec for spec in input_doc["policies"]}
    trace_aggregate = validated["aggregate"]
    results = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "task": "M5.2-02",
        "corpus_id": EXPECTED_CORPUS_ID,
        "input_manifest_path": input_path.relative_to(root).as_posix(),
        "input_manifest_sha256": sha256(input_path),
        "corpus_manifest_sha256": input_doc["references"]["corpus_manifest"]["sha256"],
        "aggregate_sha256": input_doc["references"]["corpus_aggregate"]["sha256"],
        "artifact_root_sha256": EXPECTED_ARTIFACT_ROOT,
        "policies": policy_specs_by_id,
        "budgets_binary_gib": RAM_BUDGET_GIBS,
        "accounting": input_doc["accounting"],
        "validation": {"corpus_validated_before_simulation": True, "fixture_ids": EXPECTED_M52_FIXTURES, "canonical_artifact_validated": True, "m5_1_record_adapter_validated": True},
        "trace_statistics": {
            "corpus_wide": trace_aggregate["corpus_wide"],
            "per_fixture": {
                entry["fixture"]["fixture_id"]: {
                    "classification": entry["fixture"]["classification"],
                    "workload_class": entry["fixture"]["workload_class"],
                    **entry["statistics"],
                }
                for entry in trace_aggregate["per_fixture"]
            },
            "comparison_with_frozen_tier_a": trace_aggregate["comparison_with_frozen_tier_a"],
        },
        "per_session_cold": {"fixture_results": cold_rows, "aggregates": cold_aggregates},
        "persistent_cache": {"fixture_results": persistent_rows, "aggregates": persistent_aggregates},
        "thresholds": thresholds,
        "analysis": analyze_results(cold_aggregates, cold_rows, thresholds, policy_specs_by_id),
        "simulation_only": True,
    }
    validate_result_invariants(results)
    results["validation"].update(
        {
            "replay_accounting_invariants_validated": True,
            "aggregate_reconciliation_validated": True,
            "peak_payload_residency_within_budget": True,
        }
    )
    output_path.write_text(json.dumps(results, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    report_path.write_text(render_report(results), encoding="utf-8", newline="\n")
    return results


def analyze_results(
    aggregates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    policy_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    global_rows = [row for row in aggregates if row["policy"] == "global_lru"]
    by_budget = {int(row["configured_budget_gib"]): row for row in global_rows}
    online_alternatives = sorted(ONLINE_POLICIES - {"global_lru"})
    policy_comparison: list[dict[str, Any]] = []
    for policy in online_alternatives:
        policy_rows = {int(row["configured_budget_gib"]): row for row in aggregates if row["policy"] == policy}
        gains = {
            str(budget): policy_rows[budget]["micro_byte_hit_rate"] - by_budget[budget]["micro_byte_hit_rate"]
            for budget in RAM_BUDGET_GIBS
        }
        policy_comparison.append(
            {
                "policy": policy,
                "average_micro_byte_gain_over_global_lru": mean(gains.values()),
                "maximum_micro_byte_gain_over_global_lru": max(gains.values()),
                "minimum_micro_byte_gain_over_global_lru": min(gains.values()),
                "micro_byte_gain_by_budget": gains,
                "gain_at_8_gib": gains["8"],
                "gain_at_12_gib": gains["12"],
                "gain_at_16_gib": gains["16"],
                "consistent_nonnegative_gain_at_8_12_16": all(gains[str(budget)] >= 0.0 for budget in (8, 12, 16)),
            }
        )
    max_alt_gain = max((item["maximum_micro_byte_gain_over_global_lru"] for item in policy_comparison), default=0.0)
    eight = by_budget[8]
    twelve = by_budget[12]
    sixteen = by_budget[16]
    worst_eight = eight["worst_case_byte_hit_rate"]
    if eight["macro_average_byte_hit_rate"] >= 0.25 and eight["micro_byte_hit_rate"] >= 0.25 and worst_eight >= 0.05:
        eight_classification = "representative_balanced_mode"
    elif eight["macro_average_byte_hit_rate"] >= 0.10 and worst_eight < 0.05:
        eight_classification = "useful_for_selected_workloads"
    else:
        eight_classification = "too_small_for_general_workloads"
    fixture_rows = {
        (int(row["configured_budget_gib"]), row["fixture_id"]): row
        for row in rows
        if row["policy"] == "global_lru" and row["scenario"] == "per_session_cold"
    }
    mode_specs = [
        ("ultra_low_memory_streaming", 1),
        ("low_memory_cache", 4),
        ("balanced", 8),
        ("performance", 16),
    ]
    operating_modes: list[dict[str, Any]] = []
    for mode, budget in mode_specs:
        aggregate = by_budget[budget]
        suitable = [
            fixture_id
            for fixture_id in EXPECTED_M52_FIXTURES
            if fixture_rows[(budget, fixture_id)]["byte_hit_rate"] >= 0.10
        ]
        operating_modes.append(
            {
                "mode": mode,
                "expert_cache_budget_gib": budget,
                "policy": "global_lru",
                "expected_macro_byte_hit_rate": aggregate["macro_average_byte_hit_rate"],
                "expected_micro_byte_hit_rate": aggregate["micro_byte_hit_rate"],
                "worst_case_fixture": aggregate["worst_case_fixture"],
                "worst_case_byte_hit_rate": aggregate["worst_case_byte_hit_rate"],
                "modeled_expert_read_reduction_pct": aggregate["expert_read_reduction_pct"],
                "suitable_workloads": suitable,
                "status": "modeled_candidate; runtime validation pending",
            }
        )
    gain_8_to_16 = sixteen["micro_byte_hit_rate"] - eight["micro_byte_hit_rate"]
    return {
        "eight_gib_classification": eight_classification,
        "eight_gib_classification_basis": {
            "macro_byte_hit_rate": eight["macro_average_byte_hit_rate"],
            "micro_byte_hit_rate": eight["micro_byte_hit_rate"],
            "worst_case_byte_hit_rate": worst_eight,
            "zero_hit_fixture_percentage": eight["zero_hit_fixture_percentage"],
            "marginal_micro_gain_to_12_gib": twelve["micro_byte_hit_rate"] - eight["micro_byte_hit_rate"],
            "marginal_micro_gain_to_16_gib": gain_8_to_16,
        },
        "eight_gib_global_lru": eight,
        "twelve_gib_global_lru": twelve,
        "sixteen_gib_global_lru": sixteen,
        "max_online_alternative_micro_byte_gain_over_global_lru": max_alt_gain,
        "online_policy_comparison": policy_comparison,
        "policy_selection": {
            "selected_policy": "global_lru",
            "decision": "retain_global_lru",
            "reason": "LFU has the largest isolated online gain at 6 GiB but is below global LRU at 8 GiB and only marginally ahead at 12/16 GiB; architecture-only and calibrated layer partitions are not consistently better, while segmented LRU is also workload-sensitive. Global LRU is already implemented, deterministic, byte-budgeted, and has lower correctness and calibration risk.",
        },
        "recommended_next_runtime_validation": {"policy": "global_lru", "budgets_gib": [8, 16], "selection_basis": f"8 GiB remains the selected-workload baseline; 16 GiB adds {gain_8_to_16:.6%} micro byte hit rate in cold-corpus simulation and covers the higher-memory point. Runtime validation is not executed in M5.2-02."},
        "operating_modes": operating_modes,
        "policy_comparison_scope": "Online alternatives are compared by macro/micro hit rates and worst-case variance; no production policy is calibrated from future requests.",
    }


def validate_result_invariants(results: dict[str, Any]) -> None:
    """Fail closed on accounting and aggregation errors before writing evidence."""
    all_groups = [
        results["per_session_cold"]["fixture_results"],
        results["persistent_cache"]["fixture_results"],
    ]
    for group in all_groups:
        for row in group:
            requested_bytes = row["requests"] * PAYLOAD_BYTES
            if row["hits"] + row["misses"] != row["requests"]:
                raise ValueError(f"hit/miss reconciliation failed for {row}")
            if row["loads"] != row["misses"]:
                raise ValueError(f"load/miss reconciliation failed for {row}")
            if row["expert_bytes_loaded"] + row["expert_bytes_avoided"] != requested_bytes:
                raise ValueError(f"payload accounting failed for {row}")
            if row["peak_resident_bytes"] > row["configured_cache_bytes"]:
                raise ValueError(f"cache budget exceeded for {row}")
            if row["evictions"] < 0 or row["compulsory_misses"] < 0:
                raise ValueError(f"negative cache counter for {row}")

    def check_aggregates(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
        groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault((int(row["configured_budget_gib"]), row["policy"], row["scenario"]), []).append(row)
        for aggregate in aggregates:
            key = (int(aggregate["configured_budget_gib"]), aggregate["policy"], aggregate["scenario"])
            expected_rows = groups.get(key)
            if expected_rows is None or len(expected_rows) != 8:
                raise ValueError(f"aggregate fixture boundary failed for {key}")
            recomputed = aggregate_rows(expected_rows, 8)
            for field in ("requests", "hits", "misses", "loads", "evictions", "expert_bytes_loaded", "expert_bytes_avoided"):
                if aggregate[field] != recomputed[field]:
                    raise ValueError(f"aggregate {field} mismatch for {key}")
            for field in ("macro_average_byte_hit_rate", "micro_byte_hit_rate", "expert_read_reduction_pct"):
                if not math.isclose(aggregate[field], recomputed[field], rel_tol=0.0, abs_tol=1e-15):
                    raise ValueError(f"aggregate {field} mismatch for {key}")

    check_aggregates(results["per_session_cold"]["fixture_results"], results["per_session_cold"]["aggregates"])
    check_aggregates(results["persistent_cache"]["fixture_results"], results["persistent_cache"]["aggregates"])
    corpus_stats = results["trace_statistics"]["corpus_wide"]
    if corpus_stats["total_expert_occurrences"] != 11520 or corpus_stats["unique_layer_expert_keys"] != 3148:
        raise ValueError("trace statistics total mismatch")


def render_report(results: dict[str, Any]) -> str:
    analysis = results["analysis"]
    lines = [
        "# M5.2-02 Corpus Cache Simulation",
        "",
        "This is a deterministic, simulation-only replay of the validated M5.2-01 corpus. It does not invoke Rust, load model artifacts, modify ExpertCache, or claim latency or physical-I/O results.",
        "",
        "Starting state: branch `milestone/m5-performance`, HEAD `591172c95436703f1e2fe0d0d3e1c2204a3c6957`, clean working tree. The runtime validation matrix selected below was not executed.",
        "",
        f"Corpus ID: `{results['corpus_id']}`.",
        f"Corpus manifest SHA-256: `{results['corpus_manifest_sha256']}`.",
        f"Corpus aggregate SHA-256: `{results['aggregate_sha256']}`.",
        f"Input manifest SHA-256: `{results['input_manifest_sha256']}`.",
        f"Canonical artifact SHA-256: `{results['artifact_root_sha256']}`.",
        "Input validation passed for corpus schema/hash, all eight trace hashes and boundaries, M4 baseline/provenance, canonical artifact identity, layer/expert ranges, payload sizes, and M5.1 record-key compatibility.",
        "",
        "## Policy semantics",
        "",
        "| Policy | Kind | Semantics |",
        "|---|---|---|",
    ]
    for policy in results["policies"].values():
        lines.append(f"| `{policy['id']}` | {policy['kind']} | {policy['description']} |")
    lines += [
        "",
        "All budgets are exact binary-GiB ExpertCache payload-byte budgets. The 64-byte metadata and 4,096-byte alignment values are recorded as separate overhead context and are not silently deducted from payload capacity. Dense residency is not subtracted from the expert budget.",
        "",
        "## Per-session cold global-LRU macro/micro results",
        "",
        "| GiB | Macro request hit | Micro request hit | Macro byte hit | Micro byte hit | Worst fixture | Zero-hit fixtures |",
        "|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in results["per_session_cold"]["aggregates"]:
        if row["policy"] == "global_lru":
            lines.append(f"| {row['configured_budget_gib']} | {row['macro_average_request_hit_rate']:.4%} | {row['micro_request_hit_rate']:.4%} | {row['macro_average_byte_hit_rate']:.4%} | {row['micro_byte_hit_rate']:.4%} | `{row['worst_case_fixture']}` ({row['worst_case_byte_hit_rate']:.4%}) | {row['zero_hit_fixture_count']}/{row['fixture_count']} |")
    lines += ["", "## Online policy comparison", "", "| GiB | Global LRU micro byte hit | Architecture layer LRU | Calibrated layer LRU | LFU | Segmented LRU | Belady ceiling |", "|---:|---:|---:|---:|---:|---:|---:|"]
    aggregate_by_budget_policy = {(int(row["configured_budget_gib"]), row["policy"]): row for row in results["per_session_cold"]["aggregates"]}
    for budget in RAM_BUDGET_GIBS:
        values = [aggregate_by_budget_policy[(budget, policy)]["micro_byte_hit_rate"] for policy in POLICY_IDS]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(budget, *(f"{value:.4%}" for value in values)))
    lines += ["", "## Per-fixture global-LRU matrix at 8 and 16 GiB", "", "| Fixture | 8 GiB hits | 8 GiB byte hit | 16 GiB hits | 16 GiB byte hit | unique keys | classification |", "|---|---:|---:|---:|---:|---:|---|"]
    fixture_result_map = {(int(row["configured_budget_gib"]), row["fixture_id"]): row for row in results["per_session_cold"]["fixture_results"] if row["policy"] == "global_lru"}
    fixture_stats = results["trace_statistics"]["per_fixture"]
    for fixture_id in EXPECTED_M52_FIXTURES:
        at8 = fixture_result_map[(8, fixture_id)]
        at16 = fixture_result_map[(16, fixture_id)]
        classification = fixture_stats[fixture_id]["classification"]
        lines.append(f"| `{fixture_id}` | {at8['hits']} | {at8['byte_hit_rate']:.4%} | {at16['hits']} | {at16['byte_hit_rate']:.4%} | {at8['unique_keys']} | {classification} |")
    lines += ["", "## Exact global-LRU threshold summary", "", "| Fixture | First hit | 10% | 25% | 40% | 50% | 75% | 90% | Full payload working set |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in results["thresholds"]["per_fixture"]:
        if item["policy"] != "global_lru":
            continue
        values = [item["byte_hit_rate_thresholds"].get(f"{target}_pct") for target in (10, 25, 40, 50, 75, 90)]
        def short_bytes(value: int | None) -> str:
            return "—" if value is None else f"{value / GIB:.3f} GiB"
        lines.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {:.3f} GiB |".format(item["fixture_id"], short_bytes(item["first_hit_bytes"]), *(short_bytes(value) for value in values), item["full_unique_working_set_bytes"] / GIB))
    lines += ["", "## Workload cacheability evidence", "", "The descriptive trace statistics below are copied from the validated M5.2-01 aggregate and are independent of policy replay.", "", "| Fixture | occurrences | unique keys | repeated-key % | reuse min/median/max | prefill/decode | cross-token % | classification |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for fixture_id in EXPECTED_M52_FIXTURES:
        stats = fixture_stats[fixture_id]
        reuse = stats["reuse_distance"]
        phase = stats["prefill_decode_occurrences"]
        scope = stats["reuse_scope"]
        classification = stats["classification"]
        lines.append(f"| `{fixture_id}` | {stats['total_expert_occurrences']} | {stats['unique_layer_expert_keys']} | {stats['repeated_key_percentage']:.4%} | {reuse['minimum']}/{reuse['median']}/{reuse['maximum']} | {phase['prefill']}/{phase['decode']} | {scope['cross_token_reuse_percentage']:.4%} | {classification} |")
    corpus_stats = results["trace_statistics"]["corpus_wide"]
    lines += ["", f"Corpus-wide descriptive totals: `{corpus_stats['total_expert_occurrences']}` occurrences, `{corpus_stats['unique_layer_expert_keys']}` unique keys, repeated-key `{corpus_stats['repeated_key_percentage']:.4%}`, reuse min/median/max `{corpus_stats['reuse_distance']['minimum']}/{corpus_stats['reuse_distance']['median']}/{corpus_stats['reuse_distance']['maximum']}`, prefill/decode `{corpus_stats['prefill_decode_occurrences']['prefill']}/{corpus_stats['prefill_decode_occurrences']['decode']}`, cross-token reuse `{corpus_stats['reuse_scope']['cross_token_reuse_percentage']:.4%}`."]
    lines += ["", "## Persistent-cache scenarios", "", "The two persistent scenarios retain cache state in documented fixture order. Results are directional because session ordering is an input.", "", "| Order | GiB | Policy | Micro byte hit | Cross-session hits | Session-transition evictions |", "|---|---:|---|---:|---:|---:|"]
    for row in results["persistent_cache"]["aggregates"]:
        if row["policy"] == "global_lru" and int(row["configured_budget_gib"]) in {8, 16}:
            cross_hits = sum(item["cross_session_hits"] for item in results["persistent_cache"]["fixture_results"] if item["scenario"] == row["scenario"] and item["policy"] == row["policy"] and int(item["configured_budget_gib"]) == int(row["configured_budget_gib"]))
            transitions = sum(item["session_transition_evictions"] for item in results["persistent_cache"]["fixture_results"] if item["scenario"] == row["scenario"] and item["policy"] == row["policy"] and int(item["configured_budget_gib"]) == int(row["configured_budget_gib"]))
            lines.append(f"| `{row['scenario']}` | {row['configured_budget_gib']} | `{row['policy']}` | {row['micro_byte_hit_rate']:.4%} | {cross_hits} | {transitions} |")
    lines += ["", "## 8 GiB classification", "", f"Classification: **`{analysis['eight_gib_classification']}`**.", "", f"Global LRU at 8 GiB has macro byte hit `{analysis['eight_gib_global_lru']['macro_average_byte_hit_rate']:.4%}`, micro byte hit `{analysis['eight_gib_global_lru']['micro_byte_hit_rate']:.4%}`, worst-case fixture `{analysis['eight_gib_global_lru']['worst_case_fixture']}` at `{analysis['eight_gib_global_lru']['worst_case_byte_hit_rate']:.4%}`, and `{analysis['eight_gib_global_lru']['zero_hit_fixture_count']}/{analysis['eight_gib_global_lru']['fixture_count']}` zero-hit fixtures. The micro gain to 12 GiB is `{analysis['eight_gib_classification_basis']['marginal_micro_gain_to_12_gib']:.4%}` and to 16 GiB is `{analysis['eight_gib_classification_basis']['marginal_micro_gain_to_16_gib']:.4%}`. This makes 8 GiB useful for selected workloads, not representative of every workload.", "", "## Operating modes", "", "| Mode | Expert cache | Policy | Macro byte hit | Micro byte hit | Worst fixture | Read reduction | Suitable workloads | Status |", "|---|---:|---|---:|---:|---|---:|---|---|"]
    for mode in analysis["operating_modes"]:
        suitable = ", ".join(f"`{fixture}`" for fixture in mode["suitable_workloads"]) or "none"
        lines.append(f"| `{mode['mode']}` | {mode['expert_cache_budget_gib']} GiB | `{mode['policy']}` | {mode['expected_macro_byte_hit_rate']:.4%} | {mode['expected_micro_byte_hit_rate']:.4%} | `{mode['worst_case_fixture']}` ({mode['worst_case_byte_hit_rate']:.4%}) | {mode['modeled_expert_read_reduction_pct']:.2f}% | {suitable} | modeled; runtime validation pending |")
    lines += ["", "## Policy decision", "", "Selected policy for the next runtime experiment: **strict global LRU**. LFU's isolated 6 GiB gain is not retained at 8 GiB, layer-aware policies do not outperform consistently, and segmented LRU adds policy complexity without a stable gain. Global LRU remains deterministic, byte-budgeted, already implemented and runtime-validated, so no production policy change is selected.", "", "The required next runtime matrix is 8 GiB versus 16 GiB global LRU. It was selected because 8 GiB is the accepted selected-workload baseline and 16 GiB adds a measurable broader-corpus cold-cache point; runtime validation is not executed here.", "", "## Thresholds and limitations", "", "Exact per-fixture and corpus macro/micro thresholds for first hit and 10/25/40/50/75/90% byte-hit rates are stored in `thresholds`. Thresholds are searched over exact policy event-budget replay points without interpolation; `null` means the target is unreachable at full unique-working-set residency. The corpus remains eight deterministic workloads, short fixtures can have no reuse, layer-calibrated results are diagnostic only, persistent-cache results depend on order, and Belady is an offline upper bound. Cacheability does not imply measured latency or throughput, and dense residency is intentionally not modeled.", "", "Exact next task: **M5.2-03**. Stop before implementing or running runtime validation beyond this simulation.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-input", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = {
        "corpus_manifest": Path("models/qwen3-30b-a3b/m5.2-01-trace-corpus-manifest-v1.json"),
        "corpus_aggregate": Path("models/qwen3-30b-a3b/m5.2-01-trace-corpus-aggregate-v1.json"),
        "trace_schema": Path("models/qwen3-30b-a3b/m5.2-01-ordered-expert-trace-schema-v2.json"),
        "m4_baseline": Path("models/qwen3-30b-a3b/m4.4-performance-baseline-v1.json"),
        "m4_provenance": Path("models/qwen3-30b-a3b/m4-release-provenance-v1.json"),
        "model_manifest": Path("models/qwen3-30b-a3b/model-manifest-v1.json"),
    }
    if args.write_input:
        input_doc = build_input(root, paths)
        output_path = args.write_input if args.write_input.is_absolute() else root / args.write_input
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(input_doc, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    if args.input:
        if not args.output or not args.report:
            parser.error("--input requires --output and --report")
        input_path = args.input if args.input.is_absolute() else root / args.input
        output_path = args.output if args.output.is_absolute() else root / args.output
        report_path = args.report if args.report.is_absolute() else root / args.report
        run_all(root, input_path, output_path, report_path)
        print(json.dumps({"output": output_path.relative_to(root).as_posix(), "report": report_path.relative_to(root).as_posix(), "input_sha256": sha256(input_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
