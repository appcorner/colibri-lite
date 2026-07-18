import json
import unittest
from pathlib import Path

from scripts.simulate_m5_2_corpus_cache import (
    EXPECTED_M52_FIXTURES,
    GIB,
    LAYERS,
    PAYLOAD_BYTES,
    PolicyCache,
    candidate_budgets,
    charge_for,
    make_layer_quotas,
    run_cache,
    validate_corpus,
    validate_result_invariants,
)


def record(key: str, layer: int = 0, payload: int = 1) -> dict:
    expert = int(key.rsplit(".", 1)[-1])
    return {
        "layer_expert_key": key,
        "layer_index": layer,
        "expert_id": expert,
        "payload_bytes": payload,
    }


class M52CachePolicyTests(unittest.TestCase):
    def test_global_lru_variable_size_budget_and_oversized_entry(self):
        records = [record("layer.0.expert.0", payload=3), record("layer.0.expert.1", payload=4)]
        cache, _ = run_cache(records, 3, "global_lru", {layer: 1 for layer in range(LAYERS)})
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.loads, 2)
        self.assertEqual(cache.oversized_entry_events, 1)
        self.assertLessEqual(cache.resident_bytes, 3)

    def test_global_lru_known_eviction(self):
        records = [
            record("layer.0.expert.0"),
            record("layer.0.expert.1"),
            record("layer.0.expert.0"),
        ]
        one, _ = run_cache(records, 1, "global_lru", {layer: 1 for layer in range(LAYERS)})
        two, _ = run_cache(records, 2, "global_lru", {layer: 1 for layer in range(LAYERS)})
        self.assertEqual(one.hits, 0)
        self.assertEqual(two.hits, 1)
        self.assertEqual(two.evictions, 0)

    def test_architecture_layer_partition_does_not_borrow(self):
        payload = 2
        budget = LAYERS * payload
        quotas = make_layer_quotas(budget, "layer_lru_architecture", {layer: 1 for layer in range(LAYERS)})
        self.assertEqual(quotas[0], payload)
        self.assertEqual(quotas[1], payload)
        records = [record("layer.0.expert.0", layer=0, payload=payload), record("layer.1.expert.1", layer=1, payload=payload)]
        cache, _ = run_cache(records, budget, "layer_lru_architecture", {layer: 1 for layer in range(LAYERS)})
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.resident_bytes, 2 * payload)
        self.assertEqual(cache.partition_rejection_events, 0)

    def test_frequency_lfu_tie_breaks_by_recency(self):
        records = [
            record("layer.0.expert.0"),
            record("layer.0.expert.1"),
            record("layer.0.expert.2"),
            record("layer.0.expert.1"),
        ]
        cache, _ = run_cache(records, 2, "frequency_lfu", {layer: 1 for layer in range(LAYERS)})
        self.assertEqual(cache.hit_flags, [False, False, False, True])
        self.assertEqual(cache.loads, 3)

    def test_segmented_lru_promotes_on_hit(self):
        records = [
            record("layer.0.expert.0"),
            record("layer.0.expert.1"),
            record("layer.0.expert.0"),
            record("layer.0.expert.2"),
        ]
        cache, _ = run_cache(records, 2, "segmented_lru", {layer: 1 for layer in range(LAYERS)})
        self.assertEqual(cache.hit_flags, [False, False, True, False])
        self.assertEqual(cache.hits, 1)

    def test_belady_is_upper_bound_on_known_sequence(self):
        records = [
            record("layer.0.expert.0"),
            record("layer.0.expert.1"),
            record("layer.0.expert.2"),
            record("layer.0.expert.0"),
            record("layer.0.expert.1"),
            record("layer.0.expert.2"),
        ]
        lru, _ = run_cache(records, 2, "global_lru", {layer: 1 for layer in range(LAYERS)})
        belady, _ = run_cache(records, 2, "belady", {layer: 1 for layer in range(LAYERS)})
        self.assertGreaterEqual(belady.hits, lru.hits)
        self.assertEqual(belady.hit_flags, [False, False, False, True, False, True])

    def test_persistent_session_hit_and_budget_candidates(self):
        records = [record("layer.0.expert.0"), record("layer.0.expert.0")]
        cache, _ = run_cache(records, 1, "global_lru", {layer: 1 for layer in range(LAYERS)}, [0, 1])
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.cross_session_hits, 1)
        candidates = candidate_budgets(records, "global_lru", {layer: 1 for layer in range(LAYERS)})
        self.assertEqual(candidates, [0, 1])

    def test_corpus_and_result_evidence_reconcile(self):
        root = Path(__file__).resolve().parents[1]
        input_path = root / "models/qwen3-30b-a3b/m5.2-02-simulation-input-v1.json"
        result_path = root / "models/qwen3-30b-a3b/m5.2-02-cache-simulation-results-v1.json"
        input_doc = json.loads(input_path.read_text(encoding="utf-8"))
        validated = validate_corpus(root, input_doc)
        self.assertEqual(list(validated["traces"]), EXPECTED_M52_FIXTURES)
        self.assertEqual(sum(len(trace["records"]) for trace in validated["traces"].values()), 11520)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result_invariants(result)
        self.assertEqual(result["budgets_binary_gib"], [1, 2, 4, 6, 8, 12, 16, 24, 32, 48])
        self.assertTrue(result["validation"]["replay_accounting_invariants_validated"])
        self.assertEqual(result["artifact_root_sha256"], "f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2")


if __name__ == "__main__":
    unittest.main()
