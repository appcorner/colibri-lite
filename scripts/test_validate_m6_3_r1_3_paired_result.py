import copy
import unittest

from validate_m6_3_r1_3_paired_result import (
    BINARY_SHA256,
    CANDIDATE_SHA256,
    CANDIDATE_PEAK_BYTES,
    CANDIDATE_VERIFY_BYTES,
    CONTRACT_SHA256,
    CONDITIONS,
    METRICS,
    PAIR_ORDER,
    RAM_BYTES,
    summarize,
    validate,
)


def sample(condition, pair, order_index, mode):
    candidate = mode == "candidate"
    total = 1000 + (100 if candidate else 200)
    return {
        "condition": condition, "pair": pair, "order_index": order_index, "mode": mode,
        "run_directory": "x", "exit_code": 0, "memory_samples": 2,
        "process_wall_seconds": 2.0, "working_set_peak_bytes": 1000 if candidate else 1100,
        "private_bytes_peak": 900, "physical_read_bytes": 100 if candidate else 200,
        "physical_reads_per_token": 50.0 if candidate else 100.0, "trace_sha256": "0" * 64,
        "etw_status": "correlated", "setup_seconds": 0.5, "timed_wall_seconds": 1.0,
        "ttft_seconds": 0.4 if candidate else 0.5, "prefill_tokens_per_second": 2.5 if candidate else 2.0,
        "decode_tokens_per_second": 3.0 if candidate else 2.0, "final_argmax": 0,
        "dense_logical_bytes": 1000, "f32_expert_logical_bytes": 100 if candidate else 200,
        "candidate_logical_bytes": 100 if candidate else 0,
        "total_logical_bytes": 1200 if candidate else 1200, "logical_bytes_per_token": 600.0,
        "cache_hits": 0, "cache_misses": 1, "cache_loads": 1, "cache_evictions": 0,
        "f32_cache_peak_resident_bytes": 100, "candidate_peak_packed_expert_bytes": CANDIDATE_PEAK_BYTES if candidate else 0,
        "candidate_verification_bytes": CANDIDATE_VERIFY_BYTES if candidate else 0,
        "kv_cache_bytes": 100, "vram_bytes": 0,
    }


def document():
    samples = []
    for condition in CONDITIONS:
        for pair, modes in enumerate(PAIR_ORDER, start=1):
            for order_index, mode in enumerate(modes, start=1):
                samples.append(sample(condition, pair, order_index, mode))
    return {
        "schema": "m6.3-r1.3-paired-samples-v1",
        "contract_sha256": CONTRACT_SHA256,
        "binary_sha256": BINARY_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "samples": samples,
    }


class PairedValidatorTests(unittest.TestCase):
    def test_accepts_valid_20_sample_set(self):
        self.assertEqual(validate(document()), [])

    def test_rejects_wrong_order(self):
        doc = document(); doc["samples"][0], doc["samples"][1] = doc["samples"][1], doc["samples"][0]
        self.assertIn("sample order mismatch", validate(doc))

    def test_rejects_uncorrelated_etw(self):
        doc = document(); doc["samples"][3]["etw_status"] = "not_measured"
        self.assertTrue(any("ETW status" in item for item in validate(doc)))

    def test_rejects_ram_ceiling(self):
        doc = document(); doc["samples"][0]["working_set_peak_bytes"] = RAM_BYTES + 1
        self.assertTrue(any("RAM ceiling" in item for item in validate(doc)))

    def test_rejects_candidate_identity_accounting(self):
        doc = document(); candidate = next(s for s in doc["samples"] if s["mode"] == "candidate")
        candidate["candidate_verification_bytes"] = 0
        self.assertTrue(any("candidate verification" in item for item in validate(doc)))

    def test_rejects_candidate_pair_memory_overage(self):
        doc = document(); candidate = next(s for s in doc["samples"] if s["mode"] == "candidate")
        candidate["working_set_peak_bytes"] = 10_000_000
        self.assertTrue(any("candidate memory accounting" in item for item in validate(doc)))

    def test_directional_win_requires_all_five_pairs(self):
        doc = document()
        summary = summarize(doc)
        self.assertTrue(summary["runtime_cache_cold"]["decode_tokens_per_second"]["directional_win"])
        candidate = next(s for s in doc["samples"] if s["condition"] == "runtime_cache_cold" and s["pair"] == 3 and s["mode"] == "candidate")
        candidate["decode_tokens_per_second"] = 1.0
        self.assertFalse(summarize(doc)["runtime_cache_cold"]["decode_tokens_per_second"]["directional_win"])

    def test_reporting_metrics_cover_preregistered_performance_fields(self):
        self.assertEqual(
            set(METRICS),
            {
                "process_wall_seconds", "setup_seconds", "timed_wall_seconds",
                "ttft_seconds", "prefill_tokens_per_second", "decode_tokens_per_second",
                "working_set_peak_bytes", "private_bytes_peak", "total_logical_bytes",
                "physical_read_bytes",
            },
        )


if __name__ == "__main__":
    unittest.main()
